#!/usr/bin/env python3
"""Severity of each degraded reference, measured on all 85 cases.

For every case and degraded scenario, the agreement between GT_minus_<scenario>(case) and
GT_star(case): clDice, NSD@0.5, NSD@2, HD95, Betti-0 error and volume delta. CPU only, on
the masks already on disk (no inference). Uses the same metric code
(prgt.metrics.PredictionFeatures / score_prediction) and the same bounding-box crop
(prgt.metrics.union_bbox) as the main pipeline.

Sign convention: GT_minus plays the "prediction", GT_star the "reference" in
score_prediction (as in clDice(GT-, GT*) in the paper), so
volume_delta = (vol(GT_minus) - vol(GT_star)) / vol(GT_star).

Output : results/reference_severity.csv, one row per (case, scenario):
  case, scenario, cldice_ref_vs_star, nsd05_ref_vs_star, nsd_ref_vs_star,
  hd95_ref_vs_star, betti0_ref_vs_star, dV_ref_vs_star, git_commit, seed, timestamp

    python analysis/reference_severity.py --limit 3        # timing on 3 cases first
    python analysis/reference_severity.py --workers 4      # full run (idempotent upsert)
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import nibabel as nib
import numpy as np
import pandas as pd

from prgt import results_store as rs
from prgt.metrics import PredictionFeatures, score_prediction, union_bbox

SCENARIOS = ["GT_minus_omission", "GT_minus_drift_neg", "GT_minus_drift_pos"]
OUT_CSV = "reference_severity.csv"
KEY = ["case", "scenario"]


def _dataset_dir(nnunet_data_root: str) -> str:
    return os.path.join(nnunet_data_root, "nnUNet_raw", "Dataset100_PARSE")


def _scenario_dir(dataset_dir: str, scenario: str) -> str:
    return os.path.join(dataset_dir, f"labelsTr_{scenario}")


def existing_keys(results_dir: str) -> set[tuple]:
    path = os.path.join(results_dir, OUT_CSV)
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(zip(df["case"], df["scenario"]))


def _compute_one(args: tuple) -> dict | None:
    case, scenario, star_path, minus_path = args
    if not (os.path.exists(star_path) and os.path.exists(minus_path)):
        return None
    star_nii = nib.load(star_path)
    spacing = star_nii.header.get_zooms()[:3]
    gt_star = np.asarray(star_nii.get_fdata()) > 0
    gt_minus = np.asarray(nib.load(minus_path).get_fdata()) > 0

    bbox = union_bbox([gt_star, gt_minus], margin=2)
    pf = PredictionFeatures(gt_minus[bbox].astype(np.float32), spacing=spacing)
    m = score_prediction(pf, gt_star[bbox].astype(np.float32))

    return {
        "case": case, "scenario": scenario,
        "cldice_ref_vs_star": m.get("cldice"),
        "nsd05_ref_vs_star": m.get("nsd05"),
        "nsd_ref_vs_star": m.get("nsd"),
        "hd95_ref_vs_star": m.get("hd95"),
        "betti0_ref_vs_star": m.get("betti0"),
        "dV_ref_vs_star": m.get("volume_delta"),
    }


def build_tasks(dataset_dir: str, results_dir: str, limit: int | None) -> list[tuple]:
    star_dir = os.path.join(dataset_dir, "labelsTr")
    cases = sorted(os.path.basename(p) for p in glob.glob(os.path.join(star_dir, "*.nii.gz")))
    if limit:
        cases = cases[:limit]
    done = existing_keys(results_dir)
    tasks = []
    for case in cases:
        star_path = os.path.join(star_dir, case)
        for scenario in SCENARIOS:
            if (case, scenario) in done:
                continue
            minus_path = os.path.join(_scenario_dir(dataset_dir, scenario), case)
            tasks.append((case, scenario, star_path, minus_path))
    return tasks


def upsert(results_dir: str, rows: list[dict], seed: int) -> None:
    path = os.path.join(results_dir, OUT_CSV)
    current = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=KEY)
    incoming = pd.DataFrame(rows)
    incoming["git_commit"] = rs.git_commit()
    incoming["seed"] = seed
    incoming["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    combined = pd.concat([current, incoming], ignore_index=True)
    combined = combined.drop_duplicates(subset=KEY, keep="last").sort_values(KEY)
    os.makedirs(results_dir, exist_ok=True)
    combined.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description="Severity of the degraded references (85 cases)")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--nnunet_data_root", default="nnUNet_data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="limit the number of CASES (not tasks), for a timing estimate")
    args = ap.parse_args()

    dataset_dir = _dataset_dir(args.nnunet_data_root)
    tasks = build_tasks(dataset_dir, args.results_dir, args.limit)
    if not tasks:
        print("[SKIP] nothing to do (already computed, or --limit too small).")
        return

    print(f"[START] {len(tasks)} (case x scenario) tasks, workers={args.workers}")
    t0 = time.time()
    rows = []
    if args.workers > 1:
        with mp.Pool(args.workers, maxtasksperchild=20) as pool:
            for i, res in enumerate(pool.imap_unordered(_compute_one, tasks), 1):
                if res is not None:
                    rows.append(res)
                if i % 20 == 0 or i == len(tasks):
                    dt = time.time() - t0
                    print(f"  [{i}/{len(tasks)}] {dt:.1f}s "
                          f"({dt / i:.2f}s/task, ETA {dt / i * (len(tasks) - i):.0f}s)")
    else:
        for i, task in enumerate(tasks, 1):
            res = _compute_one(task)
            if res is not None:
                rows.append(res)
            if i % 10 == 0 or i == len(tasks):
                dt = time.time() - t0
                print(f"  [{i}/{len(tasks)}] {dt:.1f}s "
                      f"({dt / i:.2f}s/task, ETA {dt / i * (len(tasks) - i):.0f}s)")

    if rows:
        upsert(args.results_dir, rows, args.seed)
    dt = time.time() - t0
    print(f"[OK] {len(rows)}/{len(tasks)} rows written to "
          f"{os.path.join(args.results_dir, OUT_CSV)} in {dt:.1f}s "
          f"({dt / max(len(tasks), 1):.2f}s/task on average)")


if __name__ == "__main__":
    main()
