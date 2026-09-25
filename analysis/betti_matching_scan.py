#!/usr/bin/env python3
"""P1.1 -- Betti-matching sur le tier A (déploiement progressif : 20 cas puis 85).

Répond à la question du brief : l'inversion de signe observée avec |ΔN| (Betti0
classique) disparaît-elle avec Betti-matching (métrique consciente de la
correspondance) ?

Coût mesuré (triage, 3 cas, crop bbox margin=15, seuil de persistance=2 voxels) :
~50-70s/paire (model,scenario,case). Pour 20 cas x 3 modèles x 4 scénarios = 240
paires, largement hors du budget interactif -- lancé en tâche de fond,
multiprocessing (comme les autres scripts CPU-lourds du projet).

Usage
-----
    # sous-ensemble de 20 cas (seed=42), tous modèles x scénarios
    python analysis/betti_matching_scan.py --n_cases 20 --workers 6

    # extension aux 85 cas une fois la cohérence vérifiée sur les 20
    python analysis/betti_matching_scan.py --n_cases 85 --workers 6
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import results_store as rs  # noqa: E402

TRAINER_DIR = {
    "M0_Star": "nnUNetTrainerStd",
    "M1_Omission": "nnUNetTrainerDegradedOmissionOnly",
    "M2_Drift_mu0": "nnUNetTrainerDriftMu0",
}
SCENARIO_DIR = {
    "GT_star": "labelsTr",
    "GT_minus_omission": "labelsTr_GT_minus_omission",
    "GT_minus_drift_neg": "labelsTr_GT_minus_drift_neg",
    "GT_minus_drift_pos": "labelsTr_GT_minus_drift_pos",
}
SCENARIOS = list(SCENARIO_DIR)
OUT_CSV = "betti_matching.csv"
KEY = ["model", "fold", "case", "scenario"]
CROP_MARGIN = 15
PERSISTENCE_THRESHOLD = 2.0


def _dataset_root(nnunet_data_root: str) -> str:
    return os.path.join(nnunet_data_root, "nnUNet_raw", "Dataset100_PARSE")


def _pred_path(nnunet_data_root: str, model: str, fold: int, case: str) -> str:
    trainer = TRAINER_DIR[model]
    return os.path.join(
        nnunet_data_root, "nnUNet_results", "Dataset100_PARSE",
        f"{trainer}__nnUNetPlans__3d_fullres", f"fold_{fold}", "validation", case,
    )


def select_cases(results_dir: str, n_cases: int, seed: int) -> list[str]:
    df = rs.load(results_dir)
    df = df[(df["eval_kind"] == "oof") & (df["model"].isin(TRAINER_DIR))]
    all_cases = sorted(df["case"].unique())
    if n_cases >= len(all_cases):
        return all_cases
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(all_cases, size=n_cases, replace=False).tolist())


def unique_predictions(results_dir: str, cases: list[str]) -> list[tuple]:
    df = rs.load(results_dir)
    df = df[(df["eval_kind"] == "oof") & (df["model"].isin(TRAINER_DIR))
            & (df["case"].isin(cases))]
    uniq = df.drop_duplicates(subset=["model", "fold", "case"])[["model", "fold", "case"]]
    return sorted(tuple(r) for r in uniq.itertuples(index=False, name=None))


def existing_keys(results_dir: str) -> set[tuple]:
    path = os.path.join(results_dir, OUT_CSV)
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(zip(df["model"], df["fold"], df["case"], df["scenario"]))


def _process_one(task: tuple) -> list[dict]:
    model, fold, case, nnunet_data_root = task
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from betti_matching_wrapper import betti_matching_error_from_paths

    pred_path = _pred_path(nnunet_data_root, model, fold, case)
    if not os.path.exists(pred_path):
        return []
    dataset_root = _dataset_root(nnunet_data_root)

    rows = []
    for scenario in SCENARIOS:
        gt_path = os.path.join(dataset_root, SCENARIO_DIR[scenario], case)
        if not os.path.exists(gt_path):
            continue
        try:
            res = betti_matching_error_from_paths(
                pred_path, gt_path, crop_margin=CROP_MARGIN,
                persistence_threshold=PERSISTENCE_THRESHOLD)
        except Exception as e:
            res = {"bm_error": np.nan, "bm_unmatched_pred": np.nan,
                  "bm_unmatched_ref": np.nan, "bm_n_matched": np.nan,
                  "bm_error_raw_unfiltered": np.nan, "_seconds": np.nan,
                  "_error": str(e)}
        rows.append({"model": model, "fold": fold, "case": case, "scenario": scenario, **res})
    return rows


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
    ap = argparse.ArgumentParser(description="P1.1 -- Betti-matching, déploiement progressif")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--nnunet_data_root", default="nnUNet_data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_cases", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cases = select_cases(args.results_dir, args.n_cases, args.seed)
    preds = unique_predictions(args.results_dir, cases)
    done = existing_keys(args.results_dir)
    done_preds = {(m, f, c) for (m, f, c, s) in done
                  if all((m, f, c, s2) in done for s2 in SCENARIOS)}
    todo = [(m, f, c, args.nnunet_data_root) for (m, f, c) in preds if (m, f, c) not in done_preds]

    if not todo:
        print(f"[SKIP] rien à faire pour ces {len(cases)} cas (déjà calculé).")
        return

    print(f"[START] {len(cases)} cas, {len(todo)} prédictions uniques à traiter "
          f"({len(todo) * len(SCENARIOS)} paires), workers={args.workers}, "
          f"crop_margin={CROP_MARGIN}, persistence_threshold={PERSISTENCE_THRESHOLD}")
    t0 = time.time()
    all_rows = []
    with mp.Pool(args.workers, maxtasksperchild=5) as pool:
        for i, rows in enumerate(pool.imap_unordered(_process_one, todo), 1):
            all_rows.extend(rows)
            if i % 5 == 0 or i == len(todo):
                dt = time.time() - t0
                print(f"  [{i}/{len(todo)}] {dt:.1f}s ({dt / i:.2f}s/pred, "
                      f"ETA {dt / i * (len(todo) - i):.0f}s)")
            if i % 20 == 0:
                upsert(args.results_dir, all_rows, args.seed)
                all_rows = []

    if all_rows:
        upsert(args.results_dir, all_rows, args.seed)
    dt = time.time() - t0
    print(f"[OK] terminé en {dt:.1f}s -> {os.path.join(args.results_dir, OUT_CSV)}")


if __name__ == "__main__":
    main()
