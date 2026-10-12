#!/usr/bin/env python3
"""Multi-process version of collect_metrics.py: same scores, distributed loop.

Uses the same metric code (prgt.metrics.PredictionFeatures, score_prediction) and the same
path resolution as collect_metrics.py; only the OOF loop (one task = one (model, fold,
case)) is distributed over a process pool. Each prediction and its references are first
cropped to their common bounding box (exact, see prgt.metrics.union_bbox), which is where
most of the speed-up comes from. The master table is the same as with the serial version:
same keys, same idempotent upsert.

Memory: each PredictionFeatures (~1.5 GB on PARSE) and its references live in one task
and are freed at its end; ``--max-tasks-per-child`` recycles workers to return fragmented
memory to the OS. Sized for 24 GB of RAM (4 workers ~ 16 GB peak).

    python pipeline/collect_metrics_parallel.py --config configs/experiment_config.yaml \
        --output results --workers 4
"""

from __future__ import annotations

import argparse
import glob
import os
import time

# One BLAS thread per process, set BEFORE numpy is imported: parallelism comes from the
# pool (avoids oversubscription and fork + threads deadlocks).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp

import nibabel as nib
import numpy as np
import yaml

from collect_metrics import (checkpoint_info, resolve_dataset_dir, resolve_results_dir,
                             scenario_gt_dir)
from prgt import results_store as rs
from prgt.metrics import PredictionFeatures, score_prediction, union_bbox


def _process_case(task):
    """Worker: score one OOF prediction against all its missing references.

    ``task`` is fully picklable (dicts, str, float). Returns master-table rows ready to
    upsert.
    """
    model, fold, case, pred_path, mtime, clean_dir, scenarios, num_epochs = task
    pred_nii = nib.load(pred_path)
    spacing = pred_nii.header.get_zooms()[:3]
    pred = np.asarray(pred_nii.get_fdata()) > 0

    gts = []
    for sc in scenarios:
        gt_path = os.path.join(scenario_gt_dir(clean_dir, "labelsTr", sc), case)
        if os.path.exists(gt_path):
            gts.append((sc, np.asarray(nib.load(gt_path).get_fdata()) > 0))
    if not gts:
        return []

    bbox = union_bbox([pred] + [g for _, g in gts], margin=2)
    pf = PredictionFeatures(pred[bbox].astype(np.float32), spacing=spacing)
    rows = []
    for sc, g in gts:
        metrics = score_prediction(pf, g[bbox].astype(np.float32))
        rows.append(rs.make_row(
            model=model["name"], trainer=model["trainer"],
            dataset=model["dataset_id"], fold=fold, scenario=sc["name"],
            case=case, eval_kind="oof", metrics=metrics,
            epochs=num_epochs, checkpoint_mtime=mtime))
    return rows


def build_tasks(config, clean_dir, nnunet_results, cfg, scenarios, done, num_epochs):
    """Tasks (model, fold, case) still to compute, skipping the keys in ``done``."""
    tasks = []
    for model in config["experiment"]["models"]:
        rdir = resolve_results_dir(nnunet_results, model["dataset_id"], model["trainer"], cfg)
        if rdir is None:
            continue
        for fold_dir in sorted(glob.glob(os.path.join(rdir, "fold_*"))):
            fold = int(os.path.basename(fold_dir).split("_")[1])
            val_dir = os.path.join(fold_dir, "validation")
            _, mtime, finished = checkpoint_info(fold_dir)
            if not finished or not os.path.isdir(val_dir):
                continue
            for pred_path in sorted(glob.glob(os.path.join(val_dir, "*.nii.gz"))):
                case = os.path.basename(pred_path)
                todo = [sc for sc in scenarios
                        if (model["name"], model["trainer"], int(model["dataset_id"]),
                            fold, sc["name"], case, "oof") not in done]
                if todo:
                    tasks.append((model, fold, case, pred_path, mtime, clean_dir,
                                  todo, num_epochs))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Parallel OOF scoring")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tasks-per-child", type=int, default=4,
                        help="recycle each worker after N cases (bounds memory)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    nnunet_raw = os.environ.get("nnUNet_raw", "nnUNet_data/nnUNet_raw")
    nnunet_results = os.environ.get("nnUNet_results", "nnUNet_data/nnUNet_results")
    cfg = config["training"]["configurations"][0]
    num_epochs = config["training"].get("num_epochs")
    scenarios = config["evaluation"]["scenarios"]

    clean_dir = resolve_dataset_dir(nnunet_raw, config["dataset"]["id"])
    if clean_dir is None:
        print(f"ERROR: source dataset not found in {nnunet_raw}")
        return

    df = rs.load(args.output)
    done = rs.existing_keys(df)
    print(f"Master table: {len(df)} rows already present")

    tasks = build_tasks(config, clean_dir, nnunet_results, cfg, scenarios, done, num_epochs)
    if not tasks:
        print("Nothing new to collect.")
        return
    print(f"{len(tasks)} cases to score on {args.workers} workers "
          f"(recycled every {args.max_tasks_per_child} cases)...")

    all_rows, t0, n_done = [], time.time(), 0
    # fork: workers start instantly, no re-import, pages shared copy-on-write.
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers,
                  maxtasksperchild=args.max_tasks_per_child) as pool:
        for rows in pool.imap_unordered(_process_case, tasks, chunksize=1):
            all_rows.extend(rows)
            n_done += 1
            if n_done % 10 == 0 or n_done == len(tasks):
                el = time.time() - t0
                rate = n_done / el if el else 0
                eta = (len(tasks) - n_done) / rate if rate else 0
                print(f"  {n_done}/{len(tasks)} cases · {el:5.0f}s elapsed · ETA {eta:4.0f}s",
                      flush=True)

    n_new = rs.upsert(args.output, all_rows)
    print(f"\n{n_new} rows upserted -> {os.path.join(args.output, rs.CSV_NAME)} "
          f"({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
