#!/usr/bin/env python3
"""Incremental scoring of the predictions into the master table (out-of-fold + test).

Evaluation strategy: out-of-fold (OOF) first, test set as confirmation.

- OOF: at the end of each fold, nnU-Net writes its validation predictions to
  ``<results>/<trainer>__nnUNetPlans__<cfg>/fold_<k>/validation/*.nii.gz`` (the ~17 cases
  that fold did not see). The finished folds together cover the 85 cases, so n grows from
  ~17 to 85 as training progresses.
- Test: if ``predictions/<model_dir>/`` exists (fold ensemble on imagesTs), it is scored
  too, with ``eval_kind=test``.

Every prediction, whatever the model or training dataset, is scored against the clean
reference (``GT_star``) and against each deterministic degraded scenario of
``evaluation.scenarios`` (``GT_minus_omission``, ``GT_minus_drift_*``), matched by file name.

Only keys that are MISSING from the master table are computed, so rerunning after an
interruption is safe and cheap. See collect_metrics_parallel.py for a multi-process
version with the same output.

    python pipeline/collect_metrics.py --config configs/experiment_config.yaml --output results
"""

import argparse
import glob
import os

import nibabel as nib
import yaml

from prgt import results_store as rs
from prgt.metrics import PredictionFeatures, score_prediction

PLANS = "nnUNetPlans"


# ---------------------------------------------------------------------------------------------
# Path resolution (robust to the suffix of the dataset name)
# ---------------------------------------------------------------------------------------------

def resolve_dataset_dir(nnunet_raw: str, dataset_id: int) -> str | None:
    """Raw folder of a dataset, found by ID (glob ``Dataset{ID}_*``)."""
    hits = sorted(glob.glob(os.path.join(nnunet_raw, f"Dataset{dataset_id:03d}_*")))
    return hits[0] if hits else None


def resolve_results_dir(nnunet_results: str, dataset_id: int, trainer: str, cfg: str) -> str | None:
    """nnU-Net results folder of (dataset, trainer, configuration)."""
    base = sorted(glob.glob(os.path.join(nnunet_results, f"Dataset{dataset_id:03d}_*")))
    if not base:
        return None
    path = os.path.join(base[0], f"{trainer}__{PLANS}__{cfg}")
    return path if os.path.isdir(path) else None


def scenario_gt_dir(clean_dataset_dir: str, split: str, scenario: dict) -> str:
    """Folder of the reference labels of a scenario.

    ``split`` is labelsTr or labelsTs. A scenario without degradation uses the clean
    labels; otherwise ``<split>_<name>`` (written by make_eval_references.py).
    """
    if scenario.get("degradation") is None:
        return os.path.join(clean_dataset_dir, split)
    return os.path.join(clean_dataset_dir, f"{split}_{scenario['name']}")


def checkpoint_info(fold_dir: str) -> tuple[int | None, float | None, bool]:
    """(epochs, mtime, finished), from checkpoint_final.pth when it exists."""
    final = os.path.join(fold_dir, "checkpoint_final.pth")
    if os.path.exists(final):
        return None, os.path.getmtime(final), True
    return None, None, False


# ---------------------------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------------------------

def collect_oof(model, clean_dir, results_dir, cfg, scenarios, done, num_epochs):
    """OOF rows of one model: finished folds x scenarios x cases."""
    rdir = resolve_results_dir(results_dir, model["dataset_id"], model["trainer"], cfg)
    if rdir is None:
        return []

    rows = []
    for fold_dir in sorted(glob.glob(os.path.join(rdir, "fold_*"))):
        fold = int(os.path.basename(fold_dir).split("_")[1])
        val_dir = os.path.join(fold_dir, "validation")
        _, mtime, finished = checkpoint_info(fold_dir)
        if not finished or not os.path.isdir(val_dir):
            continue  # unfinished fold: no OOF predictions yet

        for pred_path in sorted(glob.glob(os.path.join(val_dir, "*.nii.gz"))):
            case = os.path.basename(pred_path)
            # Scenarios still missing for this case: the prediction (skeleton, distance
            # maps, ...) is loaded and featurised once for all of them.
            todo = [sc for sc in scenarios
                    if (model["name"], model["trainer"], int(model["dataset_id"]),
                        fold, sc["name"], case, "oof") not in done]
            if not todo:
                continue
            pred_nii = nib.load(pred_path)
            pf = PredictionFeatures(pred_nii.get_fdata(), spacing=pred_nii.header.get_zooms()[:3])
            for sc in todo:
                gt_path = os.path.join(scenario_gt_dir(clean_dir, "labelsTr", sc), case)
                if not os.path.exists(gt_path):
                    continue
                metrics = score_prediction(pf, nib.load(gt_path).get_fdata())
                rows.append(rs.make_row(
                    model=model["name"], trainer=model["trainer"],
                    dataset=model["dataset_id"], fold=fold, scenario=sc["name"],
                    case=case, eval_kind="oof", metrics=metrics,
                    epochs=num_epochs, checkpoint_mtime=mtime,
                ))
    return rows


def collect_test(model, clean_dir, scenarios, done, num_epochs):
    """Test-set rows (fold ensemble) of one model, if predictions/<dir> exists."""
    pred_dir = os.path.join("predictions", _model_pred_dir(model["name"]))
    if not os.path.isdir(pred_dir):
        return []

    rows = []
    for pred_path in sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz"))):
        case = os.path.basename(pred_path)
        todo = [sc for sc in scenarios
                if (model["name"], model["trainer"], int(model["dataset_id"]),
                    -1, sc["name"], case, "test") not in done]
        if not todo:
            continue
        pred_nii = nib.load(pred_path)
        pf = PredictionFeatures(pred_nii.get_fdata(), spacing=pred_nii.header.get_zooms()[:3])
        for sc in todo:
            gt_path = os.path.join(scenario_gt_dir(clean_dir, "labelsTs", sc), case)
            if not os.path.exists(gt_path):
                continue
            metrics = score_prediction(pf, nib.load(gt_path).get_fdata())
            rows.append(rs.make_row(
                model=model["name"], trainer=model["trainer"],
                dataset=model["dataset_id"], fold=-1, scenario=sc["name"],
                case=case, eval_kind="test", metrics=metrics,
                epochs=num_epochs, checkpoint_mtime=None,
            ))
    return rows


def _model_pred_dir(model_name: str) -> str:
    """Test-prediction folder convention: the model name in lower case (M0_Star -> m0_star)."""
    return model_name.lower()


def main():
    parser = argparse.ArgumentParser(description="Incremental OOF + test scoring")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    nnunet_raw = os.environ.get("nnUNet_raw", "nnUNet_data/nnUNet_raw")
    nnunet_results = os.environ.get("nnUNet_results", "nnUNet_data/nnUNet_results")
    cfg = config["training"]["configurations"][0]
    num_epochs = config["training"].get("num_epochs")
    scenarios = config["evaluation"]["scenarios"]

    # The references always come from the clean source dataset (Dataset100).
    clean_dir = resolve_dataset_dir(nnunet_raw, config["dataset"]["id"])
    if clean_dir is None:
        print(f"ERROR: source dataset not found in {nnunet_raw}")
        return

    df = rs.load(args.output)
    done = rs.existing_keys(df)
    print(f"Master table: {len(df)} rows already present")

    all_rows = []
    for model in config["experiment"]["models"]:
        oof = collect_oof(model, clean_dir, nnunet_results, cfg, scenarios, done, num_epochs)
        test = collect_test(model, clean_dir, scenarios, done, num_epochs)
        if oof or test:
            print(f"  {model['name']:<26} +{len(oof)} OOF  +{len(test)} test")
        all_rows.extend(oof)
        all_rows.extend(test)

    if not all_rows:
        print("Nothing new to collect.")
        return

    n_new = rs.upsert(args.output, all_rows)
    print(f"\n{n_new} new rows upserted -> {os.path.join(args.output, rs.CSV_NAME)}")


if __name__ == "__main__":
    main()
