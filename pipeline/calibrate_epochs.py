#!/usr/bin/env python3
"""Choice of num_epochs from the convergence of the TOPOLOGY metrics (clDice, Betti-0).

Scores the checkpoint snapshots (``checkpoint_ep<N>.pth``, saved at fixed epochs by
nnUNetTrainerCalib) out of fold: for each snapshot, the validation split of the fold is
predicted with that checkpoint, then clDice and Betti-0 are computed against the clean
reference (GT_star) and the degraded one.

Goal: locate the plateau of the topology metrics (which converge later than the region
Dice) to choose num_epochs on evidence, with the curve to show for review.

Prediction needs nnU-Net and the checkpoints on disk, so this runs on the GPU node,
after the calibration run (``pipeline/orchestrator.py --calibrate``).

    python pipeline/calibrate_epochs.py --config configs/experiment_config.yaml
"""

import argparse
import glob
import json
import os
import tempfile

import nibabel as nib
import pandas as pd
import yaml
from tqdm import tqdm

from collect_metrics import resolve_dataset_dir, resolve_results_dir, scenario_gt_dir
from prgt.metrics import PredictionFeatures, score_prediction

PLANS = "nnUNetPlans"


def _device(override=None):
    if override:
        return override
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _val_ids(nnunet_preprocessed: str, dataset_id: int, fold: int) -> list[str]:
    """Case IDs of the validation split of a fold (from splits_final.json)."""
    base = sorted(glob.glob(os.path.join(nnunet_preprocessed, f"Dataset{dataset_id:03d}_*")))
    if not base:
        return []
    splits = json.load(open(os.path.join(base[0], "splits_final.json")))
    return list(splits[fold]["val"])


def _epoch_of(checkpoint_path: str) -> int:
    """Actual epoch of a snapshot (read from the checkpoint; falls back on the file name)."""
    try:
        import torch
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return int(ck.get("current_epoch", -1))
    except Exception:
        base = os.path.basename(checkpoint_path)
        digits = "".join(c for c in base if c.isdigit())
        return int(digits) if digits else -1


def _predict_val(model_dir, fold, checkpoint_name, images_tr, val_ids, out_dir, device):
    """Predict the validation cases with ONE given checkpoint (nnU-Net API)."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    input_lists = [[os.path.join(images_tr, f"{cid}_0000.nii.gz")] for cid in val_ids]
    output_files = [os.path.join(out_dir, f"{cid}.nii.gz") for cid in val_ids]

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=True, device=torch.device(device),
        verbose=False, allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir, use_folds=(fold,), checkpoint_name=checkpoint_name,
    )
    predictor.predict_from_files(
        input_lists, output_files, save_probabilities=False, overwrite=True,
        num_processes_preprocessing=2, num_processes_segmentation_export=2,
    )
    return output_files


def recommend(df: pd.DataFrame, tol: float) -> dict:
    """Smallest epoch whose mean clDice (GT_star) reaches (1 - tol) * max."""
    star = df[df["scenario"] == "GT_star"]
    if star.empty:
        return {}
    by_ep = star.groupby("epoch")["cldice"].mean().sort_index()
    target = (1 - tol) * by_ep.max()
    plateau = by_ep[by_ep >= target]
    rec = int(plateau.index.min()) if not plateau.empty else int(by_ep.index.max())
    return {"recommended_num_epochs": rec, "max_cldice": round(float(by_ep.max()), 4),
            "target": round(float(target), 4)}


def plot_curves(df: pd.DataFrame, out_path: str, rec: dict) -> None:
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"GT_star": "#4CAF50", "GT_minus_test": "#F44336"}

    for metric, ax, lab, lower_better in (
        ("cldice", axes[0], "clDice (higher is better)", False),
        ("betti0", axes[1], "Betti-0 error |delta components| (lower is better)", True),
    ):
        for sc, g in df.groupby("scenario"):
            agg = g.groupby("epoch")[metric].agg(["mean", "std"]).sort_index()
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], marker="o",
                        capsize=4, color=colors.get(sc, None), label=sc)
        ax.set_xlabel("Epoch"); ax.set_ylabel(lab)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        if rec.get("recommended_num_epochs"):
            ax.axvline(rec["recommended_num_epochs"], color="black", ls="--", lw=1.2,
                       label=f"reco={rec['recommended_num_epochs']}")
    axes[0].set_title("Topology convergence: clDice")
    axes[1].set_title("Topology convergence: Betti-0")
    plt.suptitle("Schedule calibration (out of fold): choice of num_epochs",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Choose num_epochs from topology convergence (OOF)")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep-preds", action="store_true", help="keep the snapshot predictions")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    cal = config["calibration"]
    cfg = config["training"]["configurations"][0]
    fold = cal["fold"]
    dataset_id = cal["dataset_id"]
    trainer = cal["trainer"]
    scenarios = config["evaluation"]["scenarios"]

    nnunet_raw = os.environ.get("nnUNet_raw", "nnUNet_data/nnUNet_raw")
    nnunet_results = os.environ.get("nnUNet_results", "nnUNet_data/nnUNet_results")
    nnunet_preprocessed = os.environ.get("nnUNet_preprocessed", "nnUNet_data/nnUNet_preprocessed")

    model_dir = resolve_results_dir(nnunet_results, dataset_id, trainer, cfg)
    if model_dir is None:
        print(f"[SKIP] no results for {trainer} (dataset {dataset_id}).")
        return
    fold_dir = os.path.join(model_dir, f"fold_{fold}")
    snapshots = sorted(glob.glob(os.path.join(fold_dir, "checkpoint_ep*.pth")))
    if not snapshots:
        print(f"[SKIP] no checkpoint_ep*.pth snapshot in {fold_dir}.")
        print("       Run first: pipeline/orchestrator.py --calibrate")
        return

    clean_dir = resolve_dataset_dir(nnunet_raw, config["dataset"]["id"])
    images_tr = os.path.join(resolve_dataset_dir(nnunet_raw, dataset_id), "imagesTr")
    val_ids = _val_ids(nnunet_preprocessed, dataset_id, fold)
    device = _device(args.device)
    print(f"Calibration: {len(snapshots)} snapshots x {len(val_ids)} validation cases (fold {fold}), device={device}")

    rows = []
    for ckpt in tqdm(snapshots, desc="Snapshots", unit="ckpt"):
        ep = _epoch_of(ckpt)
        ckpt_name = os.path.basename(ckpt)
        tmp = tempfile.mkdtemp(prefix=f"calib_ep{ep}_")
        tqdm.write(f"\n[epoch {ep}] predicting the validation cases ({ckpt_name})...")
        _predict_val(model_dir, fold, ckpt_name, images_tr, val_ids, tmp, device)
        # The prediction's skeleton and distance maps are computed once per case
        # (PredictionFeatures) and reused for every reference.
        for cid in tqdm(val_ids, desc=f"Metrics ep{ep}", unit="case", leave=False):
            pred = os.path.join(tmp, f"{cid}.nii.gz")
            if not os.path.exists(pred):
                continue
            pred_nii = nib.load(pred)
            pf = PredictionFeatures(pred_nii.get_fdata(), spacing=pred_nii.header.get_zooms()[:3])
            for sc in scenarios:
                gt = os.path.join(scenario_gt_dir(clean_dir, "labelsTr", sc), f"{cid}.nii.gz")
                if not os.path.exists(gt):
                    continue
                m = score_prediction(pf, nib.load(gt).get_fdata())
                rows.append({"trainer": trainer, "fold": fold, "epoch": ep,
                             "scenario": sc["name"], "case": f"{cid}.nii.gz", **m})
        if not args.keep_preds:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    if not rows:
        print("No metric computed.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "calibration.csv")
    df.to_csv(csv_path, index=False)

    rec = recommend(df, float(cal.get("plateau_tol", 0.01)))
    plot_curves(df, os.path.join(args.results_dir, "figures", "calibration_topology.png"), rec)

    print(f"\n{'='*60}\nCALIBRATION SUMMARY (mean GT_star clDice per epoch)\n{'='*60}")
    summary = (df[df["scenario"] == "GT_star"].groupby("epoch")[["cldice", "betti0"]]
               .mean().round(4))
    print(summary.to_string())
    if rec:
        print(f"\n-> plateau at (1-{cal.get('plateau_tol',0.01)}) * max(clDice={rec['max_cldice']}) "
              f"= {rec['target']}")
        print(f"-> RECOMMENDED num_epochs: {rec['recommended_num_epochs']}")
    print(f"\nDetails: {csv_path}")
    print(f"Curve  : {os.path.join(args.results_dir, 'figures', 'calibration_topology.png')}")


if __name__ == "__main__":
    main()
