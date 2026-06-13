#!/usr/bin/env python3
"""Calibration de num_epochs — convergence de la TOPOLOGIE (clDice / Betti0).

Évalue les snapshots de checkpoints (``checkpoint_ep<N>.pth``, posés par
nnUNetTrainerCalib aux jalons) en **out-of-fold** : pour chaque snapshot, on
prédit le split de validation du fold avec ce checkpoint précis, puis on calcule
clDice et Betti0 contre la GT propre (``GT_star``) et dégradée (``GT_minus_test``).

But : situer le **plateau des métriques de topologie** (qui convergent plus tard
que le Dice de région) pour choisir num_epochs de façon défendable, courbe à
l'appui pour la revue.

⚠️ La prédiction requiert nnU-Net + le checkpoint sur le disque → s'exécute sur le
cluster, après le run de calibration (``orchestrator.py --calibrate``).

Usage
-----
    python scripts/calibrate.py --config configs/experiment_config.yaml
"""

import argparse
import glob
import json
import os
import sys
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from collect_metrics import resolve_dataset_dir, resolve_results_dir, scenario_gt_dir
from cross_evaluate import PredFeatures, evaluate_pair_cached

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
    """IDs de cas du split de validation d'un fold (depuis splits_final.json)."""
    base = sorted(glob.glob(os.path.join(nnunet_preprocessed, f"Dataset{dataset_id:03d}_*")))
    if not base:
        return []
    splits = json.load(open(os.path.join(base[0], "splits_final.json")))
    return list(splits[fold]["val"])


def _epoch_of(checkpoint_path: str) -> int:
    """Epoch réelle du snapshot (lit le checkpoint ; fallback sur le nom)."""
    try:
        import torch
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return int(ck.get("current_epoch", -1))
    except Exception:
        base = os.path.basename(checkpoint_path)
        digits = "".join(c for c in base if c.isdigit())
        return int(digits) if digits else -1


def _predict_val(model_dir, fold, checkpoint_name, images_tr, val_ids, out_dir, device):
    """Prédit les cas de validation avec UN checkpoint donné (API nnU-Net)."""
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
    """Plus petit epoch dont le clDice moyen (GT_star) atteint (1-tol)·max."""
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
        ("cldice", axes[0], "clDice (↑)", False),
        ("betti0", axes[1], "Betti0 — |Δ comp.| (↓)", True),
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
    axes[0].set_title("Convergence topologique — clDice")
    axes[1].set_title("Convergence topologique — Betti0")
    plt.suptitle("Calibration du schedule (out-of-fold) — choix de num_epochs",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Calibration de num_epochs (topologie OOF)")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep-preds", action="store_true", help="ne pas supprimer les prédictions")
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
        print(f"[SKIP] résultats introuvables pour {trainer} (dataset {dataset_id}).")
        return
    fold_dir = os.path.join(model_dir, f"fold_{fold}")
    snapshots = sorted(glob.glob(os.path.join(fold_dir, "checkpoint_ep*.pth")))
    if not snapshots:
        print(f"[SKIP] aucun snapshot checkpoint_ep*.pth dans {fold_dir}.")
        print("       Lancez d'abord : orchestrator.py --calibrate")
        return

    clean_dir = resolve_dataset_dir(nnunet_raw, config["dataset"]["id"])
    images_tr = os.path.join(resolve_dataset_dir(nnunet_raw, dataset_id), "imagesTr")
    val_ids = _val_ids(nnunet_preprocessed, dataset_id, fold)
    device = _device(args.device)
    print(f"Calibration : {len(snapshots)} snapshots × {len(val_ids)} cas val (fold {fold}), device={device}")

    rows = []
    for ckpt in tqdm(snapshots, desc="Snapshots", unit="ckpt"):
        ep = _epoch_of(ckpt)
        ckpt_name = os.path.basename(ckpt)
        tmp = tempfile.mkdtemp(prefix=f"calib_ep{ep}_")
        tqdm.write(f"\n[epoch {ep}] prédiction val ({ckpt_name})…")
        _predict_val(model_dir, fold, ckpt_name, images_tr, val_ids, tmp, device)
        # Métriques : squelette/EDT de la prédiction calculés une seule fois par
        # cas (PredFeatures), réutilisés sur tous les scénarios GT.
        for cid in tqdm(val_ids, desc=f"Métriques ep{ep}", unit="cas", leave=False):
            pred = os.path.join(tmp, f"{cid}.nii.gz")
            if not os.path.exists(pred):
                continue
            pred_nii = nib.load(pred)
            pf = PredFeatures(pred_nii.get_fdata(), spacing=pred_nii.header.get_zooms()[:3])
            for sc in scenarios:
                gt = os.path.join(scenario_gt_dir(clean_dir, "labelsTr", sc), f"{cid}.nii.gz")
                if not os.path.exists(gt):
                    continue
                m = evaluate_pair_cached(pf, nib.load(gt).get_fdata())
                rows.append({"trainer": trainer, "fold": fold, "epoch": ep,
                             "scenario": sc["name"], "case": f"{cid}.nii.gz", **m})
        if not args.keep_preds:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    if not rows:
        print("Aucune métrique calculée.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "calibration.csv")
    df.to_csv(csv_path, index=False)

    rec = recommend(df, float(cal.get("plateau_tol", 0.01)))
    plot_curves(df, os.path.join(args.results_dir, "figures", "calibration_topology.png"), rec)

    print(f"\n{'='*60}\nRÉSUMÉ CALIBRATION (clDice moyen GT_star par epoch)\n{'='*60}")
    summary = (df[df["scenario"] == "GT_star"].groupby("epoch")[["cldice", "betti0"]]
               .mean().round(4))
    print(summary.to_string())
    if rec:
        print(f"\n→ Plateau à (1-{cal.get('plateau_tol',0.01)})·max(clDice={rec['max_cldice']}) "
              f"= {rec['target']}")
        print(f"→ num_epochs RECOMMANDÉ : {rec['recommended_num_epochs']}")
    print(f"\nDétails : {csv_path}")
    print(f"Courbe  : {os.path.join(args.results_dir, 'figures', 'calibration_topology.png')}")


if __name__ == "__main__":
    main()
