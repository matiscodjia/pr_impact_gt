#!/usr/bin/env python3
"""Collecte incrémentale des métriques → table maîtresse (out-of-fold + test).

Stratégie d'évaluation (validée) : **out-of-fold (OOF)** en primaire, **test set**
en confirmation.

- OOF : nnU-Net écrit, à la fin de chaque fold, ses prédictions de validation dans
  ``<results>/<trainer>__nnUNetPlans__<cfg>/fold_<k>/validation/*.nii.gz`` (les ~17
  cas que ce fold n'a pas vus). En cumulant les folds terminés on couvre les 85 cas
  → n croît de ~17 à 85 (propriété *anytime*).
- Test : si ``predictions/<model_dir>/`` existe (ensemble des folds sur imagesTs),
  on l'évalue aussi, tagué ``eval_kind=test``.

Toutes les prédictions, quel que soit le modèle/dataset d'entraînement, sont
évaluées contre la **GT propre** (``GT_star``) et chaque scénario dégradé déterministe
défini dans ``evaluation.scenarios`` (``GT_minus_omission``, ``GT_minus_drift_*``),
par appariement de nom de fichier.

La collecte ne calcule que les clés **absentes** de la table maîtresse : relancer
après une interruption est sûr et bon marché.

Usage
-----
    python scripts/collect_metrics.py --config configs/experiment_config.yaml --output results
"""

import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import results_store as rs
from cross_evaluate import (PredFeatures, compute_betti0, compute_cldice, compute_hd95,
                            compute_nsd, evaluate_pair_cached)

PLANS = "nnUNetPlans"


# ─────────────────────────────────────────────────────────────────
# Résolution de chemins (robuste au suffixe du nom de dataset)
# ─────────────────────────────────────────────────────────────────

def resolve_dataset_dir(nnunet_raw: str, dataset_id: int) -> str | None:
    """Dossier raw d'un dataset par ID (glob ``Dataset{ID}_*``)."""
    hits = sorted(glob.glob(os.path.join(nnunet_raw, f"Dataset{dataset_id:03d}_*")))
    return hits[0] if hits else None


def resolve_results_dir(nnunet_results: str, dataset_id: int, trainer: str, cfg: str) -> str | None:
    """Dossier de résultats nnU-Net pour (dataset, trainer, configuration)."""
    base = sorted(glob.glob(os.path.join(nnunet_results, f"Dataset{dataset_id:03d}_*")))
    if not base:
        return None
    path = os.path.join(base[0], f"{trainer}__{PLANS}__{cfg}")
    return path if os.path.isdir(path) else None


def scenario_gt_dir(clean_dataset_dir: str, split: str, scenario: dict) -> str:
    """Dossier des labels GT pour un scénario donné.

    ``split`` ∈ {labelsTr, labelsTs}. Scénario sans dégradation → labels propres ;
    sinon → ``<split>_<name>`` (produits par generate_degraded_dataset.py).
    """
    if scenario.get("degradation") is None:
        return os.path.join(clean_dataset_dir, split)
    return os.path.join(clean_dataset_dir, f"{split}_{scenario['name']}")


# ─────────────────────────────────────────────────────────────────
# Évaluation d'une prédiction
# ─────────────────────────────────────────────────────────────────

def evaluate_pair(pred_path: str, gt_path: str) -> dict | None:
    """Calcule les 4 métriques pour une paire (prédiction, GT). None si GT absente."""
    if not os.path.exists(gt_path):
        return None
    pred_nii = nib.load(pred_path)
    gt_nii = nib.load(gt_path)
    spacing = pred_nii.header.get_zooms()[:3]
    pred = pred_nii.get_fdata()
    gt = gt_nii.get_fdata()
    return {
        "cldice": compute_cldice(pred, gt),
        "hd95": compute_hd95(pred, gt, spacing=spacing),
        "nsd": compute_nsd(pred, gt, spacing=spacing),
        "betti0": compute_betti0(pred, gt),
    }


def _checkpoint_info(fold_dir: str) -> tuple[int | None, float | None, bool]:
    """(epochs, mtime, terminé) à partir de checkpoint_final.pth si présent."""
    final = os.path.join(fold_dir, "checkpoint_final.pth")
    if os.path.exists(final):
        return None, os.path.getmtime(final), True
    return None, None, False


# ─────────────────────────────────────────────────────────────────
# Collecte
# ─────────────────────────────────────────────────────────────────

def collect_oof(model, clean_dir, results_dir, cfg, scenarios, done, num_epochs):
    """Lignes OOF pour un modèle : balaie les folds terminés × scénarios × cas."""
    rdir = resolve_results_dir(results_dir, model["dataset_id"], model["trainer"], cfg)
    if rdir is None:
        return []

    rows = []
    for fold_dir in sorted(glob.glob(os.path.join(rdir, "fold_*"))):
        fold = int(os.path.basename(fold_dir).split("_")[1])
        val_dir = os.path.join(fold_dir, "validation")
        _, mtime, finished = _checkpoint_info(fold_dir)
        if not finished or not os.path.isdir(val_dir):
            continue  # fold non terminé → pas de prédictions OOF

        for pred_path in sorted(glob.glob(os.path.join(val_dir, "*.nii.gz"))):
            case = os.path.basename(pred_path)
            # Scénarios encore à calculer pour ce cas : la prédiction (squelette,
            # EDT…) n'est chargée et factorisée qu'une fois pour tous (PredFeatures).
            todo = [sc for sc in scenarios
                    if (model["name"], model["trainer"], int(model["dataset_id"]),
                        fold, sc["name"], case, "oof") not in done]
            if not todo:
                continue
            pred_nii = nib.load(pred_path)
            pf = PredFeatures(pred_nii.get_fdata(), spacing=pred_nii.header.get_zooms()[:3])
            for sc in todo:
                gt_path = os.path.join(scenario_gt_dir(clean_dir, "labelsTr", sc), case)
                if not os.path.exists(gt_path):
                    continue
                metrics = evaluate_pair_cached(pf, nib.load(gt_path).get_fdata())
                rows.append(rs.make_row(
                    model=model["name"], trainer=model["trainer"],
                    dataset=model["dataset_id"], fold=fold, scenario=sc["name"],
                    case=case, eval_kind="oof", metrics=metrics,
                    epochs=num_epochs, checkpoint_mtime=mtime,
                ))
    return rows


def collect_test(model, clean_dir, scenarios, done, num_epochs):
    """Lignes test (ensemble) pour un modèle, si predictions/<dir> existe."""
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
        pf = PredFeatures(pred_nii.get_fdata(), spacing=pred_nii.header.get_zooms()[:3])
        for sc in todo:
            gt_path = os.path.join(scenario_gt_dir(clean_dir, "labelsTs", sc), case)
            if not os.path.exists(gt_path):
                continue
            metrics = evaluate_pair_cached(pf, nib.load(gt_path).get_fdata())
            rows.append(rs.make_row(
                model=model["name"], trainer=model["trainer"],
                dataset=model["dataset_id"], fold=-1, scenario=sc["name"],
                case=case, eval_kind="test", metrics=metrics,
                epochs=num_epochs, checkpoint_mtime=None,
            ))
    return rows


def _model_pred_dir(model_name: str) -> str:
    """Convention de dossier de prédictions test : nom du modèle en minuscules
    (ex. M0_Star → m0_star)."""
    return model_name.lower()


def main():
    parser = argparse.ArgumentParser(description="Collecte incrémentale OOF + test")
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

    # GT canonique : toujours le dataset source propre (Dataset100).
    clean_dir = resolve_dataset_dir(nnunet_raw, config["dataset"]["id"])
    if clean_dir is None:
        print(f"ERREUR: dataset source introuvable dans {nnunet_raw}")
        return

    df = rs.load(args.output)
    done = rs.existing_keys(df)
    print(f"Table maîtresse : {len(df)} lignes déjà présentes")

    all_rows = []
    for model in config["experiment"]["models"]:
        oof = collect_oof(model, clean_dir, nnunet_results, cfg, scenarios, done, num_epochs)
        test = collect_test(model, clean_dir, scenarios, done, num_epochs)
        if oof or test:
            print(f"  {model['name']:<26} +{len(oof)} OOF  +{len(test)} test")
        all_rows.extend(oof)
        all_rows.extend(test)

    if not all_rows:
        print("Rien de nouveau à collecter.")
        return

    n_new = rs.upsert(args.output, all_rows)
    print(f"\n{n_new} nouvelles lignes upsertées → {os.path.join(args.output, rs.CSV_NAME)}")


if __name__ == "__main__":
    main()
