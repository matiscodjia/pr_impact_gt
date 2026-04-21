#!/usr/bin/env python3
"""Cross-évaluation des modèles Model_Star et Model_Minus.

Calcule les métriques de segmentation (Dice, HD95, NSD) pour chaque
combinaison modèle × scénario de GT :

    ┌──────────────┬──────────┬────────────────┬──────────────────┐
    │              │ GT*      │ GT- (mild)     │ GT- (severe)     │
    ├──────────────┼──────────┼────────────────┼──────────────────┤
    │ Model_Star   │ baseline │ robustesse     │ robustesse       │
    │ Model_Minus  │ qualité  │ cible          │ cible            │
    └──────────────┴──────────┴────────────────┴──────────────────┘

Les résultats sont sauvegardés en CSV et des visualisations (heatmaps,
bar charts) sont générées automatiquement.

Usage
-----
    python cross_evaluate.py --config configs/experiment_config.yaml
"""

import argparse
import glob
import os
from collections import defaultdict

import nibabel as nib
import numpy as np
import pandas as pd
import yaml


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calcule le coefficient de Dice entre deux masques binaires.

    Parameters
    ----------
    pred : np.ndarray
        Masque prédit (binaire).
    gt : np.ndarray
        Masque de référence (binaire).

    Returns
    -------
    float
        Score Dice entre 0 et 1. Retourne 1.0 si les deux masques
        sont vides, 0.0 si un seul est vide.
    """
    pred_bool = pred > 0.5
    gt_bool = gt > 0.5

    if not pred_bool.any() and not gt_bool.any():
        return 1.0
    if not pred_bool.any() or not gt_bool.any():
        return 0.0

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    return 2.0 * intersection / (pred_bool.sum() + gt_bool.sum())


def compute_hd95(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1, 1, 1)) -> float:
    """Calcule la distance de Hausdorff au 95e percentile.

    Parameters
    ----------
    pred : np.ndarray
        Masque prédit (binaire).
    gt : np.ndarray
        Masque de référence (binaire).
    spacing : tuple of float
        Espacement voxel en mm.

    Returns
    -------
    float
        HD95 en mm. Retourne ``np.inf`` si un des masques est vide.
    """
    from scipy.ndimage import distance_transform_edt

    pred_bool = pred > 0.5
    gt_bool = gt > 0.5

    if not pred_bool.any() or not gt_bool.any():
        return np.inf

    # Distance de chaque voxel GT au plus proche voxel prédit
    dt_pred = distance_transform_edt(~pred_bool, sampling=spacing)
    # Distance de chaque voxel prédit au plus proche voxel GT
    dt_gt = distance_transform_edt(~gt_bool, sampling=spacing)

    # Distances surface → surface
    distances_gt_to_pred = dt_pred[gt_bool]
    distances_pred_to_gt = dt_gt[pred_bool]

    all_distances = np.concatenate([distances_gt_to_pred, distances_pred_to_gt])
    return np.percentile(all_distances, 95)


def evaluate_predictions(
    pred_dir: str,
    gt_dir: str,
    model_name: str,
    scenario_name: str,
) -> list[dict]:
    """Évalue un dossier de prédictions contre un dossier de GT.

    Parameters
    ----------
    pred_dir : str
        Dossier contenant les prédictions (.nii.gz).
    gt_dir : str
        Dossier contenant les labels de référence (.nii.gz).
    model_name : str
        Nom du modèle (pour le logging).
    scenario_name : str
        Nom du scénario GT (pour le logging).

    Returns
    -------
    list[dict]
        Liste de dictionnaires avec les métriques par cas.
    """
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
    results = []

    for pred_path in pred_files:
        fname = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, fname)

        if not os.path.exists(gt_path):
            print(f"  [WARN] GT manquant pour {fname}, skip")
            continue

        pred_nii = nib.load(pred_path)
        gt_nii = nib.load(gt_path)

        pred_data = pred_nii.get_fdata()
        gt_data = gt_nii.get_fdata()

        # Récupérer le spacing pour HD95
        spacing = pred_nii.header.get_zooms()[:3]

        dice = compute_dice(pred_data, gt_data)
        hd95 = compute_hd95(pred_data, gt_data, spacing=spacing)

        results.append({
            "model": model_name,
            "scenario": scenario_name,
            "case": fname,
            "dice": dice,
            "hd95": hd95,
        })

    return results


def generate_summary_plots(df: pd.DataFrame, output_dir: str) -> None:
    """Génère les visualisations de synthèse.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec toutes les métriques.
    output_dir : str
        Dossier de sortie pour les figures.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Heatmap Dice ──
    pivot = df.groupby(["model", "scenario"])["dice"].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.3, vmax=0.95, linewidths=0.5, ax=ax,
    )
    ax.set_title("Cross-évaluation — Dice moyen")
    ax.set_ylabel("Modèle")
    ax.set_xlabel("Scénario GT")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_dice.png"), dpi=150)
    plt.close()

    # ── 2. Heatmap HD95 ──
    pivot_hd = df.groupby(["model", "scenario"])["hd95"].mean().unstack()
    # Clipper les inf pour la visualisation
    pivot_hd = pivot_hd.replace([np.inf], np.nan)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot_hd, annot=True, fmt=".1f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Cross-évaluation — HD95 moyen (mm)")
    ax.set_ylabel("Modèle")
    ax.set_xlabel("Scénario GT")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_hd95.png"), dpi=150)
    plt.close()

    # ── 3. Box plots par modèle ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(
        data=df, x="scenario", y="dice", hue="model",
        ax=axes[0], palette="Set2",
    )
    axes[0].set_title("Distribution Dice par scénario")
    axes[0].set_ylim(0, 1)

    df_finite = df[df["hd95"] < np.inf]
    if not df_finite.empty:
        sns.boxplot(
            data=df_finite, x="scenario", y="hd95", hue="model",
            ax=axes[1], palette="Set2",
        )
    axes[1].set_title("Distribution HD95 par scénario")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplots.png"), dpi=150)
    plt.close()

    print(f"  Figures sauvegardées dans {output_dir}/")


def main():
    """Point d'entrée principal de la cross-évaluation."""
    parser = argparse.ArgumentParser(
        description="Cross-évaluation Model_Star vs Model_Minus"
    )
    parser.add_argument(
        "--config", type=str, default="configs/experiment_config.yaml",
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Dossier de sortie",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_id = config["dataset"]["id"]
    dataset_name = f"Dataset{dataset_id:03d}_PARSE"
    nnunet_raw = os.environ.get("nnUNet_raw", "nnUNet_data/nnUNet_raw")
    dataset_dir = os.path.join(nnunet_raw, dataset_name)

    os.makedirs(args.output, exist_ok=True)

    all_results = []

    # Modèles à évaluer
    models = {
        "Model_Star": os.path.join("predictions", "model_star"),
        "Model_Minus": os.path.join("predictions", "model_minus"),
    }

    # Scénarios GT
    scenarios = {}
    for sc in config["evaluation"]["scenarios"]:
        name = sc["name"]
        if sc.get("degradation") is None:
            scenarios[name] = os.path.join(dataset_dir, "labelsTs")
        else:
            scenarios[name] = os.path.join(dataset_dir, f"labelsTs_{name}")

    # Cross-évaluation
    for model_name, pred_dir in models.items():
        if not os.path.isdir(pred_dir):
            print(f"[SKIP] {pred_dir} n'existe pas")
            continue

        for scenario_name, gt_dir in scenarios.items():
            if not os.path.isdir(gt_dir):
                print(f"[SKIP] {gt_dir} n'existe pas")
                continue

            print(f"\nÉvaluation: {model_name} × {scenario_name}")
            results = evaluate_predictions(
                pred_dir, gt_dir, model_name, scenario_name,
            )
            all_results.extend(results)

            # Résumé rapide
            if results:
                dices = [r["dice"] for r in results]
                print(f"  Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")

    if not all_results:
        print("\nAucun résultat. Vérifiez que les prédictions et labels existent.")
        return

    # Sauvegarder en CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output, "cross_evaluation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nRésultats sauvegardés : {csv_path}")

    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    summary = df.groupby(["model", "scenario"]).agg(
        dice_mean=("dice", "mean"),
        dice_std=("dice", "std"),
        hd95_mean=("hd95", lambda x: np.mean(x[x < np.inf])),
    ).round(4)
    print(summary)

    # Générer les plots
    generate_summary_plots(df, os.path.join(args.output, "figures"))


if __name__ == "__main__":
    main()
