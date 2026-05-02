#!/usr/bin/env python3
"""Cross-évaluation des trois modèles : Model_Star, Model_Minus_Stoch, Model_Minus_Fixed.

Matrice d'évaluation complète (3 modèles × 3 scénarios GT) :

    ┌──────────────────┬──────────┬────────────────┬──────────────────┐
    │                  │ GT*      │ GT_minus_test  │
    ├──────────────────┼──────────┼────────────────┤
    │ Model_Star       │ baseline │ pénalité eval  │
    │ Model_Minus_Stoch│ qualité  │ biais mémorisé?│
    │ Model_Minus_Fixed│ qualité  │ biais mémorisé?│
    └──────────────────┴──────────┴────────────────┘

Outcomes adressés :
    1 — Quality vs variability : Minus_Fixed vs Minus_Stoch sur GT*
    3 — Robustesse induite     : Minus_Stoch vs Star sur GT_minus
    4 — Pénalité injuste       : Star sur GT* vs GT_minus
    6 — Asymétrie train/test   : pénalité train vs pénalité eval

Usage
-----
    python cross_evaluate.py --config configs/experiment_config.yaml
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon


def compute_cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    from skimage.morphology import skeletonize

    pred_bool = pred > 0.5
    gt_bool = gt > 0.5

    if not pred_bool.any() and not gt_bool.any():
        return 1.0
    if not pred_bool.any() or not gt_bool.any():
        return 0.0

    skel_pred = skeletonize(pred_bool)
    skel_gt = skeletonize(gt_bool)

    # Tprec : fraction du squelette prédit couverte par le volume GT
    tprec = np.logical_and(skel_pred, gt_bool).sum() / max(skel_pred.sum(), 1)
    # Tsens : fraction du squelette GT couverte par le volume prédit
    tsens = np.logical_and(skel_gt, pred_bool).sum() / max(skel_gt.sum(), 1)

    if tprec + tsens == 0:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))


def compute_nsd(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1, 1, 1), tolerance: float = 2.0) -> float:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    pred_bool = pred > 0.5
    gt_bool = gt > 0.5

    if not pred_bool.any() and not gt_bool.any():
        return 1.0
    if not pred_bool.any() or not gt_bool.any():
        return 0.0

    struct = np.ones((3, 3, 3))
    surf_pred = pred_bool ^ binary_erosion(pred_bool, structure=struct)
    surf_gt = gt_bool ^ binary_erosion(gt_bool, structure=struct)

    dt_from_surf_gt = distance_transform_edt(~surf_gt, sampling=spacing)
    dt_from_surf_pred = distance_transform_edt(~surf_pred, sampling=spacing)

    pred_within = (dt_from_surf_gt[surf_pred] <= tolerance).sum()
    gt_within = (dt_from_surf_pred[surf_gt] <= tolerance).sum()

    return float((pred_within + gt_within) / (surf_pred.sum() + surf_gt.sum()))


def compute_betti0(pred: np.ndarray, gt: np.ndarray) -> int:
    from scipy.ndimage import label

    _, n_pred = label(pred > 0.5)
    _, n_gt = label(gt)
    return abs(n_pred - n_gt)


def compute_hd95(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1, 1, 1)) -> float:
    from scipy.ndimage import distance_transform_edt

    pred_bool = pred > 0.5
    gt_bool = gt

    if not pred_bool.any() or not gt_bool.any():
        return np.inf

    dt_pred = distance_transform_edt(~pred_bool, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_bool, sampling=spacing)

    all_distances = np.concatenate([dt_pred[gt_bool], dt_gt[pred_bool]])
    return float(np.percentile(all_distances, 95))


def _evaluate_one(pred_path: str, gt_dir: str, model_name: str, scenario_name: str) -> dict | None:
    fname = os.path.basename(pred_path)
    gt_path = os.path.join(gt_dir, fname)

    if not os.path.exists(gt_path):
        print(f"  [WARN] GT manquant pour {fname}, skip")
        return None

    pred_nii = nib.load(pred_path)
    gt_nii = nib.load(gt_path)
    spacing = pred_nii.header.get_zooms()[:3]

    pred_data = pred_nii.get_fdata()
    gt_data = gt_nii.get_fdata()

    cldice = compute_cldice(pred_data, gt_data)
    hd95 = compute_hd95(pred_data, gt_data, spacing=spacing)
    nsd = compute_nsd(pred_data, gt_data, spacing=spacing)
    betti0 = compute_betti0(pred_data, gt_data)

    return {
        "model": model_name,
        "scenario": scenario_name,
        "case": fname,
        "cldice": cldice,
        "hd95": hd95,
        "nsd": nsd,
        "betti0": betti0,
    }


def evaluate_predictions(
    pred_dir: str,
    gt_dir: str,
    model_name: str,
    scenario_name: str,
    n_workers: int = 4,
) -> list[dict]:
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_evaluate_one, p, gt_dir, model_name, scenario_name): p
            for p in pred_files
        }
        with tqdm(as_completed(futures), total=len(futures), desc=f"{model_name} × {scenario_name}", unit="cas") as pbar:
            for future in pbar:
                result = future.result()
                if result is not None:
                    results.append(result)
                    pbar.set_postfix(case=result["case"], CLDice=f"{result['cldice']:.4f}")

    return sorted(results, key=lambda r: r["case"])


def pairwise_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
    """Tests de Wilcoxon par paires de modèles pour chaque scénario GT.

    Inclut la taille d'effet (Cohen's d apparié) et la correction FDR
    de Benjamini-Hochberg sur l'ensemble des comparaisons.

    Returns
    -------
    pd.DataFrame
        Colonnes : scenario, model_a, model_b, mean_a, mean_b,
                   delta, cohens_d, p_value, p_fdr, significant
    """
    models = sorted(df["model"].unique())
    scenarios = sorted(df["scenario"].unique())
    rows = []

    for scenario in scenarios:
        sc_df = df[df["scenario"] == scenario]
        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                a_vals = (
                    sc_df[sc_df["model"] == model_a]
                    .set_index("case")["cldice"]
                )
                b_vals = (
                    sc_df[sc_df["model"] == model_b]
                    .set_index("case")["cldice"]
                )
                # Garder les cas communs pour le test apparié
                common = a_vals.index.intersection(b_vals.index)
                if len(common) < 5:
                    continue

                a = a_vals.loc[common].values
                b = b_vals.loc[common].values
                diff = b - a

                try:
                    _, p_value = wilcoxon(a, b)
                except ValueError:
                    p_value = np.nan

                # Cohen's d apparié
                std_diff = np.std(diff, ddof=1)
                cohens_d = np.mean(diff) / std_diff if std_diff > 0 else np.nan

                rows.append({
                    "scenario": scenario,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_cases": len(common),
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "delta": float(np.mean(diff)),
                    "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else None,
                    "p_value": float(p_value) if not np.isnan(p_value) else None,
                })

    if not rows:
        return pd.DataFrame()

    stat_df = pd.DataFrame(rows)

    # Correction FDR Benjamini-Hochberg
    p_vals = stat_df["p_value"].fillna(1.0).values
    n = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    p_fdr = np.empty(n)
    p_fdr[sorted_idx] = p_vals[sorted_idx] * n / (np.arange(n) + 1)
    # Correction monotone
    for k in range(n - 2, -1, -1):
        p_fdr[sorted_idx[k]] = min(p_fdr[sorted_idx[k]], p_fdr[sorted_idx[k + 1]])
    p_fdr = np.clip(p_fdr, 0, 1)

    stat_df["p_fdr"] = p_fdr
    stat_df["significant"] = stat_df["p_fdr"] < 0.05

    return stat_df.round(4)


def noise_learning_test(
    df: pd.DataFrame,
    gt_clean: str = "GT_star",
    gt_degraded: str = "GT_minus_test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Teste si un modèle a appris le bruit via la pénalité d'évaluation par cas.

    Pénalité = CLDice(GT_star) - CLDice(GT_minus_test) par cas.
    Un modèle dont les prédictions imitent les GT dégradés est moins pénalisé
    quand l'évaluation se fait sur GT dégradés → pénalité systématiquement plus faible.

    Returns
    -------
    summary_df : pénalité moyenne/std/médiane par modèle
    test_df    : Wilcoxon apparié sur les pénalités entre modèles
    """
    models = sorted(df["model"].unique())

    penalties: dict[str, pd.Series] = {}
    for model in models:
        star = (
            df[(df["model"] == model) & (df["scenario"] == gt_clean)]
            .set_index("case")["cldice"]
        )
        minus = (
            df[(df["model"] == model) & (df["scenario"] == gt_degraded)]
            .set_index("case")["cldice"]
        )
        common = star.index.intersection(minus.index)
        if len(common) >= 5:
            penalties[model] = star.loc[common] - minus.loc[common]

    if len(penalties) < 2:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows = [
        {
            "model": model,
            "n_cases": len(pen),
            "mean_penalty": round(float(pen.mean()), 4),
            "std_penalty": round(float(pen.std(ddof=1)), 4),
            "median_penalty": round(float(pen.median()), 4),
        }
        for model, pen in penalties.items()
    ]

    model_list = sorted(penalties.keys())
    test_rows = []
    for i, model_a in enumerate(model_list):
        for model_b in model_list[i + 1:]:
            a = penalties[model_a]
            b = penalties[model_b]
            common = a.index.intersection(b.index)
            if len(common) < 5:
                continue

            a_vals = a.loc[common].values
            b_vals = b.loc[common].values
            diff = b_vals - a_vals  # positif = model_b plus pénalisé → model_a a plus appris le bruit

            try:
                _, p_value = wilcoxon(a_vals, b_vals)
            except ValueError:
                p_value = np.nan

            std_diff = np.std(diff, ddof=1)
            cohens_d = np.mean(diff) / std_diff if std_diff > 0 else np.nan

            test_rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "n_cases": len(common),
                "mean_penalty_a": round(float(np.mean(a_vals)), 4),
                "mean_penalty_b": round(float(np.mean(b_vals)), 4),
                "delta_penalty": round(float(np.mean(diff)), 4),
                "cohens_d": round(float(cohens_d), 4) if not np.isnan(cohens_d) else None,
                "p_value": round(float(p_value), 4) if not np.isnan(p_value) else None,
                "significant": bool(p_value < 0.05) if not np.isnan(p_value) else False,
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(test_rows)


def generate_summary_plots(df: pd.DataFrame, output_dir: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # ── Heatmap Dice ──
    pivot = df.groupby(["model", "scenario"])["cldice"].mean().unstack()
    _, ax = plt.subplots(figsize=(9, len(pivot) * 1.2 + 1))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.3, vmax=0.95, linewidths=0.5, ax=ax,
    )
    ax.set_title("Cross-évaluation — CLDice moyen")
    ax.set_ylabel("Modèle")
    ax.set_xlabel("Scénario GT")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_cldice.png"), dpi=150)
    plt.close()

    # ── Heatmap Betti0 ──
    pivot_b0 = df.groupby(["model", "scenario"])["betti0"].mean().unstack()
    _, ax = plt.subplots(figsize=(9, len(pivot_b0) * 1.2 + 1))
    sns.heatmap(
        pivot_b0, annot=True, fmt=".1f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Cross-évaluation — Betti0 moyen (|Δ composantes connexes|)")
    ax.set_ylabel("Modèle")
    ax.set_xlabel("Scénario GT")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_betti0.png"), dpi=150)
    plt.close()

    # ── Heatmap HD95 ──
    pivot_hd = df.groupby(["model", "scenario"])["hd95"].mean().unstack()
    pivot_hd = pivot_hd.replace([np.inf], np.nan)
    _, ax = plt.subplots(figsize=(9, len(pivot_hd) * 1.2 + 1))
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

    # ── Box plots Dice ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    palette = {"Model_Star": "#4CAF50", "Model_Minus_Stoch": "#FF9800", "Model_Minus_Fixed": "#2196F3"}

    sns.boxplot(
        data=df, x="scenario", y="cldice", hue="model",
        ax=axes[0], palette=palette,
    )
    axes[0].set_title("Distribution CLDice par scénario")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="lower left")

    df_finite = df[df["hd95"] < np.inf]
    if not df_finite.empty:
        sns.boxplot(
            data=df_finite, x="scenario", y="hd95", hue="model",
            ax=axes[1], palette=palette,
        )
    axes[1].set_title("Distribution HD95 par scénario")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplots.png"), dpi=150)
    plt.close()

    # ── Penalty plot — apprentissage du bruit ──
    penalties_by_model: dict[str, np.ndarray] = {}
    for model in df["model"].unique():
        star = df[(df["model"] == model) & (df["scenario"] == "GT_star")].set_index("case")["cldice"]
        minus = df[(df["model"] == model) & (df["scenario"] == "GT_minus_test")].set_index("case")["cldice"]
        common = star.index.intersection(minus.index)
        if len(common) > 0:
            penalties_by_model[model] = (star.loc[common] - minus.loc[common]).values

    if penalties_by_model:
        sorted_models = sorted(penalties_by_model.keys())
        _, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot(
            [penalties_by_model[m] for m in sorted_models],
            labels=sorted_models,
            patch_artist=True,
            boxprops=dict(facecolor="#e3f2fd"),
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title("Pénalité par modèle (CLDice GT_star − GT_minus_test)\nUn modèle ayant appris le bruit a une pénalité plus faible")
        ax.set_ylabel("Pénalité CLDice")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "penalty_noise_learning.png"), dpi=150)
        plt.close()

    print(f"  Figures sauvegardées dans {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-évaluation Model_Star / Model_Minus_Stoch / Model_Minus_Fixed"
    )
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_id = config["dataset"]["id"]
    dataset_name = f"Dataset{dataset_id:03d}_PARSE"
    nnunet_raw = os.environ.get("nnUNet_raw", "nnUNet_data/nnUNet_raw")
    dataset_dir = os.path.join(nnunet_raw, dataset_name)

    os.makedirs(args.output, exist_ok=True)

    # Modèles — Model_Minus_Stoch garde le dossier "model_minus" pour
    # la compatibilité ascendante avec les runs existants
    models = {
        "Model_Star":        os.path.join("predictions", "model_star"),
        "Model_Minus_Stoch": os.path.join("predictions", "model_minus_stoch"),
        "Model_Minus_Fixed": os.path.join("predictions", "model_minus_fixed"),
    }
    # Rétrocompatibilité : si model_minus_stoch absent, chercher model_minus
    if not os.path.isdir(models["Model_Minus_Stoch"]):
        legacy = os.path.join("predictions", "model_minus")
        if os.path.isdir(legacy):
            print(f"[COMPAT] Utilisation de {legacy} pour Model_Minus_Stoch")
            models["Model_Minus_Stoch"] = legacy

    # Scénarios GT
    scenarios = {}
    for sc in config["evaluation"]["scenarios"]:
        name = sc["name"]
        if sc.get("degradation") is None:
            scenarios[name] = os.path.join(dataset_dir, "labelsTs")
        else:
            scenarios[name] = os.path.join(dataset_dir, f"labelsTs_{name}")

    all_results = []

    for model_name, pred_dir in models.items():
        if not os.path.isdir(pred_dir):
            print(f"[SKIP] {pred_dir} introuvable — {model_name} ignoré")
            continue

        for scenario_name, gt_dir in scenarios.items():
            if not os.path.isdir(gt_dir):
                print(f"[SKIP] {gt_dir} introuvable")
                continue

            print(f"\nÉvaluation: {model_name} × {scenario_name}")
            results = evaluate_predictions(pred_dir, gt_dir, model_name, scenario_name)
            all_results.extend(results)

            if results:
                cldices = [r["cldice"] for r in results]
                print(f"  CLDice: {np.mean(cldices):.4f} ± {np.std(cldices):.4f}")

    if not all_results:
        print("\nAucun résultat. Vérifiez que les prédictions et labels existent.")
        return

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output, "cross_evaluation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nRésultats sauvegardés : {csv_path}")

    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    summary = df.groupby(["model", "scenario"]).agg(
        cldice_mean=("cldice", "mean"),
        cldice_std=("cldice", "std"),
        hd95_mean=("hd95", lambda x: np.mean(x[x < np.inf])),
        nsd_mean=("nsd", "mean"),
        betti0_mean=("betti0", "mean"),
    ).round(4)
    print(summary.to_string())

    # Tests statistiques appariés (pairwise, FDR corrigé)
    print(f"\n{'='*60}")
    print("TESTS STATISTIQUES (Wilcoxon apparié, FDR Benjamini-Hochberg)")
    print(f"{'='*60}")
    stat_df = pairwise_wilcoxon(df)
    if not stat_df.empty:
        stat_path = os.path.join(args.output, "statistical_tests.csv")
        stat_df.to_csv(stat_path, index=False)
        print(stat_df.to_string(index=False))
        print(f"\nTests sauvegardés : {stat_path}")

    # Test d'apprentissage du bruit
    print(f"\n{'='*60}")
    print("TEST D'APPRENTISSAGE DU BRUIT")
    print("Pénalité = CLDice(GT_star) − CLDice(GT_minus_test) par cas")
    print("Hypothèse : prédictions biaisées vers GT dégradé → pénalité plus faible")
    print(f"{'='*60}")
    penalty_summary, penalty_test = noise_learning_test(df)
    if not penalty_summary.empty:
        print("\nPénalité par modèle :")
        print(penalty_summary.to_string(index=False))
        if not penalty_test.empty:
            print("\nComparaison des pénalités (Wilcoxon apparié) :")
            print(penalty_test.to_string(index=False))
            penalty_path = os.path.join(args.output, "noise_learning_test.csv")
            penalty_test.to_csv(penalty_path, index=False)
            print(f"\nTest sauvegardé : {penalty_path}")

    generate_summary_plots(df, os.path.join(args.output, "figures"))


if __name__ == "__main__":
    main()
