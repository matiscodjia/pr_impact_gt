#!/usr/bin/env python3
"""Agrégation et visualisation des résultats expérimentaux.

Génère un rapport complet avec :
- Tableau récapitulatif des métriques
- Heatmaps du grid search
- Comparaisons Model_Star vs Model_Minus
- Analyses statistiques (Wilcoxon signed-rank test)

Usage
-----
    python aggregate_results.py --results_dir results
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)


def plot_grid_search_heatmaps(df: pd.DataFrame, output_dir: str) -> None:
    """Génère les heatmaps pour chaque phase du grid search.

    Parameters
    ----------
    df : pd.DataFrame
        Résultats du grid search.
    output_dir : str
        Dossier de sortie pour les figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Phase 1 : Morpho ──
    phase1 = df[df["phase"] == 1]
    if not phase1.empty:
        probs = sorted(phase1["morpho_prob"].unique())
        radii = sorted(phase1["morpho_max_radius"].unique())

        matrix = np.zeros((len(radii), len(probs)))
        for _, row in phase1.iterrows():
            pi = probs.index(row["morpho_prob"])
            ri = radii.index(row["morpho_max_radius"])
            matrix[ri, pi] = row["dice_star"]

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.3, vmax=0.9)
        ax.set_xticks(range(len(probs)))
        ax.set_xticklabels([f"{p:.1f}" for p in probs])
        ax.set_yticks(range(len(radii)))
        ax.set_yticklabels([str(r) for r in radii])
        ax.set_xlabel("Probabilité morpho")
        ax.set_ylabel("Rayon max (voxels)")
        ax.set_title("Phase 1 — Morpho seule — Dice sur GT*")

        for i in range(len(radii)):
            for j in range(len(probs)):
                ax.text(
                    j, i, f"{matrix[i, j]:.3f}",
                    ha="center", va="center",
                    color="white" if matrix[i, j] < 0.55 else "black",
                    fontsize=11, fontweight="bold",
                )

        plt.colorbar(im, ax=ax, label="Dice Score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmap_phase1_morpho.png"), dpi=150)
        plt.close()
        print(f"  Heatmap phase 1 sauvegardée")

    # ── Phase 2 : Omission ──
    phase2 = df[df["phase"] == 2]
    if not phase2.empty:
        probs = sorted(phase2["omission_prob"].unique())
        sizes = sorted(phase2["omission_min_size"].unique())

        matrix = np.zeros((len(sizes), len(probs)))
        for _, row in phase2.iterrows():
            pi = probs.index(row["omission_prob"])
            si = sizes.index(row["omission_min_size"])
            matrix[si, pi] = row["dice_star"]

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.3, vmax=0.9)
        ax.set_xticks(range(len(probs)))
        ax.set_xticklabels([f"{p:.1f}" for p in probs])
        ax.set_yticks(range(len(sizes)))
        ax.set_yticklabels([str(s) for s in sizes])
        ax.set_xlabel("Probabilité omission")
        ax.set_ylabel("Taille min (voxels)")
        ax.set_title("Phase 2 — Omission seule — Dice sur GT*")

        for i in range(len(sizes)):
            for j in range(len(probs)):
                ax.text(
                    j, i, f"{matrix[i, j]:.3f}",
                    ha="center", va="center",
                    color="white" if matrix[i, j] < 0.55 else "black",
                    fontsize=11, fontweight="bold",
                )

        plt.colorbar(im, ax=ax, label="Dice Score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmap_phase2_omission.png"), dpi=150)
        plt.close()
        print(f"  Heatmap phase 2 sauvegardée")

    # ── Phase 3 : Bar chart comparatif ──
    phase3 = df[df["phase"] == 3]
    if not phase3.empty:
        phase3_sorted = phase3.sort_values("dice_star", ascending=False)

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(phase3_sorted))
        ax.bar(x, phase3_sorted["dice_star"], color="#4fc3f7", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(
            phase3_sorted["trainer"].str.replace("nnUNetTrainerDeg_", ""),
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_ylabel("Dice Score")
        ax.set_title("Phase 3 — Combinaisons — Dice sur GT*")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "barchart_phase3.png"), dpi=150)
        plt.close()
        print(f"  Bar chart phase 3 sauvegardé")


def statistical_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Effectue un test statistique Model_Star vs Model_Minus.

    Parameters
    ----------
    df : pd.DataFrame
        Résultats de la cross-évaluation (un row par cas).
    output_dir : str
        Dossier de sortie.
    """
    from scipy.stats import wilcoxon

    os.makedirs(output_dir, exist_ok=True)

    # Pour chaque scénario GT, comparer les Dice des deux modèles
    scenarios = df["scenario"].unique()
    stat_results = []

    for scenario in scenarios:
        sc_df = df[df["scenario"] == scenario]

        star_dices = sc_df[sc_df["model"] == "Model_Star"]["dice"].values
        minus_dices = sc_df[sc_df["model"] == "Model_Minus"]["dice"].values

        if len(star_dices) != len(minus_dices) or len(star_dices) < 5:
            continue

        try:
            stat, p_value = wilcoxon(star_dices, minus_dices)
        except ValueError:
            stat, p_value = np.nan, np.nan

        stat_results.append({
            "scenario": scenario,
            "star_mean": np.mean(star_dices),
            "minus_mean": np.mean(minus_dices),
            "delta": np.mean(minus_dices) - np.mean(star_dices),
            "p_value": p_value,
            "significant": p_value < 0.05 if not np.isnan(p_value) else False,
        })

    if stat_results:
        stat_df = pd.DataFrame(stat_results)
        stat_path = os.path.join(output_dir, "statistical_tests.csv")
        stat_df.to_csv(stat_path, index=False)

        print(f"\n{'='*60}")
        print("TESTS STATISTIQUES (Wilcoxon signed-rank)")
        print(f"{'='*60}")
        print(stat_df.to_string(index=False))


def main():
    """Point d'entrée de l'agrégation."""
    parser = argparse.ArgumentParser(
        description="Agrégation et visualisation des résultats"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
    )
    args = parser.parse_args()

    output_dir = os.path.join(args.results_dir, "figures")

    # ── Grid search ──
    gs_path = os.path.join(args.results_dir, "grid_search_results.csv")
    if os.path.exists(gs_path):
        print("Génération des visualisations grid search...")
        gs_df = pd.read_csv(gs_path)
        plot_grid_search_heatmaps(gs_df, output_dir)
    else:
        print(f"[SKIP] {gs_path} n'existe pas")

    # ── Cross-évaluation ──
    ce_path = os.path.join(args.results_dir, "cross_evaluation.csv")
    if os.path.exists(ce_path):
        print("\nAnalyse statistique de la cross-évaluation...")
        ce_df = pd.read_csv(ce_path)
        statistical_comparison(ce_df, args.results_dir)
    else:
        print(f"[SKIP] {ce_path} n'existe pas")

    print(f"\nAgrégation terminée. Figures dans {output_dir}/")


if __name__ == "__main__":
    main()
