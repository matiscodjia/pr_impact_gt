#!/usr/bin/env python3
"""Test d'apprentissage du bruit à partir du CSV de cross-évaluation.

Pénalité par cas = CLDice(GT_star) - CLDice(GT_minus_test).
Un modèle dont les prédictions imitent les GT dégradés est moins pénalisé
lors d'une évaluation sur GT dégradés → pénalité systématiquement plus faible.

Usage
-----
    python scripts/noise_learning_test.py --csv results/cross_evaluation.csv
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def run(csv_path: str, output_dir: str) -> None:
    df = pd.read_csv(csv_path)

    models = sorted(df["model"].unique())
    GT_CLEAN = "GT_star"
    GT_DEGRADED = "GT_minus_test"

    penalties: dict[str, pd.Series] = {}
    for model in models:
        star = (
            df[(df["model"] == model) & (df["scenario"] == GT_CLEAN)]
            .set_index("case")["cldice"]
        )
        minus = (
            df[(df["model"] == model) & (df["scenario"] == GT_DEGRADED)]
            .set_index("case")["cldice"]
        )
        common = star.index.intersection(minus.index)
        if len(common) >= 5:
            penalties[model] = star.loc[common] - minus.loc[common]

    if len(penalties) < 2:
        print("Pas assez de modèles avec les deux scénarios.")
        return

    print(f"\n{'='*60}")
    print("TEST D'APPRENTISSAGE DU BRUIT")
    print("Pénalité = CLDice(GT_star) − CLDice(GT_minus_test) par cas")
    print("Hypothèse : prédictions biaisées vers GT dégradé → pénalité plus faible")
    print(f"{'='*60}")

    print("\nPénalité par modèle :")
    summary_rows = []
    for model, pen in penalties.items():
        row = {
            "model": model,
            "n_cases": len(pen),
            "mean_penalty": round(float(pen.mean()), 4),
            "std_penalty": round(float(pen.std(ddof=1)), 4),
            "median_penalty": round(float(pen.median()), 4),
        }
        summary_rows.append(row)
    print(pd.DataFrame(summary_rows).to_string(index=False))

    model_list = sorted(penalties.keys())
    test_rows = []
    for i, model_a in enumerate(model_list):
        for model_b in model_list[i + 1:]:
            a_vals = penalties[model_a].loc[
                penalties[model_a].index.intersection(penalties[model_b].index)
            ].values
            b_vals = penalties[model_b].loc[
                penalties[model_b].index.intersection(penalties[model_a].index)
            ].values

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
                "n_cases": len(a_vals),
                "mean_penalty_a": round(float(np.mean(a_vals)), 4),
                "mean_penalty_b": round(float(np.mean(b_vals)), 4),
                "delta_penalty": round(float(np.mean(diff)), 4),
                "cohens_d": round(float(cohens_d), 4) if not np.isnan(cohens_d) else None,
                "p_value": round(float(p_value), 4) if not np.isnan(p_value) else None,
                "significant": bool(p_value < 0.05) if not np.isnan(p_value) else False,
            })

    test_df = pd.DataFrame(test_rows)
    print(f"\nComparaison des pénalités (Wilcoxon apparié) :")
    print("delta_penalty > 0 : model_b plus pénalisé → model_a a plus appris le bruit")
    print(test_df.to_string(index=False))

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "noise_learning_test.csv")
    test_df.to_csv(out_path, index=False)
    print(f"\nTest sauvegardé : {out_path}")

    # ── Plot ──
    sorted_models = sorted(penalties.keys())
    _, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        [penalties[m].values for m in sorted_models],
        labels=sorted_models,
        patch_artist=True,
        boxprops=dict(facecolor="#e3f2fd"),
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(
        "Pénalité par modèle (CLDice GT_star − GT_minus_test)\n"
        "Pénalité plus faible = prédictions biaisées vers GT dégradé"
    )
    ax.set_ylabel("Pénalité CLDice")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "penalty_noise_learning.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure sauvegardée : {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/cross_evaluation.csv")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    run(args.csv, args.output)
