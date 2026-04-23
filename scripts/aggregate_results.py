#!/usr/bin/env python3
"""Agrégation et visualisation des résultats expérimentaux.

Génère un rapport complet couvrant les 7 outcomes de recherche :

    1  Qualité du signal vs variabilité (Fixed vs Stoch sur GT*)
    2  Seuils de rupture (grid search heatmaps)
    3  Robustesse induite par le bruit stochastique
    4  Pénalisation injuste par une GT d'évaluation biaisée
    5  Morpho vs omission (phases grid search)
    6  Asymétrie train/test de la pénalité
    7  Spécificité de la robustesse acquise

Usage
-----
    python aggregate_results.py --results_dir results
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=FutureWarning)

PALETTE = {
    "Model_Star":        "#4CAF50",
    "Model_Minus_Stoch": "#FF9800",
    "Model_Minus_Fixed": "#2196F3",
}

SCENARIO_ORDER = ["GT_star", "GT_minus_test"]


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _wilcoxon_annotation(a: np.ndarray, b: np.ndarray) -> str:
    """Retourne une annotation p-value et taille d'effet Cohen's d."""
    if len(a) < 5 or len(a) != len(b):
        return ""
    try:
        _, p = wilcoxon(a, b)
    except ValueError:
        return ""
    diff = b - a
    std_diff = np.std(diff, ddof=1)
    d = np.mean(diff) / std_diff if std_diff > 0 else 0.0
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    return f"{stars} (p={p:.3f}, d={d:.2f})"


def _annotate_bar_diff(ax, x1: float, x2: float, y: float, text: str) -> None:
    """Dessine une accolade horizontale avec texte entre deux barres."""
    ax.annotate(
        "", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-", lw=1.2, color="gray"),
    )
    ax.text(
        (x1 + x2) / 2, y * 1.01, text,
        ha="center", va="bottom", fontsize=7.5, color="gray",
    )


# ─────────────────────────────────────────────────────────────────
# Outcome 1 — Qualité du signal vs variabilité
# ─────────────────────────────────────────────────────────────────

def plot_quality_vs_variability(df: pd.DataFrame, output_dir: str) -> None:
    """Outcome 1 : compare Model_Minus_Fixed vs Model_Minus_Stoch sur GT*.

    La différence de Dice entre les deux, à GT d'évaluation identique (GT*),
    isole l'effet de la variabilité du bruit d'annotation.
    """
    needed = {"Model_Minus_Fixed", "Model_Minus_Stoch"}
    available = set(df["model"].unique())
    if not needed.issubset(available):
        print(f"  [SKIP] plot_quality_vs_variability — modèles manquants: {needed - available}")
        return

    gt_star = df[df["scenario"] == "GT_star"].copy()
    subset = gt_star[gt_star["model"].isin(needed)]
    if subset.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Violin / strip plot ──
    ax = axes[0]
    sns.violinplot(
        data=subset, x="model", y="cldice", palette=PALETTE,
        inner=None, ax=ax, alpha=0.6, order=sorted(needed),
    )
    sns.stripplot(
        data=subset, x="model", y="cldice", palette=PALETTE,
        alpha=0.7, jitter=True, ax=ax, order=sorted(needed),
    )
    ax.set_title("Outcome 1 — Dice sur GT*\n(qualité signal vs variabilité)")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)

    fixed_vals = gt_star[gt_star["model"] == "Model_Minus_Fixed"].set_index("case")["cldice"]
    stoch_vals = gt_star[gt_star["model"] == "Model_Minus_Stoch"].set_index("case")["cldice"]
    common = fixed_vals.index.intersection(stoch_vals.index)
    annot = _wilcoxon_annotation(fixed_vals.loc[common].values, stoch_vals.loc[common].values)
    ax.set_xlabel(annot, fontsize=8, color="gray")

    # ── Différences par cas (scatter Fixed vs Stoch) ──
    ax2 = axes[1]
    if len(common) > 0:
        ax2.scatter(
            fixed_vals.loc[common].values,
            stoch_vals.loc[common].values,
            alpha=0.6, color="#9C27B0",
        )
        lims = [
            min(fixed_vals.loc[common].min(), stoch_vals.loc[common].min()) - 0.02,
            max(fixed_vals.loc[common].max(), stoch_vals.loc[common].max()) + 0.02,
        ]
        ax2.plot(lims, lims, "k--", lw=0.8, alpha=0.5, label="égalité")
        ax2.set_xlim(lims)
        ax2.set_ylim(lims)
        ax2.set_xlabel("Dice — Model_Minus_Fixed")
        ax2.set_ylabel("Dice — Model_Minus_Stoch")
        ax2.set_title("Scatter per-case (GT*)\npoints au-dessus = Stoch meilleur")
        ax2.legend(fontsize=8)

    plt.suptitle("Outcome 1 : Qualité du signal vs variabilité", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outcome1_quality_vs_variability.png"), dpi=150)
    plt.close()
    print("  outcome1_quality_vs_variability.png")


# ─────────────────────────────────────────────────────────────────
# Outcome 4 — Pénalisation injuste à l'évaluation
# ─────────────────────────────────────────────────────────────────

def plot_evaluation_penalty(df: pd.DataFrame, output_dir: str) -> None:
    """Outcome 4 : Model_Star seul à travers tous les scénarios GT.

    La chute de performance n'est pas due au modèle mais à la qualité
    du masque d'évaluation — c'est la pénalité injuste.
    """
    if "Model_Star" not in df["model"].unique():
        return

    star_df = df[df["model"] == "Model_Star"].copy()
    scenarios = [s for s in SCENARIO_ORDER if s in star_df["scenario"].unique()]
    if len(scenarios) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Courbe moyenne ± std ──
    ax = axes[0]
    means = [star_df[star_df["scenario"] == s]["cldice"].mean() for s in scenarios]
    stds = [star_df[star_df["scenario"] == s]["cldice"].std() for s in scenarios]

    ax.errorbar(
        scenarios, means, yerr=stds,
        marker="o", linewidth=2, capsize=5, color="#4CAF50",
        label="Model_Star",
    )
    ax.fill_between(
        scenarios,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.15, color="#4CAF50",
    )
    ax.set_title("Outcome 4 — Pénalité injuste (Model_Star)\nle modèle est identique, seule la GT change")
    ax.set_ylabel("Dice moyen")
    ax.set_ylim(max(0, min(means) - 0.15), min(1, max(means) + 0.1))
    ax.tick_params(axis="x", rotation=20)

    # Annoter les chutes
    for i in range(1, len(scenarios)):
        drop = means[i] - means[0]
        ax.annotate(
            f"Δ={drop:+.3f}",
            xy=(scenarios[i], means[i]),
            xytext=(scenarios[i], means[i] - 0.04),
            ha="center", fontsize=8, color="red",
        )

    # ── Box plots per scenario ──
    ax2 = axes[1]
    star_ordered = star_df[star_df["scenario"].isin(scenarios)].copy()
    star_ordered["scenario"] = pd.Categorical(star_ordered["scenario"], categories=scenarios, ordered=True)
    sns.boxplot(
        data=star_ordered, x="scenario", y="cldice",
        color="#4CAF50", ax=ax2, width=0.5,
    )
    ax2.set_title("Distribution Dice (Model_Star)\npar qualité de GT de test")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="x", rotation=20)

    plt.suptitle("Outcome 4 : Pénalisation injuste par la qualité de la GT d'évaluation",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outcome4_evaluation_penalty.png"), dpi=150)
    plt.close()
    print("  outcome4_evaluation_penalty.png")


# ─────────────────────────────────────────────────────────────────
# Outcome 6 — Asymétrie train/test de la pénalité
# ─────────────────────────────────────────────────────────────────

def plot_train_test_asymmetry(df: pd.DataFrame, output_dir: str) -> None:
    """Outcome 6 : compare la pénalité d'une GT biaisée selon qu'elle affecte
    l'entraînement ou l'évaluation.

    Pénalité train  = Dice(Star, GT*) - Dice(Minus_Fixed, GT*)
    Pénalité eval   = Dice(Star, GT*) - Dice(Star, GT_minus_test)
    """
    needed_models = {"Model_Star", "Model_Minus_Fixed"}
    needed_scenarios = {"GT_star", "GT_minus_test"}

    if not needed_models.issubset(set(df["model"].unique())):
        print(f"  [SKIP] plot_train_test_asymmetry — modèles manquants")
        return
    if not needed_scenarios.issubset(set(df["scenario"].unique())):
        print(f"  [SKIP] plot_train_test_asymmetry — scénarios manquants")
        return

    # Pénalité train (par cas)
    star_gtstar = df[(df["model"] == "Model_Star") & (df["scenario"] == "GT_star")]\
        .set_index("case")["cldice"]
    fixed_gtstar = df[(df["model"] == "Model_Minus_Fixed") & (df["scenario"] == "GT_star")]\
        .set_index("case")["cldice"]
    common_train = star_gtstar.index.intersection(fixed_gtstar.index)
    train_penalty = (star_gtstar.loc[common_train] - fixed_gtstar.loc[common_train]).values

    # Pénalité eval (par cas)
    star_gtminus = df[(df["model"] == "Model_Star") & (df["scenario"] == "GT_minus_test")]\
        .set_index("case")["cldice"]
    common_eval = star_gtstar.index.intersection(star_gtminus.index)
    eval_penalty = (star_gtstar.loc[common_eval] - star_gtminus.loc[common_eval]).values

    if len(train_penalty) == 0 or len(eval_penalty) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Bar chart comparatif ──
    ax = axes[0]
    categories = ["Pénalité\nd'entraînement\n(Fixed vs Star sur GT*)",
                  "Pénalité\nd'évaluation\n(Star: GT* → GT_minus_test)"]
    means = [np.mean(train_penalty), np.mean(eval_penalty)]
    stds = [np.std(train_penalty, ddof=1), np.std(eval_penalty, ddof=1)]
    colors = ["#2196F3", "#F44336"]

    bars = ax.bar(categories, means, yerr=stds, capsize=8, color=colors, alpha=0.8, width=0.5)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_ylabel("Δ Dice (positif = pénalité)")
    ax.set_title("Outcome 6 — Asymétrie train vs eval\n(pénalité moyenne ± std)")

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.005,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # ── Violin comparatif ──
    ax2 = axes[1]
    combined = pd.DataFrame({
        "pénalité": np.concatenate([train_penalty, eval_penalty]),
        "type": (["Train (GT biaisée → entraînement)"] * len(train_penalty) +
                 ["Eval (GT biaisée → évaluation)"] * len(eval_penalty)),
    })
    sns.violinplot(
        data=combined, x="type", y="pénalité",
        palette={"Train (GT biaisée → entraînement)": "#2196F3",
                 "Eval (GT biaisée → évaluation)": "#F44336"},
        inner="box", ax=ax2,
    )
    ax2.axhline(0, color="black", lw=0.8, linestyle="--")
    ax2.set_title("Distribution des pénalités par cas")
    ax2.tick_params(axis="x", rotation=10)

    plt.suptitle("Outcome 6 : Asymétrie train/test — où la GT biaisée pénalise-t-elle le plus ?",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outcome6_train_test_asymmetry.png"), dpi=150)
    plt.close()
    print("  outcome6_train_test_asymmetry.png")


# ─────────────────────────────────────────────────────────────────
# Outcome 7 — Spécificité de la robustesse
# ─────────────────────────────────────────────────────────────────

def plot_robustness_specificity(df: pd.DataFrame, output_dir: str) -> None:
    """Outcome 7 : la robustesse acquise à un type de dégradation est-elle générique ?

    Compare les modèles morpho-only et omission-only sur des scénarios GT
    qui correspondent ou non à leur type d'entraînement.
    Requiert les prédictions de Model_Minus_MorphoOnly et Model_Minus_OmissionOnly.
    """
    specificity_models = {"Model_Minus_MorphoOnly", "Model_Minus_OmissionOnly"}
    available = set(df["model"].unique())
    if not specificity_models.issubset(available):
        print(f"  [SKIP] plot_robustness_specificity — "
              f"manquent: {specificity_models - available}")
        print("         Lancez les trainers MorphoOnly / OmissionOnly pour activer ce plot.")
        return

    subset = df[df["model"].isin(specificity_models | {"Model_Star"})]
    scenarios = [s for s in SCENARIO_ORDER if s in subset["scenario"].unique()]

    fig, ax = plt.subplots(figsize=(10, 5))
    local_palette = {
        "Model_Star": "#4CAF50",
        "Model_Minus_MorphoOnly": "#FF5722",
        "Model_Minus_OmissionOnly": "#9C27B0",
    }
    sns.barplot(
        data=subset, x="scenario", y="cldice", hue="model",
        palette=local_palette, order=scenarios, ax=ax,
        capsize=0.05, errorbar="sd",
    )
    ax.set_title("Outcome 7 — Spécificité de la robustesse\n"
                 "MorphoOnly vs OmissionOnly selon le type de dégradation GT")
    ax.set_ylabel("Dice moyen")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outcome7_robustness_specificity.png"), dpi=150)
    plt.close()
    print("  outcome7_robustness_specificity.png")


# ─────────────────────────────────────────────────────────────────
# Outcome 3 — Robustesse induite par le bruit stochastique
# ─────────────────────────────────────────────────────────────────

def plot_stochastic_robustness(df: pd.DataFrame, output_dir: str) -> None:
    """Outcome 3 : Model_Minus_Stoch vs Model_Star sur les scénarios dégradés.

    Un score Stoch > Star sur GT_minus indique que le bruit d'entraînement
    a rendu le modèle plus tolérant aux annotations imparfaites au test.
    """
    needed = {"Model_Star", "Model_Minus_Stoch"}
    if not needed.issubset(set(df["model"].unique())):
        print(f"  [SKIP] plot_stochastic_robustness — modèles manquants")
        return

    degraded_scenarios = [s for s in ["GT_minus_test"]
                          if s in df["scenario"].unique()]
    if not degraded_scenarios:
        return

    subset = df[
        df["model"].isin(needed) &
        df["scenario"].isin(degraded_scenarios)
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=subset, x="scenario", y="cldice", hue="model",
        palette=PALETTE, order=degraded_scenarios, ax=ax,
        capsize=0.05, errorbar="sd",
    )

    # Annoter les deltas avec tests statistiques
    for i, sc in enumerate(degraded_scenarios):
        sc_df = df[df["scenario"] == sc]
        star_v = sc_df[sc_df["model"] == "Model_Star"].set_index("case")["cldice"]
        stoch_v = sc_df[sc_df["model"] == "Model_Minus_Stoch"].set_index("case")["cldice"]
        common = star_v.index.intersection(stoch_v.index)
        if len(common) >= 5:
            annot = _wilcoxon_annotation(star_v.loc[common].values, stoch_v.loc[common].values)
            ax.text(i, ax.get_ylim()[1] * 0.98, annot,
                    ha="center", fontsize=7.5, color="gray", va="top")

    ax.set_title("Outcome 3 — Robustesse induite par le bruit stochastique\n"
                 "Model_Minus_Stoch vs Model_Star sur GT dégradée")
    ax.set_ylabel("Dice moyen")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outcome3_stochastic_robustness.png"), dpi=150)
    plt.close()
    print("  outcome3_stochastic_robustness.png")


# ─────────────────────────────────────────────────────────────────
# Récapitulatif statistique étendu
# ─────────────────────────────────────────────────────────────────

def statistical_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Tests de Wilcoxon par paires de modèles, FDR Benjamini-Hochberg, Cohen's d.

    Ce tableau est la base statistique de tous les outcomes comparatifs.
    """
    os.makedirs(output_dir, exist_ok=True)

    models = sorted(df["model"].unique())
    scenarios = sorted(df["scenario"].unique())
    rows = []

    for scenario in scenarios:
        sc_df = df[df["scenario"] == scenario]
        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                a_vals = sc_df[sc_df["model"] == model_a].set_index("case")["cldice"]
                b_vals = sc_df[sc_df["model"] == model_b].set_index("case")["cldice"]
                common = a_vals.index.intersection(b_vals.index)
                if len(common) < 5:
                    continue

                a = a_vals.loc[common].values
                b = b_vals.loc[common].values
                diff = b - a
                std_diff = np.std(diff, ddof=1)

                try:
                    _, p = wilcoxon(a, b)
                except ValueError:
                    p = np.nan

                cohens_d = float(np.mean(diff) / std_diff) if std_diff > 0 else np.nan

                rows.append({
                    "scenario": scenario,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_cases": len(common),
                    "mean_a": round(float(np.mean(a)), 4),
                    "mean_b": round(float(np.mean(b)), 4),
                    "delta_b_minus_a": round(float(np.mean(diff)), 4),
                    "cohens_d": round(cohens_d, 3) if not np.isnan(cohens_d) else None,
                    "p_raw": round(float(p), 4) if not np.isnan(p) else None,
                })

    if not rows:
        print("  Pas assez de données pour les tests statistiques.")
        return

    stat_df = pd.DataFrame(rows)

    # FDR Benjamini-Hochberg
    p_vals = stat_df["p_raw"].fillna(1.0).values
    n = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    p_fdr = p_vals.copy().astype(float)
    p_fdr[sorted_idx] = p_vals[sorted_idx] * n / (np.arange(n) + 1.0)
    for k in range(n - 2, -1, -1):
        p_fdr[sorted_idx[k]] = min(p_fdr[sorted_idx[k]], p_fdr[sorted_idx[k + 1]])
    stat_df["p_fdr"] = np.round(np.clip(p_fdr, 0, 1), 4)
    stat_df["significant"] = stat_df["p_fdr"] < 0.05

    stat_path = os.path.join(output_dir, "statistical_tests.csv")
    stat_df.to_csv(stat_path, index=False)

    print(f"\n{'='*70}")
    print("TESTS STATISTIQUES (Wilcoxon apparié, FDR BH, Cohen's d)")
    print(f"{'='*70}")
    print(stat_df.to_string(index=False))
    print(f"\nSauvegardé : {stat_path}")


# ─────────────────────────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────────────────────────

def plot_grid_search_heatmaps(df: pd.DataFrame, output_dir: str) -> None:
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
        ax.set_title("Phase 1 — Morpho seule — Dice sur GT*\n(Outcome 2 & 5 : seuils + impact morpho)")
        for i in range(len(radii)):
            for j in range(len(probs)):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        color="white" if matrix[i, j] < 0.55 else "black",
                        fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Dice Score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmap_phase1_morpho.png"), dpi=150)
        plt.close()
        print("  heatmap_phase1_morpho.png")

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
        ax.set_title("Phase 2 — Omission seule — Dice sur GT*\n(Outcome 2 & 5 : seuils + impact omission)")
        for i in range(len(sizes)):
            for j in range(len(probs)):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        color="white" if matrix[i, j] < 0.55 else "black",
                        fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Dice Score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmap_phase2_omission.png"), dpi=150)
        plt.close()
        print("  heatmap_phase2_omission.png")

    # ── Phase 3 : Bar chart ──
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
        ax.set_title("Phase 3 — Combinaisons morpho+omission — Dice sur GT*")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "barchart_phase3.png"), dpi=150)
        plt.close()
        print("  barchart_phase3.png")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agrégation et visualisation — 7 outcomes"
    )
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    output_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # ── Grid search ──
    gs_path = os.path.join(args.results_dir, "grid_search_results.csv")
    if os.path.exists(gs_path):
        print("\n[Grid search] Génération des heatmaps (outcomes 2 & 5)...")
        gs_df = pd.read_csv(gs_path)
        plot_grid_search_heatmaps(gs_df, output_dir)
    else:
        print(f"[SKIP] {gs_path} absent")

    # ── Cross-évaluation ──
    ce_path = os.path.join(args.results_dir, "cross_evaluation.csv")
    if not os.path.exists(ce_path):
        print(f"[SKIP] {ce_path} absent — cross-évaluation requise")
        return

    ce_df = pd.read_csv(ce_path)
    print(f"\nModèles disponibles : {sorted(ce_df['model'].unique())}")
    print(f"Scénarios disponibles : {sorted(ce_df['scenario'].unique())}")

    print("\n[Visualisations outcomes cross-évaluation]")
    plot_quality_vs_variability(ce_df, output_dir)       # outcome 1
    plot_stochastic_robustness(ce_df, output_dir)        # outcome 3
    plot_evaluation_penalty(ce_df, output_dir)           # outcome 4
    plot_train_test_asymmetry(ce_df, output_dir)         # outcome 6
    plot_robustness_specificity(ce_df, output_dir)       # outcome 7

    print("\n[Tests statistiques appariés]")
    statistical_comparison(ce_df, args.results_dir)

    print(f"\nAgrégation terminée. Figures dans {output_dir}/")


if __name__ == "__main__":
    main()
