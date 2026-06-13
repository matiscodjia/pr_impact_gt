"""Figures comparatives M0–M4 / Q1–Q3 à partir de la table de métriques.

Consommé par report.py (make_figures) et utilisable en standalone :
    python scripts/aggregate_results.py --results_dir results

Deux figures :
  - metrics_boxplots.png   : chaque métrique clé, boxplots model × scenario.
  - questions_summary.png  : un panneau par question (Q1/Q2/Q3), métrique appariée.

Robuste : ignore les modèles/scénarios/métriques absents (résultats partiels).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

_MODEL_ORDER = ["M0_Star", "M1_Omission", "M2_Drift_mu0",
                "M3_Drift_muMinus", "M4_Drift_muPlus"]
COLORS = {
    "M0_Star": "#4CAF50", "M1_Omission": "#FF9800", "M2_Drift_mu0": "#2196F3",
    "M3_Drift_muMinus": "#9C27B0", "M4_Drift_muPlus": "#E91E63",
}
_SCEN_ORDER = ["GT_star", "GT_minus_omission", "GT_minus_drift_neg", "GT_minus_drift_pos"]
_KEY_METRICS = [("cldice", "clDice ↑"), ("betti0", "Betti₀ ↓"),
                ("nsd05", "NSD@0.5mm ↑"), ("volume_delta", "ΔV signé")]


def _finite(series: pd.Series) -> np.ndarray:
    v = series.dropna().values.astype(float)
    return v[np.isfinite(v)]


# ── Figure 1 : boxplots model × scenario par métrique ────────────────────────

def _grouped_boxplot(ax, df, metric):
    scen = [s for s in _SCEN_ORDER if s in set(df["scenario"])]
    models = [m for m in _MODEL_ORDER if m in set(df["model"])]
    if not scen or not models:
        ax.axis("off"); return
    n = len(models); width = 0.8 / n
    for j, m in enumerate(models):
        data, pos = [], []
        for i, s in enumerate(scen):
            vals = _finite(df[(df["model"] == m) & (df["scenario"] == s)][metric])
            data.append(vals if len(vals) else [np.nan])
            pos.append(i + (j - (n - 1) / 2) * width)
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9,
                        patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set_facecolor(COLORS.get(m, "#888")); box.set_alpha(0.85)
        for med in bp["medians"]:
            med.set_color("black")
    ax.set_xticks(range(len(scen)))
    ax.set_xticklabels([s.replace("GT_", "").replace("minus_", "−") for s in scen],
                       rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_metric_boxplots(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metrics = [(m, l) for m, l in _KEY_METRICS if m in df.columns]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (m, l) in zip(axes.ravel(), metrics):
        _grouped_boxplot(ax, df, m); ax.set_title(l, fontweight="bold")
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    handles = [Patch(facecolor=COLORS[m], label=m)
               for m in _MODEL_ORDER if m in set(df["model"])]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles) or 1,
               frameon=False, fontsize=9)
    fig.suptitle("Métriques par modèle × scénario (M0–M4)", y=1.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(output_dir, "metrics_boxplots.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [figures] {out}")


# ── Figure 2 : un panneau par question, métrique appariée ────────────────────

def _box_pairs(ax, df, pairs, metric, title, ylabel):
    data, labels, colors = [], [], []
    for model, sc, lab in pairs:
        vals = _finite(df[(df["model"] == model) & (df["scenario"] == sc)][metric]) \
            if metric in df.columns else []
        if len(vals) == 0:
            continue
        data.append(vals); labels.append(lab); colors.append(COLORS.get(model, "#888"))
    if not data:
        ax.axis("off"); ax.set_title(f"{title}\n(n/a)", fontsize=10); return
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c); box.set_alpha(0.85)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)


def plot_question_summary(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    _box_pairs(axes[0], df,
               [("M0_Star", "GT_star", "M0 | GT*"),
                ("M0_Star", "GT_minus_omission", "M0 | GT⁻")],
               "cldice", "Q1 — pénalité d'éval (omission)", "clDice ↑")
    _box_pairs(axes[1], df,
               [("M0_Star", "GT_star", "M0"),
                ("M2_Drift_mu0", "GT_star", "M2 (μ=0)")],
               "nsd05", "Q2 — robustesse sur GT* propre", "NSD@0.5 ↑")
    _box_pairs(axes[2], df,
               [("M0_Star", "GT_minus_drift_neg", "M0"),
                ("M3_Drift_muMinus", "GT_minus_drift_neg", "M3 (μ<0)")],
               "nsd05", "Q3 — colle au biais (GT⁻ biaisée)", "NSD@0.5 ↑")
    fig.suptitle("Synthèse Q1/Q2/Q3 — métrique appariée", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(output_dir, "questions_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [figures] {out}")


def main():
    import results_store as rs
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    df = rs.load(args.results_dir)
    if df.empty:
        print(f"[SKIP] {rs.CSV_NAME} vide ou absent.")
        return
    oof = df[df["eval_kind"] == "oof"]
    base = oof if not oof.empty else df
    p = base.groupby(["model", "scenario", "case"], as_index=False).mean(numeric_only=True)

    figs = os.path.join(args.results_dir, "figures")
    plot_metric_boxplots(p, figs)
    plot_question_summary(p, figs)


if __name__ == "__main__":
    main()
