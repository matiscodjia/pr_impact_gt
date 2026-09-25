#!/usr/bin/env python3
"""P0.5 -- Figure 1 pour la soumission ISBI (composite, 2 panneaux).

Contraintes ISBI (4 pages) : une seule figure composite + une table. Tout en anglais.
Format vectoriel PDF, largeur colonne IEEE (~3.4 in), police >=8pt à la taille finale,
palette lisible en niveaux de gris (linestyle/marker distincts, pas seulement la
couleur).

Panel (a) -- the HD95 reversal: per-case paired M0 vs M1 under `star` and `omission`,
mean +/- bootstrap CI overlaid on faint per-case connecting lines, showing the crossing.
Panel (b) -- the mu sweep (boundary_drift, results/noise_calibration.csv), cleaned
version of Table 5.1: 4 small multiples (clDice, NSD@0.5, HD95, Betti0) vs mu, showing
the metric dissociation (unsigned/symmetric NSD vs. signed/monotone clDice&Betti0 vs.
saturating HD95).

Usage
-----
    python analysis/figures_isbi.py --results_dir results --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from rank_reversal import build_wide, load_pooled  # noqa: E402
from stats_utils import bca_mean_diff  # noqa: E402

MIN_PT = 8  # exigence ISBI : toute police >= 8pt à la taille finale colonne (~3.4in)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": MIN_PT,
    "axes.labelsize": MIN_PT,
    "axes.titlesize": MIN_PT,
    "xtick.labelsize": MIN_PT,
    "ytick.labelsize": MIN_PT,
    "legend.fontsize": MIN_PT,
    "pdf.fonttype": 42,  # texte éditable, pas de courbes -- exigence IEEE/ISBI usuelle
    "axes.linewidth": 0.6,
})

M0, M1 = "M0_Star", "M1_Omission"
COLOR = {M0: "#000000", M1: "#555555"}
MARKER = {M0: "o", M1: "s"}
LINESTYLE = {M0: "-", M1: "--"}


def panel_a(ax, p: pd.DataFrame, seed: int, n_boot: int = 2000) -> None:
    wide = build_wide(p, "hd95")
    star = {m: wide[(m, "GT_star")] for m in (M0, M1)}
    om = {m: wide[(m, "GT_minus_omission")] for m in (M0, M1)}
    common = star[M0].dropna().index.intersection(star[M1].dropna().index) \
        .intersection(om[M0].dropna().index).intersection(om[M1].dropna().index)

    x = np.array([0.0, 1.0])
    for m in (M0, M1):
        v_star = star[m].loc[common].values
        v_om = om[m].loc[common].values
        # cas individuels, discrets (nuage jitterisé), pour montrer la dispersion sans spaghetti
        jitter = (np.random.default_rng(0).uniform(-0.03, 0.03, size=len(common)))
        ax.plot(np.column_stack([x[0] + jitter, x[1] + jitter]).T,
                np.column_stack([v_star, v_om]).T,
                color=COLOR[m], alpha=0.08, linewidth=0.4, zorder=1)

        mean_star, ci_lo_s, ci_hi_s = bca_mean_diff(v_star, n_boot=n_boot, seed=seed)
        mean_om, ci_lo_o, ci_hi_o = bca_mean_diff(v_om, n_boot=n_boot, seed=seed)
        means = [v_star.mean(), v_om.mean()]
        err_lo = [v_star.mean() - ci_lo_s, v_om.mean() - ci_lo_o]
        err_hi = [ci_hi_s - v_star.mean(), ci_hi_o - v_om.mean()]
        ax.errorbar(x, means, yerr=[err_lo, err_hi], color=COLOR[m],
                    linestyle=LINESTYLE[m], marker=MARKER[m], markersize=3.5,
                    linewidth=1.3, capsize=2, label=m.replace("_", " "), zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["$GT^{*}$\n(star)", "$GT^{-}$\n(omission)"], fontsize=MIN_PT)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("HD95 (mm, log scale)\n($\\downarrow$ better)", fontsize=MIN_PT)
    ax.set_title("(a) HD95 rank reversal, $M_0$ vs $M_1$", loc="left",
                 fontweight="bold", fontsize=MIN_PT)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4, fontsize=MIN_PT,
              borderaxespad=0.2, handletextpad=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def panel_b(fig, gridspec_slot, calib_csv: str) -> None:
    df = pd.read_csv(calib_csv)
    df = df[df["family"] == "boundary_drift"].sort_values("mu")
    metrics = [("clDice", "clDice ($\\uparrow$)"), ("NSD@0.5", "NSD@0.5 ($\\uparrow$)"),
               ("HD95", "HD95 mm ($\\downarrow$)"), ("Betti0", "Betti$_0$ ($\\downarrow$)")]

    # 2x2 (pas 1x4) : à largeur colonne IEEE fixe (~3.4in), 4 sous-graphes côte à côte
    # ne laissent pas assez de place pour des ticks >=8pt lisibles. Ligne 0 réservée au
    # titre du panneau (sinon il chevauche la première rangée de sous-graphes).
    inner = gridspec_slot.subgridspec(3, 2, wspace=0.7, hspace=0.9,
                                       height_ratios=[0.10, 1.0, 1.0])
    for i, (col, label) in enumerate(metrics):
        ax = fig.add_subplot(inner[1 + i // 2, i % 2])
        ax.plot(df["mu"], df[col], color="#000000", marker="o", markersize=3,
                linewidth=1.1)
        ax.axvline(0.0, color="#bbbbbb", linewidth=0.5, zorder=0)
        ax.set_xlabel("$\\mu$", labelpad=1, fontsize=MIN_PT)
        ax.set_ylabel(label, labelpad=1, fontsize=MIN_PT)
        ax.set_xticks([-1, 0, 1])
        ax.tick_params(pad=1.0, labelsize=MIN_PT)
        ax.spines[["top", "right"]].set_visible(False)

    bbox = gridspec_slot.get_position(fig)
    fig.text(0.03, bbox.y1 - 0.008, "(b) The $\\mu$-sweep dissociates the instruments",
              fontweight="bold", fontsize=MIN_PT, va="top")


def main():
    ap = argparse.ArgumentParser(description="P0.5 -- Figure 1 ISBI (composite)")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = load_pooled(args.results_dir)
    calib_csv = os.path.join(args.results_dir, "noise_calibration.csv")
    if not os.path.exists(calib_csv):
        raise SystemExit(f"[FATAL] {calib_csv} absent -- lancer calibrate_noise.py d'abord.")

    fig = plt.figure(figsize=(3.4, 5.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.18,
                          top=0.94, bottom=0.08, left=0.22, right=0.97)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_a(ax_a, p, args.seed)
    panel_b(fig, gs[1, 0], calib_csv)

    out_dir = args.out or os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig1_isbi.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
