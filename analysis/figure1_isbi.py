#!/usr/bin/env python3
"""Figure 1 of the ISBI paper (one composite figure, two panels).

ISBI constraints (4 pages): a single composite figure and one table. Vector PDF at IEEE
column width (~3.4 in), every font >= 8 pt at final size, readable in grey scale
(distinct line styles and markers, not only colours).

Panel (a) -- the HD95 reversal: per-case paired M0 vs M1 under GT* and under omission,
mean +/- BCa interval over faint per-case lines, showing the crossing.
Panel (b) -- the mu sweep (boundary_drift r=1 p=0.5, GT- vs GT*, no model), mean over the
85 cases (grid rows of results/calibration_85_summary.csv, written by calibration_85.py):
four small multiples (clDice, NSD@0.5, HD95, Betti-0) against mu, showing how the metrics
dissociate (unsigned and symmetric NSD, signed and monotone clDice and Betti-0,
saturating HD95).

Output : results/figures/fig1_isbi.pdf (or <--out>/fig1_isbi.pdf)

    python analysis/figure1_isbi.py --results_dir results --seed 42
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prgt.stats import bca_mean_diff
from rank_reversal import build_wide, load_pooled

MIN_PT = 8  # ISBI: every font >= 8 pt at final column width (~3.4 in)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": MIN_PT,
    "axes.labelsize": MIN_PT,
    "axes.titlesize": MIN_PT,
    "xtick.labelsize": MIN_PT,
    "ytick.labelsize": MIN_PT,
    "legend.fontsize": MIN_PT,
    "pdf.fonttype": 42,  # embed text as text, not outlines (usual IEEE/ISBI requirement)
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
        # individual cases, faint and jittered, to show the spread without clutter
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
    df = df[(df["source"] == "grid") & (df["family"] == "boundary_drift")].sort_values("mu")
    metrics = [("clDice_mean", "clDice ($\\uparrow$)"), ("NSD@0.5_mean", "NSD@0.5 ($\\uparrow$)"),
               ("HD95_mean", "HD95 mm ($\\downarrow$)"), ("Betti0_mean", "Betti$_0$ ($\\downarrow$)")]

    # 2x2 (not 1x4): at a fixed IEEE column width (~3.4 in), four plots side by side leave
    # no room for readable 8 pt ticks. Row 0 is reserved for the panel title (otherwise it
    # overlaps the first row of plots).
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
    fig.text(0.03, bbox.y1 - 0.008, "(b) GT$^-$ vs. GT$^*$, no model: the $\\mu$-sweep",
              fontweight="bold", fontsize=MIN_PT, va="top")


def main():
    ap = argparse.ArgumentParser(description="Figure 1 of the ISBI paper")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="output directory (default: <results_dir>/figures)")
    args = ap.parse_args()

    p = load_pooled(args.results_dir)
    calib_csv = os.path.join(args.results_dir, "calibration_85_summary.csv")
    if not os.path.exists(calib_csv):
        raise SystemExit(f"[FATAL] {calib_csv} missing: run analysis/calibration_85.py first.")

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
