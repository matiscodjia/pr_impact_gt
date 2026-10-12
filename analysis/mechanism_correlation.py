#!/usr/bin/env python3
"""Direct check of the mechanism behind the HD95 reversal.

Claim: under the omission reference, distal branches are missing from GT-; M0 (trained
on clean labels) still predicts them, they become distant false positives, and HD95 grows
with what was omitted. If the mechanism is real, the volume omitted from the reference
(per case) must correlate with the HD95 penalty of M0, hd95(M0, omission) - hd95(M0, star),
case by case and not only on average.

Sources: results/reference_severity.csv (dV_ref_vs_star for GT_minus_omission, negative
because omission only removes volume) and results/metrics.csv (HD95 of M0 under GT* and
GT-). No mask is read: this only joins tables that already exist.

Output : results/mechanism_correlation.csv (one row: Pearson and Spearman coefficients
with bootstrap 95 % intervals).

    python analysis/mechanism_correlation.py --results_dir results --seed 42
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from prgt import results_store as rs
from rank_reversal import build_wide, load_pooled


def _bootstrap_ci_corr(x, y, func, n_boot=10000, seed=42):
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([func(x[ix], y[ix])[0] for ix in idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description="Correlation between omitted volume and the HD95 penalty of M0")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    sev_path = os.path.join(args.results_dir, "reference_severity.csv")
    if not os.path.exists(sev_path):
        raise SystemExit(f"[FATAL] {sev_path} missing: run analysis/reference_severity.py first.")
    sev = pd.read_csv(sev_path)
    sev_om = sev[sev["scenario"] == "GT_minus_omission"].set_index("case")

    p = load_pooled(args.results_dir)
    wide_hd95 = build_wide(p, "hd95")
    m0_star = wide_hd95[("M0_Star", "GT_star")]
    m0_om = wide_hd95[("M0_Star", "GT_minus_omission")]
    penalty = (m0_om - m0_star).dropna()

    common = penalty.index.intersection(sev_om.index)
    penalty = penalty.loc[common]
    # dV_ref_vs_star is negative under omission (removed volume); use its magnitude, the
    # omitted fraction of the clean volume.
    vol_omitted = (-sev_om.loc[common, "dV_ref_vs_star"]).values
    hd95_penalty = penalty.values

    r_pearson, p_pearson = pearsonr(vol_omitted, hd95_penalty)
    rho_spearman, p_spearman = spearmanr(vol_omitted, hd95_penalty)
    r_ci = _bootstrap_ci_corr(vol_omitted, hd95_penalty,
                              lambda a, b: pearsonr(a, b), args.n_boot, args.seed)
    rho_ci = _bootstrap_ci_corr(vol_omitted, hd95_penalty,
                                lambda a, b: spearmanr(a, b), args.n_boot, args.seed)

    print(f"n={len(common)}")
    print(f"Pearson r    = {r_pearson:+.3f}  (95% CI=[{r_ci[0]:+.3f},{r_ci[1]:+.3f}])  p={p_pearson:.3g}")
    print(f"Spearman rho = {rho_spearman:+.3f}  (95% CI=[{rho_ci[0]:+.3f},{rho_ci[1]:+.3f}])  p={p_spearman:.3g}")

    row = {
        "n": len(common), "pearson_r": r_pearson, "pearson_ci_low": r_ci[0],
        "pearson_ci_high": r_ci[1], "pearson_p": p_pearson,
        "spearman_rho": rho_spearman, "spearman_ci_low": rho_ci[0],
        "spearman_ci_high": rho_ci[1], "spearman_p": p_spearman,
        "git_commit": rs.git_commit(), "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path = os.path.join(args.results_dir, "mechanism_correlation.csv")
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
