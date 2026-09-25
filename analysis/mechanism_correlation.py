#!/usr/bin/env python3
"""Vérification directe du mécanisme du retournement HD95 (demandée après relecture).

Thèse à vérifier : sous `omission`, la référence perd des branches distales ; M0
(propre) les prédit quand même, elles deviennent des faux positifs éloignés, et HD95
explose proportionnellement à ce qui a été omis. Si le mécanisme est réel, le volume
omis dans la référence (par cas) doit être corrélé à la pénalité HD95 de M0
(hd95(M0,omission)-hd95(M0,star)) -- pas seulement en moyenne, cas par cas.

Sources : `results/reference_severity.csv` (dV_ref_vs_star, scenario=GT_minus_omission,
déjà signé négatif car omission ne fait que retirer du volume) et `results/metrics.csv`
(hd95 de M0 sous star et omission). Aucun nouveau calcul sur les masques -- pur
recoupement de tables déjà produites, CPU trivial.

Sortie : results/mechanism_correlation.csv (une ligne : les deux coefficients + IC
bootstrap), et un nuage de points sauvegardé pour inspection visuelle si demandé.

Usage
-----
    python analysis/mechanism_correlation.py --results_dir results --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import results_store as rs  # noqa: E402
from rank_reversal import build_wide, load_pooled  # noqa: E402


def _bootstrap_ci_corr(x, y, func, n_boot=10000, seed=42):
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([func(x[ix], y[ix])[0] for ix in idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description="Corrélation volume omis / pénalité HD95 de M0")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    sev_path = os.path.join(args.results_dir, "reference_severity.csv")
    if not os.path.exists(sev_path):
        raise SystemExit(f"[FATAL] {sev_path} absent -- lancer reference_severity.py (P0.3) d'abord.")
    sev = pd.read_csv(sev_path)
    sev_om = sev[sev["scenario"] == "GT_minus_omission"].set_index("case")

    p = load_pooled(args.results_dir)
    wide_hd95 = build_wide(p, "hd95")
    m0_star = wide_hd95[("M0_Star", "GT_star")]
    m0_om = wide_hd95[("M0_Star", "GT_minus_omission")]
    penalty = (m0_om - m0_star).dropna()

    common = penalty.index.intersection(sev_om.index)
    penalty = penalty.loc[common]
    # dV_ref_vs_star est négatif sous omission (volume retiré) -- on prend la magnitude
    # ("volume omis", une fraction positive du volume star) pour une lecture directe.
    vol_omitted = (-sev_om.loc[common, "dV_ref_vs_star"]).values
    hd95_penalty = penalty.values

    r_pearson, p_pearson = pearsonr(vol_omitted, hd95_penalty)
    rho_spearman, p_spearman = spearmanr(vol_omitted, hd95_penalty)
    r_ci = _bootstrap_ci_corr(vol_omitted, hd95_penalty,
                              lambda a, b: pearsonr(a, b), args.n_boot, args.seed)
    rho_ci = _bootstrap_ci_corr(vol_omitted, hd95_penalty,
                                lambda a, b: spearmanr(a, b), args.n_boot, args.seed)

    print(f"n={len(common)}")
    print(f"Pearson r  = {r_pearson:+.3f}  (IC95%=[{r_ci[0]:+.3f},{r_ci[1]:+.3f}])  p={p_pearson:.3g}")
    print(f"Spearman ρ = {rho_spearman:+.3f}  (IC95%=[{rho_ci[0]:+.3f},{rho_ci[1]:+.3f}])  p={p_spearman:.3g}")

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
