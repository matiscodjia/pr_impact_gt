#!/usr/bin/env python3
"""Fold-by-fold stability of the ranking reversal (a proxy, NOT a training-seed study).

The evaluation is out of fold: in fold k, M0, M1 and M2 are three separately trained
networks (own initialisation, 68 training cases) scored on the same ~17 held-out cases.
Recomputing the four retained contrasts fold by fold tells whether the signs of
Delta(GT*), Delta(GT-) and of the interaction hold across five independent trainings per
regime. This is not a seed variance: both the training data and the evaluated cases
change between folds, so the spread mixes seed and case difficulty (see seed_variance.py
for the seed itself).

Input  : results/metrics.csv (eval_kind == "oof")
Output : results/fold_stability.csv + printed table

    python analysis/fold_stability.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prgt.stats import paired_cohens_d, wilcoxon_p_safe

CONTRASTS = [("hd95", "M1_Omission"), ("hd95", "M2_Drift_mu0"),
             ("nsd", "M1_Omission"), ("nsd", "M2_Drift_mu0")]     # nsd = NSD@2mm
REF = "GT_minus_omission"


def _row(g, metric, other, fold):
    w = g.pivot_table(index="case", columns=["model", "scenario"], values=metric)
    d_star = (w[("M0_Star", "GT_star")] - w[(other, "GT_star")]).dropna()
    d_ref = (w[("M0_Star", REF)] - w[(other, REF)]).loc[d_star.index].dropna()
    d_star = d_star.loc[d_ref.index]
    iota = (d_ref - d_star).to_numpy()
    return {"metric": metric, "pair": f"M0_vs_{other}", "fold": fold, "n": len(iota),
            "delta_star": d_star.mean(), "delta_ref": d_ref.mean(),
            "sign_flip": bool(np.sign(d_star.mean()) != np.sign(d_ref.mean())),
            "interaction_mean": iota.mean(), "d": paired_cohens_d(iota),
            "frac_cases_same_sign": float((np.sign(iota) == np.sign(np.median(iota))).mean()),
            "p_raw": wilcoxon_p_safe(d_ref.to_numpy(), d_star.to_numpy())}


def main():
    df = pd.read_csv("results/metrics.csv")
    df = df[df.eval_kind == "oof"]
    rows = []
    for metric, other in CONTRASTS:
        for fold, g in df.groupby("fold"):
            rows.append(_row(g, metric, other, int(fold)))
        rows.append(_row(df, metric, other, "pooled"))
    out = pd.DataFrame(rows)
    out.to_csv("results/fold_stability.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))

    per_fold = out[out.fold != "pooled"]
    print("\nSummary (5 folds):")
    for (metric, pair), g in per_fold.groupby(["metric", "pair"], sort=False):
        lead = (g.delta_star < 0) if metric == "hd95" else (g.delta_star > 0)   # M0 ahead on GT*
        print(f"  {metric:5s} {pair:18s} M0 ahead on GT*: {int(lead.sum())}/5 | "
              f"reversal: {int(g.sign_flip.sum())}/5 | interaction of the same sign: "
              f"{int((np.sign(g.interaction_mean) == np.sign(g.interaction_mean.median())).sum())}/5 | "
              f"d in [{g.d.min():+.2f}, {g.d.max():+.2f}]")


if __name__ == "__main__":
    main()
