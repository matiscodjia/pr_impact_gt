#!/usr/bin/env python3
"""Stabilité du renversement pli par pli (critique R4b, proxy -- PAS une étude de graine).

L'évaluation est out-of-fold : dans le pli k, M0, M1 et M2 sont trois réseaux entraînés
séparément (initialisation propre, 68 cas d'entraînement) et scorés sur les mêmes ~17 cas
tenus à l'écart. Recalculer les quatre contrastes retenus pli par pli dit si le signe de
Δ(GT*), de Δ(GT-) et de l'interaction tient à travers cinq entraînements indépendants par
régime. Ce n'est pas une variance de graine : les données d'entraînement ET les cas
évalués changent d'un pli à l'autre, donc la dispersion mélange graine et difficulté.

Entrée  : results/metrics.csv (eval_kind == "oof")
Sortie  : results/fold_stability.csv + tableau imprimé

    python analysis/fold_stability.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_utils import paired_cohens_d, wilcoxon_p_safe  # noqa: E402

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
    print("\nRésumé (5 plis) :")
    for (metric, pair), g in per_fold.groupby(["metric", "pair"], sort=False):
        lead = (g.delta_star < 0) if metric == "hd95" else (g.delta_star > 0)   # M0 en tête sur GT*
        print(f"  {metric:5s} {pair:18s} M0 en tête sur GT* : {int(lead.sum())}/5 | "
              f"renversement : {int(g.sign_flip.sum())}/5 | interaction de même signe : "
              f"{int((np.sign(g.interaction_mean) == np.sign(g.interaction_mean.median())).sum())}/5 | "
              f"d ∈ [{g.d.min():+.2f}, {g.d.max():+.2f}]")


if __name__ == "__main__":
    main()
