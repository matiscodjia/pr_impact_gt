#!/usr/bin/env python3
"""Variance de graine d'ENTRAÎNEMENT : l'interaction et l'écart sur référence propre
sont-ils stables d'un réplicat à l'autre ?

Réplicats : graine 0 = l'étude (results/metrics.csv : M0_Star, M1_Omission, M2_Drift_mu0) ;
graines 1, 2... = results_seeds/<modèle>/metrics.csv (noms M0_Star_s1, ...), produits par
scripts/cluster/launch_gpu.sh. Les analyses portent sur les cas COMMUNS à tous les
réplicats et tous les modèles (avec un sous-ensemble de plis, n < 85 : à déclarer).

Pour chaque réplicat, métrique (HD95, NSD@2) et paire (M0 vs M1, M0 vs M2) :
    delta_star  = moyenne_cas[ m(M0|GT*) - m(Mk|GT*) ]           (écart sur référence propre)
    delta_omis  = moyenne_cas[ m(M0|GT-) - m(Mk|GT-) ]
    interaction = delta_omis - delta_star, d de Cohen apparié, changement de signe
Puis, entre réplicats : moyenne / écart-type / min / max de delta_star et de l'interaction,
et nombre de réplicats où l'écart change de signe. C'est ce qui répond à « −0,12 mm de HD95
sur référence propre dépasse-t-il la variance de graine ? » -- sans prétendre à un IC :
2 à 3 réplicats ne permettent qu'une plage, pas un intervalle.

Sortie : results/seed_variance.csv

Usage : python analysis/seed_variance.py [--seeds-dir results_seeds]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_utils import paired_cohens_d  # noqa: E402

BASE = {"M0": "M0_Star", "M1": "M1_Omission", "M2": "M2_Drift_mu0"}
METRICS = ("hd95", "nsd")


def load(seeds_dir):
    frames = [pd.read_csv("results/metrics.csv").assign(seed=0)]
    for f in sorted(glob.glob(os.path.join(seeds_dir, "*", "metrics.csv"))):
        d = pd.read_csv(f)
        d["seed"] = d.model.str.extract(r"_s(\d+)$")[0].astype(float)
        frames.append(d.dropna(subset=["seed"]).assign(seed=lambda x: x.seed.astype(int)))
    df = pd.concat(frames, ignore_index=True)
    df = df[df.scenario.isin(["GT_star", "GT_minus_omission"])]
    df["role"] = df.model.str.replace(r"_s\d+$", "", regex=True).map({v: k for k, v in BASE.items()})
    return df.dropna(subset=["role"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-dir", default="results_seeds")
    args = ap.parse_args()
    df = load(args.seeds_dir)

    # cas communs : présents pour les 3 modèles, les 2 références, dans TOUS les réplicats
    need = df.groupby(["seed", "case"]).apply(lambda g: g[["role", "scenario"]].drop_duplicates().shape[0] == 6)
    ok = need.unstack("seed").fillna(False).all(axis=1)
    common = ok[ok].index
    seeds = sorted(int(s) for s in df.seed.unique())
    print(f"réplicats: {seeds} | cas communs: {len(common)}")
    df = df[df.case.isin(common)]

    rows = []
    for seed in seeds:
        d = df[df.seed == seed]
        for metric in METRICS:
            w = d.pivot_table(index=["case", "scenario"], columns="role", values=metric).reset_index()
            star = w[w.scenario == "GT_star"].set_index("case")
            omis = w[w.scenario == "GT_minus_omission"].set_index("case")
            for other in ("M1", "M2"):
                d_star = star["M0"] - star[other]
                d_om = omis["M0"] - omis[other]
                iota = d_om - d_star
                rows.append({"seed": seed, "metric": metric, "pair": f"M0_vs_{other}", "n": len(iota),
                             "delta_star": d_star.mean(), "delta_omis": d_om.mean(),
                             "interaction": iota.mean(), "d": paired_cohens_d(iota.values),
                             "sign_flip": bool(np.sign(d_star.mean()) != np.sign(d_om.mean()))})
    per = pd.DataFrame(rows)
    summ = []
    for (metric, pair), g in per.groupby(["metric", "pair"]):
        summ.append({"seed": "ALL", "metric": metric, "pair": pair, "n": int(g.n.iloc[0]),
                     "delta_star": g.delta_star.mean(), "delta_star_sd": g.delta_star.std(ddof=1),
                     "delta_star_min": g.delta_star.min(), "delta_star_max": g.delta_star.max(),
                     "delta_omis": g.delta_omis.mean(),
                     "interaction": g.interaction.mean(), "interaction_sd": g.interaction.std(ddof=1),
                     "interaction_min": g.interaction.min(), "interaction_max": g.interaction.max(),
                     "sign_flip": int(g.sign_flip.sum())})
    out = pd.concat([per, pd.DataFrame(summ)], ignore_index=True)
    out.to_csv("results/seed_variance.csv", index=False)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print(per.round(4).to_string(index=False))
    print("\n-- entre réplicats (sign_flip = nombre de réplicats avec inversion) --")
    print(pd.DataFrame(summ).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
