#!/usr/bin/env python3
"""Training-seed variance: are the interaction and the clean-reference gap stable across
independent trainings of the same regime?

Replicates: seed 0 = the study (results/metrics.csv: M0_Star, M1_Omission, M2_Drift_mu0);
seeds 1, 2, ... = results/cluster_runs/<model>/metrics.csv (models M0_Star_s1, ...), scored on
the GPU node (cluster/launch_gpu.sh) and copied there. The analysis uses the cases COMMON to every
replicate and every model (a subset of folds, so n < 85 must be reported).

For each replicate, metric (HD95, NSD@2) and pair (M0 vs M1, M0 vs M2):
    delta_star  = mean over cases of m(M0|GT*) - m(Mk|GT*)     (gap on the clean reference)
    delta_omis  = mean over cases of m(M0|GT-) - m(Mk|GT-)
    interaction = delta_omis - delta_star, paired Cohen's d, sign change
The sign change is also judged by the study's MDD rule (analysis/mdd_crossing.py): the
crossing is resolved if |delta_omis| >= MDD = (z_.975 + z_.80) sd(iota) / sqrt(n), and
strict if, in addition, the raw Wilcoxon p(iota) < 0.05. Across replicates: mean / standard
deviation / min / max of delta_star and of the interaction, and the number of replicates
with a sign change (or a strict reversal). Two or three replicates give a range, not a
confidence interval.

Output : results/seed_variance.csv

    python analysis/seed_variance.py [--seeds-dir results/cluster_runs]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

from prgt.stats import paired_cohens_d

BASE = {"M0": "M0_Star", "M1": "M1_Omission", "M2": "M2_Drift_mu0"}
METRICS = ("hd95", "nsd")
Z = norm.ppf(0.975) + norm.ppf(0.80)   # MDD at 80 % power, two-sided 5 %


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
    ap.add_argument("--seeds-dir", default="results/cluster_runs")
    args = ap.parse_args()
    df = load(args.seeds_dir)

    # common cases: present for the 3 models and the 2 references in EVERY replicate
    need = df.groupby(["seed", "case"]).apply(lambda g: g[["role", "scenario"]].drop_duplicates().shape[0] == 6)
    ok = need.unstack("seed").fillna(False).all(axis=1)
    common = ok[ok].index
    seeds = sorted(int(s) for s in df.seed.unique())
    print(f"replicates: {seeds} | common cases: {len(common)}")
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
                flip = bool(np.sign(d_star.mean()) != np.sign(d_om.mean()))
                mdd = Z * iota.std(ddof=1) / np.sqrt(len(iota))
                p = wilcoxon(iota.values).pvalue
                rows.append({"seed": seed, "metric": metric, "pair": f"M0_vs_{other}", "n": len(iota),
                             "delta_star": d_star.mean(), "delta_omis": d_om.mean(),
                             "interaction": iota.mean(), "d": paired_cohens_d(iota.values),
                             "sign_flip": flip, "mdd": mdd, "p_wilcoxon": p,
                             "resolved": flip and abs(d_om.mean()) >= mdd,
                             "strict": flip and abs(d_om.mean()) >= mdd and p < 0.05})
    per = pd.DataFrame(rows)
    summ = []
    for (metric, pair), g in per.groupby(["metric", "pair"]):
        summ.append({"seed": "ALL", "metric": metric, "pair": pair, "n": int(g.n.iloc[0]),
                     "delta_star": g.delta_star.mean(), "delta_star_sd": g.delta_star.std(ddof=1),
                     "delta_star_min": g.delta_star.min(), "delta_star_max": g.delta_star.max(),
                     "delta_omis": g.delta_omis.mean(),
                     "interaction": g.interaction.mean(), "interaction_sd": g.interaction.std(ddof=1),
                     "interaction_min": g.interaction.min(), "interaction_max": g.interaction.max(),
                     "sign_flip": int(g.sign_flip.sum()), "strict": int(g.strict.sum())})
    out = pd.concat([per, pd.DataFrame(summ)], ignore_index=True)
    out.to_csv("results/seed_variance.csv", index=False)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print(per.round(4).to_string(index=False))
    print("\n-- across replicates (sign_flip / strict = number of replicates with a sign change /"
          " with a change beyond the MDD and p < 0.05) --")
    print(pd.DataFrame(summ).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
