#!/usr/bin/env python3
"""Does the HD95 / NSD@2 reversal survive largest-connected-component (LCC) post-processing?

Reads results/metrics_lcc.csv (written by lcc_metrics.py: 1020 pairs x {none, lcc}) and
reruns the interaction test of rank_reversal.py (GT* -> omission, M0 vs M1 and M0 vs M2)
separately for each post-processing, for HD95 and NSD@2. Also reports what LCC costs in
clDice (with vs without LCC, per model, under GT* and under omission).

Outputs: results/lcc_reversal_check.csv, results/lcc_cldice_cost.csv

    python analysis/lcc_reversal.py --results_dir results --seed 42
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from prgt.stats import bca_median_diff, bh_fdr, paired_cohens_d, wilcoxon_p_safe

PAIRS = [("M0_Star", "M1_Omission"), ("M0_Star", "M2_Drift_mu0")]
METRICS_REVERSAL = ["hd95", "nsd"]


def build_wide(df: pd.DataFrame, postproc: str, metric: str) -> pd.DataFrame:
    sub = df[df["postprocessing"] == postproc]
    return sub.pivot_table(index="case", columns=["model", "scenario"], values=metric)


def interaction_test(wide, model_a, model_b, s1, s2, n_boot, seed):
    a1, b1 = wide[(model_a, s1)], wide[(model_b, s1)]
    a2, b2 = wide[(model_a, s2)], wide[(model_b, s2)]
    common = a1.dropna().index.intersection(b1.dropna().index) \
        .intersection(a2.dropna().index).intersection(b2.dropna().index)
    d1 = (a1.loc[common] - b1.loc[common]).values
    d2 = (a2.loc[common] - b2.loc[common]).values
    interaction = d2 - d1
    p_raw = wilcoxon_p_safe(d2, d1)
    d_eff = paired_cohens_d(interaction)
    med, ci_lo, ci_hi = bca_median_diff(interaction, n_boot=n_boot, seed=seed)
    return {
        "delta_s1": float(np.mean(d1)), "delta_s2": float(np.mean(d2)),
        "interaction_median": med, "d": d_eff, "ci_low": ci_lo, "ci_high": ci_hi,
        "p_raw": p_raw, "n": len(common),
    }


def main():
    ap = argparse.ArgumentParser(description="Does the reversal survive LCC post-processing?")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    path = os.path.join(args.results_dir, "metrics_lcc.csv")
    df = pd.read_csv(path)

    rows = []
    for metric in METRICS_REVERSAL:
        for postproc in ("none", "lcc"):
            wide = build_wide(df, postproc, metric)
            for model_a, model_b in PAIRS:
                res = interaction_test(wide, model_a, model_b, "GT_star",
                                       "GT_minus_omission", args.n_boot, args.seed)
                rows.append({"metric": metric, "postprocessing": postproc,
                            "model_a": model_a, "model_b": model_b, **res})

    result = pd.DataFrame(rows)
    result["p_fdr"] = bh_fdr(result["p_raw"].values)
    result["sign_flip"] = (np.sign(result["delta_s1"]) != 0) & \
        (np.sign(result["delta_s2"]) != 0) & \
        (np.sign(result["delta_s1"]) != np.sign(result["delta_s2"]))
    result["is_reversal"] = result["sign_flip"] & (result["p_fdr"] < 0.05)

    out_path = os.path.join(args.results_dir, "lcc_reversal_check.csv")
    result.to_csv(out_path, index=False)

    print("\n=== HD95 / NSD@2 reversal, none vs. lcc ===")
    for _, r in result.iterrows():
        print(f"  {r['metric']:<6} {r['postprocessing']:<5} {r['model_a']} vs {r['model_b']}: "
              f"delta_star={r['delta_s1']:+.4f} delta_omission={r['delta_s2']:+.4f} "
              f"d={r['d']:+.3f} p_fdr={r['p_fdr']:.3g} "
              f"reversal={'YES' if r['is_reversal'] else 'no'}")

    print("\n=== clDice cost of LCC (per model, GT* and omission) ===")
    cl_rows = []
    for postproc in ("none", "lcc"):
        wide_cl = build_wide(df, postproc, "cldice")
        for model in ["M0_Star", "M1_Omission", "M2_Drift_mu0"]:
            for scenario in ["GT_star", "GT_minus_omission"]:
                if (model, scenario) not in wide_cl.columns:
                    continue
                v = wide_cl[(model, scenario)].dropna()
                cl_rows.append({"postprocessing": postproc, "model": model,
                                "scenario": scenario, "cldice_mean": v.mean(), "n": len(v)})
                print(f"  {postproc:<5} {model:<14} {scenario:<20} clDice={v.mean():.4f}")
    pd.DataFrame(cl_rows).to_csv(os.path.join(args.results_dir, "lcc_cldice_cost.csv"), index=False)

    print(f"\n[OK] {out_path}, {os.path.join(args.results_dir, 'lcc_cldice_cost.csv')}")


if __name__ == "__main__":
    main()
