#!/usr/bin/env python3
"""Targeted robustness checks on the eight reversals of rank_reversal.py.

  (0)  Is the interaction on the SIGNED volume_delta (not |Delta V|) significant? If not
       (or only marginally), the four ``volume_delta_abs`` rows of rank_reversal_tests.csv
       are an artefact of the absolute value -- the same pathology as Betti-0: an almost
       identical additive shift for every model, where the absolute value creates a zero
       crossing that does not exist in the signed signal.
  (0b) Minimum detectable difference (80 % power, normal approximation:
       MDD = (z_.975 + z_.80) * sd(interaction) / sqrt(n)) for the two volume_delta_abs rows
       under drift_neg: is Delta_scenario (0.0002, 0.0003) below the resolution?
  (extra) For the two NSD@2 rows (M0 vs M1, M0 vs M2, under omission): how many cases
       share the sign of the interaction? Two identical p_FDR (2.54e-14) suggest a Wilcoxon
       floor (unanimous sign -> saturated statistic).

rank_reversal.py is left unchanged (it keeps volume_delta_abs for the ranking and the
rank matrix, where a directional magnitude has no meaning); this is a separate check on
the same data, with the same functions (prgt.stats) and the same seed.

Output : results/reversal_robustness_checks.csv

    python analysis/reversal_robustness.py --results_dir results --seed 42
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

from prgt import results_store as rs
from prgt.stats import bca_median_diff, bh_fdr, paired_cohens_d, wilcoxon_p_safe
from rank_reversal import MODELS, build_wide, load_pooled

TARGET_PAIRS_OMISSION = [("M0_Star", "M1_Omission"), ("M0_Star", "M2_Drift_mu0")]
TARGET_PAIRS_DRIFTNEG = [("M0_Star", "M1_Omission"), ("M0_Star", "M2_Drift_mu0")]


def _interaction_case_array(p, metric, model_a, model_b, s1, s2):
    wide = build_wide(p, metric)
    a1, b1 = wide[(model_a, s1)], wide[(model_b, s1)]
    a2, b2 = wide[(model_a, s2)], wide[(model_b, s2)]
    common = a1.dropna().index.intersection(b1.dropna().index) \
        .intersection(a2.dropna().index).intersection(b2.dropna().index)
    d1 = (a1.loc[common] - b1.loc[common]).values
    d2 = (a2.loc[common] - b2.loc[common]).values
    return d1, d2, common


def check_signed_volume_delta(p, n_boot, seed):
    rows = []
    print("\n=== (0) Interaction on the SIGNED volume_delta (vs |Delta V|) ===")
    for s2, pairs in [("GT_minus_omission", TARGET_PAIRS_OMISSION),
                      ("GT_minus_drift_neg", TARGET_PAIRS_DRIFTNEG)]:
        for model_a, model_b in pairs:
            d1, d2, common = _interaction_case_array(
                p, "volume_delta", model_a, model_b, "GT_star", s2)
            interaction = d2 - d1
            p_raw = wilcoxon_p_safe(d2, d1)
            d_eff = paired_cohens_d(interaction)
            med, ci_lo, ci_hi = bca_median_diff(interaction, n_boot=n_boot, seed=seed)
            delta_s1, delta_s2 = float(np.mean(d1)), float(np.mean(d2))
            rows.append({
                "check": "signed_volume_delta_interaction",
                "model_a": model_a, "model_b": model_b, "scenario_2": s2,
                "delta_s1_signed": delta_s1, "delta_s2_signed": delta_s2,
                "interaction_median": med, "d": d_eff,
                "ci_low": ci_lo, "ci_high": ci_hi, "p_raw": p_raw, "n": len(common),
            })
            print(f"  {model_a} vs {model_b}, star->{s2}: "
                  f"delta_star={delta_s1:+.4f} delta_scenario={delta_s2:+.4f} "
                  f"d={d_eff:+.3f} p_raw={p_raw:.4g}")
    return pd.DataFrame(rows)


def check_mdd_driftneg(p, seed, power=0.80, alpha=0.05):
    print("\n=== (0b) MDD (80% power) -- volume_delta_abs, drift_neg ===")
    rows = []
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    for model_a, model_b in TARGET_PAIRS_DRIFTNEG:
        d1, d2, common = _interaction_case_array(
            p, "volume_delta_abs", model_a, model_b, "GT_star", "GT_minus_drift_neg")
        interaction = d2 - d1
        sd = np.std(interaction, ddof=1)
        n = len(interaction)
        mdd = (z_alpha + z_power) * sd / np.sqrt(n)
        observed = float(np.mean(d2))
        below_mdd = abs(observed) < mdd
        rows.append({
            "check": "mdd_drift_neg", "model_a": model_a, "model_b": model_b,
            "observed_delta_s2": observed, "mdd_80pct": mdd,
            "below_resolution": below_mdd, "n": n,
        })
        print(f"  {model_a} vs {model_b}: observed={observed:+.5f}  MDD80%={mdd:.5f}  "
              f"{'BELOW THE RESOLUTION -> undecided' if below_mdd else 'above the resolution'}")
    return pd.DataFrame(rows)


def check_sign_floor_nsd(p):
    print("\n=== (extra) Wilcoxon floor on NSD@2 (omission): unanimous sign? ===")
    rows = []
    for model_a, model_b in TARGET_PAIRS_OMISSION:
        d1, d2, common = _interaction_case_array(
            p, "nsd", model_a, model_b, "GT_star", "GT_minus_omission")
        interaction = d2 - d1
        n_pos = int((interaction > 0).sum())
        n_neg = int((interaction < 0).sum())
        n_zero = int((interaction == 0).sum())
        n = len(interaction)
        unanimous = (n_pos == n) or (n_neg == n)
        rows.append({
            "check": "sign_floor_nsd", "model_a": model_a, "model_b": model_b,
            "n_positive": n_pos, "n_negative": n_neg, "n_zero": n_zero, "n": n,
            "unanimous_sign": unanimous,
        })
        print(f"  {model_a} vs {model_b}: {n_pos}/{n} positive, {n_neg}/{n} negative, "
              f"{n_zero}/{n} zero -> {'UNANIMOUS sign (Wilcoxon floor)' if unanimous else 'not unanimous'}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Robustness checks on the reported reversals")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    p = load_pooled(args.results_dir)

    df_signed = check_signed_volume_delta(p, args.n_boot, args.seed)
    df_signed["p_fdr"] = bh_fdr(df_signed["p_raw"].values)  # own family (4 tests)

    df_mdd = check_mdd_driftneg(p, args.seed)
    df_sign = check_sign_floor_nsd(p)

    for df in (df_signed, df_mdd, df_sign):
        df["git_commit"] = rs.git_commit()
        df["seed"] = args.seed
        df["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out_path = os.path.join(args.results_dir, "reversal_robustness_checks.csv")
    combined = pd.concat([df_signed, df_mdd, df_sign], ignore_index=True, sort=False)
    combined.to_csv(out_path, index=False)
    print(f"\n[OK] {out_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
