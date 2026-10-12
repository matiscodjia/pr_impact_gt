#!/usr/bin/env python3
"""Size of the reference effect relative to the model effect, with uncertainty.

The technical report compared two point means ("the reference weighs 5 to 20 times the
model") without an interval. This script redoes the comparison:

  - reference effect, per model M and scenario s: m(M|s, case) - m(M|star, case)
  - model effect, per pair (A, B) and scenario s: m(A|s, case) - m(B|s, case)
  - for one (metric, scenario, pair {A, B}), the reference effect is the case-wise MEAN
    of the reference effects of A and B (the two models compared on that row), not the
    effect of an arbitrary single model;
  - ratio = median(|reference effect|) / median(|model effect|), both medians recomputed
    on the SAME bootstrap resample of the cases (BCa, 10,000 resamples, seed 42).
    Resampling numerator and denominator independently would ignore their case-wise
    correlation and understate the uncertainty; prgt.stats.bca_ratio couples them.

Output : results/effect_ratio.csv
  metric, reference, model_pair, ratio_median, ratio_ci_low, ratio_ci_high,
  ref_effect_abs_median, model_effect_abs_median, n, git_commit, seed, timestamp

    python analysis/effect_ratio.py --results_dir results --seed 42 --n_boot 10000
"""

from __future__ import annotations

import argparse
import itertools
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from prgt import results_store as rs
from prgt.stats import bca_ratio
from rank_reversal import ANALYSIS_METRICS, MODELS, load_pooled

DEGRADED_SCENARIOS = ["GT_minus_omission", "GT_minus_drift_neg", "GT_minus_drift_pos"]


def build_wide(p: pd.DataFrame, metric: str) -> pd.DataFrame:
    return p.pivot_table(index="case", columns=["model", "scenario"], values=metric)


def effect_ratio(p: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for metric in ANALYSIS_METRICS:
        wide = build_wide(p, metric)
        for scenario in DEGRADED_SCENARIOS:
            for model_a, model_b in itertools.combinations(MODELS, 2):
                a_star = wide[(model_a, "GT_star")]
                a_s = wide[(model_a, scenario)]
                b_star = wide[(model_b, "GT_star")]
                b_s = wide[(model_b, scenario)]

                common = (a_star.dropna().index
                          .intersection(a_s.dropna().index)
                          .intersection(b_star.dropna().index)
                          .intersection(b_s.dropna().index))
                if len(common) < 5:
                    continue

                ref_effect_a = (a_s.loc[common] - a_star.loc[common]).values
                ref_effect_b = (b_s.loc[common] - b_star.loc[common]).values
                ref_effect = (ref_effect_a + ref_effect_b) / 2.0  # mean over the two models of the pair

                model_effect = (a_s.loc[common] - b_s.loc[common]).values

                ratio, ci_lo, ci_hi = bca_ratio(ref_effect, model_effect,
                                                 n_boot=n_boot, seed=seed)
                rows.append({
                    "metric": metric, "reference": scenario,
                    "model_pair": f"{model_a}_vs_{model_b}",
                    "ratio_median": ratio, "ratio_ci_low": ci_lo, "ratio_ci_high": ci_hi,
                    "ref_effect_abs_median": float(np.median(np.abs(ref_effect))),
                    "model_effect_abs_median": float(np.median(np.abs(model_effect))),
                    "n": len(common),
                })
    return pd.DataFrame(rows)


def _stamp(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["git_commit"] = rs.git_commit()
    df["seed"] = seed
    df["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return df


def main():
    ap = argparse.ArgumentParser(description="Reference effect / model effect ratio")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    p = load_pooled(args.results_dir)
    df = effect_ratio(p, args.n_boot, args.seed)
    df = _stamp(df, args.seed)

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "effect_ratio.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] {out_path} ({len(df)} rows)")

    print("\n" + "=" * 90)
    print("REFERENCE EFFECT / MODEL EFFECT RATIO (median, 95% BCa CI)")
    print("=" * 90)
    for _, r in df.sort_values("ratio_median", ascending=False).iterrows():
        print(f"  {r['metric']:<18} {r['reference']:<22} {r['model_pair']:<28} "
              f"ratio={r['ratio_median']:>8.2f}  CI=[{r['ratio_ci_low']:.2f}, "
              f"{r['ratio_ci_high']:.2f}]  (|ref|={r['ref_effect_abs_median']:.4g}, "
              f"|mod|={r['model_effect_abs_median']:.4g})")


if __name__ == "__main__":
    main()
