#!/usr/bin/env python3
"""The main result: does the ranking of the models reverse when the reference is degraded?

Reads results/metrics.csv (M0_Star, M1_Omission, M2_Drift_mu0 x GT_star,
GT_minus_omission, GT_minus_drift_neg, GT_minus_drift_pos, n = 85 cases, out of fold) and
writes:

  (a) results/inter_model_by_reference.csv -- paired contrasts per pair of models, for
      every scenario and metric.
  (b) results/rank_reversal_tests.csv -- interaction test (GT* vs each degraded
      scenario): the proper test of a ranking reversal.
  (c) results/rank_matrix.csv -- rank of each model (1 = best) per (metric, scenario).
  (d) results/rank_stability_bootstrap.csv -- bootstrap stability of the ranking.

Conventions
-----------
- ``volume_delta`` is signed (a directional bias with no natural "better"); for ranking
  and interaction its magnitude ``volume_delta_abs = |volume_delta|`` is used (smaller
  bias = better), always alongside the five other metrics. The signed value stays in
  metrics.csv for directional analyses.
- The paired point estimate is the MEDIAN of the per-case differences (columns
  ``delta_median`` / ``interaction_median``), consistent with the Wilcoxon signed-rank
  test. Cohen's d follows the usual convention (mean / standard deviation of the
  differences), so its magnitude can differ slightly from the median: they are two
  summaries of the same distribution of differences.
- In (b), ``delta_s1`` / ``delta_s2`` are the paired MEANS (A - B) in each scenario, on
  purpose different from ``interaction_median``: the reversal is defined on the means
  (the order of the mean scores of M0/M1/M2 flips). The paired MEDIAN can be exactly 0
  (many near-ties case by case) while the MEAN clearly changes sign -- e.g. HD95(M0, M1)
  under GT*: median difference 0.0 but mean -0.12 -- so ``is_reversal`` uses the means.
- 95 % BCa bootstrap intervals, 10,000 resamples of the CASES (not of the metrics),
  seed 42.
- Benjamini-Hochberg FDR applied ONCE to the union of the tests (a) + (b) (column
  ``family`` records the subgroup), q = 0.05.
- ``is_reversal`` in (b) is a strict crossing: the sign of m(A|s1) - m(B|s1) differs
  from that of m(A|s2) - m(B|s2) (each computed separately, not inferred from the sign of
  the interaction), AND the interaction test is significant after FDR. This is stricter
  than "the interaction has the opposite sign of the s1 gap", which can be a mere
  attenuation without crossing zero.

    python analysis/rank_reversal.py --results_dir results --seed 42 --n_boot 10000
"""

from __future__ import annotations

import argparse
import itertools
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from prgt import results_store as rs
from prgt.results_store import pooled
from prgt.stats import bca_median_diff, bh_fdr, paired_cohens_d, wilcoxon_p_safe

MODELS = ["M0_Star", "M1_Omission", "M2_Drift_mu0"]
SCENARIOS = ["GT_star", "GT_minus_omission", "GT_minus_drift_neg", "GT_minus_drift_pos"]
RAW_METRICS = ["cldice", "hd95", "nsd", "nsd05", "betti0", "volume_delta"]
# Metrics used for ranking and interaction: volume_delta -> magnitude of the bias.
ANALYSIS_METRICS = ["cldice", "hd95", "nsd", "nsd05", "betti0", "volume_delta_abs"]
DIRECTION = {  # "higher" = larger is better; "lower" = smaller is better
    "cldice": "higher", "nsd": "higher", "nsd05": "higher",
    "hd95": "lower", "betti0": "lower", "volume_delta_abs": "lower",
}


def load_pooled(results_dir: str) -> pd.DataFrame:
    df = rs.load(results_dir)
    if df.empty:
        raise SystemExit(f"[FATAL] {results_dir}/metrics.csv empty or missing: nothing to analyse.")
    p = pooled(df, "oof")
    missing_models = set(MODELS) - set(p["model"].unique())
    missing_scen = set(SCENARIOS) - set(p["scenario"].unique())
    if missing_models or missing_scen:
        print(f"[WARN] missing models: {missing_models or 'none'}, "
              f"missing scenarios: {missing_scen or 'none'}")
    p = p[p["model"].isin(MODELS) & p["scenario"].isin(SCENARIOS)].copy()
    p["volume_delta_abs"] = p["volume_delta"].abs()
    return p


def build_wide(p: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Table indexed by case with (model, scenario) columns, so values align by case."""
    return p.pivot_table(index="case", columns=["model", "scenario"], values=metric)


def _series(wide: pd.DataFrame, model: str, scenario: str) -> pd.Series:
    return wide[(model, scenario)].dropna()


# ---------------------------------------------------------------------------------------------
# (a) Between-model contrasts for each reference
# ---------------------------------------------------------------------------------------------

def inter_model_by_reference(p: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for metric in ANALYSIS_METRICS:
        wide = build_wide(p, metric)
        for scenario in SCENARIOS:
            for model_a, model_b in itertools.combinations(MODELS, 2):
                a = _series(wide, model_a, scenario)
                b = _series(wide, model_b, scenario)
                common = a.index.intersection(b.index)
                if len(common) < 5:
                    continue
                av, bv = a.loc[common].values, b.loc[common].values
                diff = av - bv
                p_raw = wilcoxon_p_safe(av, bv)
                d = paired_cohens_d(diff)
                med, ci_lo, ci_hi = bca_median_diff(diff, n_boot=n_boot, seed=seed)
                rows.append({
                    "family": "inter_model_by_reference",
                    "metric": metric, "scenario": scenario,
                    "model_a": model_a, "model_b": model_b,
                    "delta_median": med, "d": d,
                    "ci_low": ci_lo, "ci_high": ci_hi,
                    "p_raw": p_raw, "n": len(common),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# (b) Interaction test: the test of the ranking reversal
# ---------------------------------------------------------------------------------------------

def rank_reversal_tests(p: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    s1 = "GT_star"
    for metric in ANALYSIS_METRICS:
        wide = build_wide(p, metric)
        for model_a, model_b in itertools.combinations(MODELS, 2):
            for s2 in ["GT_minus_omission", "GT_minus_drift_neg", "GT_minus_drift_pos"]:
                a1, b1 = _series(wide, model_a, s1), _series(wide, model_b, s1)
                a2, b2 = _series(wide, model_a, s2), _series(wide, model_b, s2)
                common = a1.index.intersection(b1.index).intersection(
                    a2.index).intersection(b2.index)
                if len(common) < 5:
                    continue
                a1v, b1v = a1.loc[common].values, b1.loc[common].values
                a2v, b2v = a2.loc[common].values, b2.loc[common].values
                delta_s1_case = a1v - b1v
                delta_s2_case = a2v - b2v
                interaction = delta_s2_case - delta_s1_case

                p_raw = wilcoxon_p_safe(delta_s2_case, delta_s1_case)  # interaction vs 0
                d = paired_cohens_d(interaction)
                med, ci_lo, ci_hi = bca_median_diff(interaction, n_boot=n_boot, seed=seed)

                # Means (not medians) define the reversal; see the module docstring.
                delta_s1 = float(np.mean(delta_s1_case))
                delta_s2 = float(np.mean(delta_s2_case))
                rows.append({
                    "family": "rank_reversal_interaction",
                    "metric": metric, "model_a": model_a, "model_b": model_b,
                    "scenario_1": s1, "scenario_2": s2,
                    "delta_s1": delta_s1, "delta_s2": delta_s2,
                    "interaction_median": med, "d": d,
                    "ci_low": ci_lo, "ci_high": ci_hi,
                    "p_raw": p_raw, "n": len(common),
                    "_sign_flip": (np.sign(delta_s1) != 0 and np.sign(delta_s2) != 0
                                   and np.sign(delta_s1) != np.sign(delta_s2)),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# (c) Rank matrix
# ---------------------------------------------------------------------------------------------

def rank_matrix(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ANALYSIS_METRICS:
        wide = build_wide(p, metric)
        for scenario in SCENARIOS:
            means = {}
            for model in MODELS:
                s = _series(wide, model, scenario)
                means[model] = s.mean() if len(s) else np.nan
            ascending = DIRECTION[metric] == "lower"
            order = sorted(MODELS, key=lambda m: means[m], reverse=not ascending)
            ranks = {m: order.index(m) + 1 for m in MODELS}
            row = {"metric": metric, "scenario": scenario}
            for m in MODELS:
                row[f"rank_{m}"] = ranks[m]
            row["order_signature"] = "<".join(order)  # best -> worst
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# (d) Bootstrap stability of the ranking
# ---------------------------------------------------------------------------------------------

def _order_of(means: dict, metric: str) -> tuple:
    ascending = DIRECTION[metric] == "lower"
    return tuple(sorted(MODELS, key=lambda m: means[m], reverse=not ascending))


def rank_stability_bootstrap(p: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for metric in ANALYSIS_METRICS:
        wide = build_wide(p, metric)
        # cases common to the 3 models under GT_star (reference for the observed order)
        star_cols = {m: wide[(m, "GT_star")] for m in MODELS}
        common_star = None
        for s in star_cols.values():
            idx = s.dropna().index
            common_star = idx if common_star is None else common_star.intersection(idx)
        star_vals = {m: star_cols[m].loc[common_star].values for m in MODELS}
        n_star = len(common_star)
        observed_order_star = _order_of({m: star_vals[m].mean() for m in MODELS}, metric)

        rng = np.random.default_rng(seed)
        for scenario in SCENARIOS:
            cols = {m: wide[(m, scenario)] for m in MODELS}
            common = None
            for s in cols.values():
                idx = s.dropna().index
                common = idx if common is None else common.intersection(idx)
            vals = {m: cols[m].loc[common].values for m in MODELS}
            n = len(common)
            if n < 5:
                continue

            observed_order = _order_of({m: vals[m].mean() for m in MODELS}, metric)

            idx_boot = rng.integers(0, n, size=(n_boot, n))
            means_boot = {m: vals[m][idx_boot].mean(axis=1) for m in MODELS}
            n_differ = 0
            taus = np.empty(n_boot)

            # for tau: joint resampling of the cases common to GT* AND the scenario
            # (same case identities -> same index draw on the common subset)
            common_both = common_star.intersection(common)
            n_both = len(common_both)
            idx_tau = rng.integers(0, n_both, size=(n_boot, n_both)) if n_both >= 5 else None
            star_vals_both = {m: star_cols[m].loc[common_both].values for m in MODELS} \
                if n_both >= 5 else None
            vals_both = {m: cols[m].loc[common_both].values for m in MODELS} \
                if n_both >= 5 else None

            for b in range(n_boot):
                boot_order = tuple(sorted(
                    MODELS, key=lambda m: means_boot[m][b],
                    reverse=(DIRECTION[metric] != "lower")))
                if boot_order != observed_order_star:
                    n_differ += 1

            if n_both >= 5:
                rank_star_obs = {m: observed_order_star.index(m) + 1 for m in MODELS}
                rank_scn_obs = {m: observed_order.index(m) + 1 for m in MODELS}
                tau_point, _ = kendalltau(
                    [rank_star_obs[m] for m in MODELS], [rank_scn_obs[m] for m in MODELS])
                for b in range(n_boot):
                    m_star_b = {m: star_vals_both[m][idx_tau[b]].mean() for m in MODELS}
                    m_scn_b = {m: vals_both[m][idx_tau[b]].mean() for m in MODELS}
                    o_star_b = _order_of(m_star_b, metric)
                    o_scn_b = _order_of(m_scn_b, metric)
                    r_star_b = {m: o_star_b.index(m) + 1 for m in MODELS}
                    r_scn_b = {m: o_scn_b.index(m) + 1 for m in MODELS}
                    t, _ = kendalltau([r_star_b[m] for m in MODELS],
                                      [r_scn_b[m] for m in MODELS])
                    taus[b] = t if np.isfinite(t) else np.nan
                valid = taus[np.isfinite(taus)]
                tau_ci_lo, tau_ci_hi = (np.percentile(valid, [2.5, 97.5])
                                         if len(valid) > 10 else (np.nan, np.nan))
            else:
                tau_point, tau_ci_lo, tau_ci_hi = np.nan, np.nan, np.nan

            rows.append({
                "metric": metric, "scenario": scenario,
                "frac_order_differs_from_star": n_differ / n_boot,
                "kendall_tau": tau_point,
                "tau_ci_low": tau_ci_lo, "tau_ci_high": tau_ci_hi,
                "n": n, "n_star": n_star,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Combined FDR and output
# ---------------------------------------------------------------------------------------------

def apply_combined_fdr(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_a = len(df_a)
    all_p = pd.concat([df_a["p_raw"], df_b["p_raw"]], ignore_index=True).values
    all_q = bh_fdr(all_p)
    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a["p_fdr"] = all_q[:n_a]
    df_b["p_fdr"] = all_q[n_a:]
    df_b["is_reversal"] = df_b["_sign_flip"] & (df_b["p_fdr"] < 0.05)
    df_b = df_b.drop(columns=["_sign_flip"])
    return df_a, df_b


def _stamp(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["git_commit"] = rs.git_commit()
    df["seed"] = seed
    df["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return df


def print_reversals(df_b: pd.DataFrame) -> None:
    sig = df_b[df_b["is_reversal"]].sort_values("p_fdr")
    print("\n" + "=" * 100)
    print("SIGNIFICANT RANKING REVERSALS (p_FDR < 0.05, strict sign crossing)")
    print("=" * 100)
    if sig.empty:
        print("  None.")
        return
    print(f"  {'metric':<18} {'pair':<28} {'star -> scenario':<24} "
          f"{'d_star':>10} {'d_scenario':>10} {'p_FDR':>10} {'d':>8}")
    for _, r in sig.iterrows():
        pair = f"{r['model_a']} vs {r['model_b']}"
        transition = f"star -> {r['scenario_2']}"
        print(f"  {r['metric']:<18} {pair:<28} {transition:<24} "
              f"{r['delta_s1']:>10.4f} {r['delta_s2']:>10.4f} {r['p_fdr']:>10.4g} {r['d']:>8.3f}")


def main():
    ap = argparse.ArgumentParser(description="Ranking-reversal test")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    p = load_pooled(args.results_dir)
    print(f"[DATA] {p['case'].nunique()} cases, models={sorted(p['model'].unique())}, "
          f"scenarios={sorted(p['scenario'].unique())}")

    df_a = inter_model_by_reference(p, args.n_boot, args.seed)
    df_b = rank_reversal_tests(p, args.n_boot, args.seed)
    df_a, df_b = apply_combined_fdr(df_a, df_b)
    df_c = rank_matrix(p)
    df_d = rank_stability_bootstrap(p, args.n_boot, args.seed)

    os.makedirs(args.results_dir, exist_ok=True)
    df_a = _stamp(df_a, args.seed)
    df_b = _stamp(df_b, args.seed)
    df_c = _stamp(df_c, args.seed)
    df_d = _stamp(df_d, args.seed)

    df_a.to_csv(os.path.join(args.results_dir, "inter_model_by_reference.csv"), index=False)
    df_b.to_csv(os.path.join(args.results_dir, "rank_reversal_tests.csv"), index=False)
    df_c.to_csv(os.path.join(args.results_dir, "rank_matrix.csv"), index=False)
    df_d.to_csv(os.path.join(args.results_dir, "rank_stability_bootstrap.csv"), index=False)

    print(f"[OK] inter_model_by_reference.csv ({len(df_a)} rows)")
    print(f"[OK] rank_reversal_tests.csv ({len(df_b)} rows, "
          f"{int(df_b['is_reversal'].sum())} reversal(s))")
    print(f"[OK] rank_matrix.csv ({len(df_c)} rows)")
    print(f"[OK] rank_stability_bootstrap.csv ({len(df_d)} rows)")

    print_reversals(df_b)


if __name__ == "__main__":
    main()
