#!/usr/bin/env python3
"""All paired tests between models and between references, for the progress report.

Two kinds of paired comparisons (Wilcoxon signed-rank) on every metric:
  - inter_model    : same reference, different models
  - inter_scenario : same model, different references (evaluation penalty)
with one Benjamini-Hochberg FDR correction over all tests.

Used by report.py (``build_tests``), which writes ``<results_dir>/metrics_stats_<kind>.csv``.
The paper's tests are in analysis/rank_reversal.py.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = {
    "cldice": "clDice (higher is better)",
    "hd95":   "HD95 mm (lower is better)",
    "nsd":    "NSD@2mm (higher is better)",
    "nsd05":  "NSD@0.5mm (higher is better)",
    "betti0": "Betti-0 error (lower is better)",
    "volume_delta": "signed volume delta",
}


def _paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = b - a
    try:
        _, p = wilcoxon(a, b)
    except ValueError:
        p = np.nan
    std_diff = np.std(diff, ddof=1)
    d = float(np.mean(diff) / std_diff) if std_diff > 0 else np.nan
    return p, d


def _filter_finite(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def build_tests(df: pd.DataFrame) -> pd.DataFrame:
    models = sorted(df["model"].unique())
    scenarios = sorted(df["scenario"].unique())
    rows = []

    # between models: same reference, different models
    for scenario in scenarios:
        sc_df = df[df["scenario"] == scenario]
        for i, model_a in enumerate(models):
            for model_b in models[i + 1:]:
                for metric in METRICS:
                    a_s = sc_df[sc_df["model"] == model_a].set_index("case")[metric]
                    b_s = sc_df[sc_df["model"] == model_b].set_index("case")[metric]
                    common = a_s.index.intersection(b_s.index)
                    if len(common) < 5:
                        continue
                    a, b = a_s.loc[common].values, b_s.loc[common].values
                    if metric == "hd95":
                        a, b = _filter_finite(a, b)
                    if len(a) < 5:
                        continue
                    p, d = _paired_wilcoxon(a, b)
                    rows.append({
                        "type": "inter_model",
                        "context": scenario,
                        "group_a": model_a,
                        "group_b": model_b,
                        "metric": metric,
                        "n": len(a),
                        "mean_a": round(float(np.mean(a)), 4),
                        "mean_b": round(float(np.mean(b)), 4),
                        "delta": round(float(np.mean(b - a)), 4),
                        "cohens_d": round(d, 4) if d is not np.nan and not np.isnan(d) else None,
                        "p_value": round(p, 4) if not np.isnan(p) else None,
                    })

    # between references: same model, different references
    for model in models:
        m_df = df[df["model"] == model]
        for i, sc_a in enumerate(scenarios):
            for sc_b in scenarios[i + 1:]:
                for metric in METRICS:
                    a_s = m_df[m_df["scenario"] == sc_a].set_index("case")[metric]
                    b_s = m_df[m_df["scenario"] == sc_b].set_index("case")[metric]
                    common = a_s.index.intersection(b_s.index)
                    if len(common) < 5:
                        continue
                    a, b = a_s.loc[common].values, b_s.loc[common].values
                    if metric == "hd95":
                        a, b = _filter_finite(a, b)
                    if len(a) < 5:
                        continue
                    p, d = _paired_wilcoxon(a, b)
                    rows.append({
                        "type": "inter_scenario",
                        "context": model,
                        "group_a": sc_a,
                        "group_b": sc_b,
                        "metric": metric,
                        "n": len(a),
                        "mean_a": round(float(np.mean(a)), 4),
                        "mean_b": round(float(np.mean(b)), 4),
                        "delta": round(float(np.mean(b - a)), 4),
                        "cohens_d": round(d, 4) if d is not np.nan and not np.isnan(d) else None,
                        "p_value": round(p, 4) if not np.isnan(p) else None,
                    })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # one Benjamini-Hochberg FDR over all tests
    p_vals = result["p_value"].fillna(1.0).values
    n = len(p_vals)
    order = np.argsort(p_vals)
    p_fdr = p_vals.copy()
    p_fdr[order] = p_vals[order] * n / (np.arange(n) + 1)
    for k in range(n - 2, -1, -1):
        p_fdr[order[k]] = min(p_fdr[order[k]], p_fdr[order[k + 1]])
    result["p_fdr"] = np.clip(p_fdr, 0, 1).round(4)
    result["significant"] = result["p_fdr"] < 0.05

    return result
