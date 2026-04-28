#!/usr/bin/env python3
"""Analyse statistique complète à partir du CSV de cross-évaluation.

Deux types de comparaisons appariées (Wilcoxon) sur toutes les métriques :
  - inter_model   : même scénario GT, modèles différents
  - inter_scenario: même modèle, scénarios GT différents (pénalité d'évaluation)

Correction FDR Benjamini-Hochberg appliquée sur l'ensemble des tests.

Usage
-----
    python scripts/full_statistical_analysis.py --csv results/cross_evaluation.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = {
    "cldice": "CLDice (↑)",
    "hd95":   "HD95 mm (↓)",
    "nsd":    "NSD (↑)",
    "betti0": "Betti0 (↓)",
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

    # ── Inter-modèles : même scénario, modèles différents ──
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

    # ── Inter-scénarios : même modèle, scénarios différents ──
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

    # ── FDR Benjamini-Hochberg global ──
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


def print_report(df_tests: pd.DataFrame) -> None:
    sep = "=" * 75

    for comp_type, label in [
        ("inter_model",    "INTER-MODÈLES  (même scénario GT, modèles différents)"),
        ("inter_scenario", "INTER-SCÉNARIOS (même modèle, scénarios GT différents — pénalité)"),
    ]:
        print(f"\n{sep}")
        print(label)
        print(sep)

        subset = df_tests[df_tests["type"] == comp_type]
        contexts = sorted(subset["context"].unique())

        for ctx in contexts:
            ctx_df = subset[subset["context"] == ctx]
            print(f"\n  [{ctx}]")
            print(f"  {'metric':<10} {'group_a':<25} {'group_b':<25} "
                  f"{'mean_a':>8} {'mean_b':>8} {'delta':>8} "
                  f"{'d':>7} {'p_fdr':>7} {'sig':>4}")
            print(f"  {'-'*10} {'-'*25} {'-'*25} "
                  f"{'-'*8} {'-'*8} {'-'*8} "
                  f"{'-'*7} {'-'*7} {'-'*4}")
            for _, row in ctx_df.iterrows():
                sig = "✓" if row["significant"] else ""
                d_str = f"{row['cohens_d']:>7.3f}" if row["cohens_d"] is not None else "    N/A"
                p_str = f"{row['p_fdr']:>7.4f}" if row["p_fdr"] is not None else "    N/A"
                print(f"  {row['metric']:<10} {row['group_a']:<25} {row['group_b']:<25} "
                      f"{row['mean_a']:>8.4f} {row['mean_b']:>8.4f} {row['delta']:>8.4f} "
                      f"{d_str} {p_str} {sig:>4}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/cross_evaluation.csv")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df_tests = build_tests(df)

    if df_tests.empty:
        print("Aucun test produit.")
        return

    print_report(df_tests)

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "full_statistical_analysis.csv")
    df_tests.to_csv(out_path, index=False)
    print(f"\n{df_tests['significant'].sum()}/{len(df_tests)} tests significatifs (p_fdr < 0.05)")
    print(f"Résultats sauvegardés : {out_path}")


if __name__ == "__main__":
    main()
