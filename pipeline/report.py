#!/usr/bin/env python3
"""Progress report that can be regenerated at any time: statistics, figures, dashboard.

Reads ONLY the master table (``<results_dir>/metrics.csv``) and regenerates everything from
what exists, at any moment (also during training). Precision grows with the folds: out of
fold, each finished fold adds its ~17 cases, so n goes from ~17 to 85.

Writes:
  - ``<results_dir>/metrics_stats_<kind>.csv`` : every paired test (Wilcoxon, Cohen's d,
    FDR), from report_stats.build_tests.
  - ``<results_dir>/outcomes_<kind>.csv``      : the pre-specified comparisons Q1/Q2/Q3, each
    with its matched metric and a bootstrap interval on the difference.
  - ``<results_dir>/figures/``                 : figures (report_figures.py).
  - ``<results_dir>/STATUS.md``                : dashboard (fold progress, ledger, current
    effects, p-values and intervals), marked PRELIMINARY until all 5 folds are done.

The analyses of the paper are in analysis/; this report is the running summary of the
training campaign.

    python pipeline/report.py --results_dir results [--kind oof|test]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from prgt import results_store as rs
from prgt.results_store import pooled
from report_stats import build_tests

ALL_FOLDS = 5
TIER_A = ["M0_Star", "M1_Omission", "M2_Drift_mu0"]


def folds_done(df: pd.DataFrame) -> dict:
    """Number of finished OOF folds per model."""
    oof = df[df["eval_kind"] == "oof"]
    if oof.empty:
        return {}
    return oof.groupby("model")["fold"].nunique().astype(int).to_dict()


# ---------------------------------------------------------------------------------------------
# Paired statistics with bootstrap intervals
# ---------------------------------------------------------------------------------------------

def _bootstrap_ci(deltas: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                  seed: int = 42) -> tuple[float, float]:
    """Percentile (1 - alpha) interval of the mean of the paired differences."""
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def _paired(a: pd.Series, b: pd.Series, metric: str) -> dict | None:
    """Paired comparison a vs b (series indexed by case): difference, d, p, interval."""
    from scipy.stats import wilcoxon

    common = a.index.intersection(b.index)
    av, bv = a.loc[common].values.astype(float), b.loc[common].values.astype(float)
    if metric == "hd95":
        m = np.isfinite(av) & np.isfinite(bv)
        av, bv = av[m], bv[m]
    if len(av) < 5:
        return None
    diff = bv - av
    try:
        _, p = wilcoxon(av, bv)
    except ValueError:
        p = np.nan
    sd = np.std(diff, ddof=1)
    d = float(np.mean(diff) / sd) if sd > 0 else np.nan
    lo, hi = _bootstrap_ci(diff)
    return {
        "n": len(av), "mean_a": round(float(np.mean(av)), 4),
        "mean_b": round(float(np.mean(bv)), 4), "delta": round(float(np.mean(diff)), 4),
        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
        "cohens_d": round(d, 3) if not np.isnan(d) else None,
        "p_value": round(float(p), 4) if not np.isnan(p) else None,
    }


def _series(p: pd.DataFrame, model: str, scenario: str, metric: str) -> pd.Series:
    return (p[(p["model"] == model) & (p["scenario"] == scenario)]
            .set_index("case")[metric])


# Metric matched to each kind of noise (see docs/experiment_plan.md):
#   omission (topological) -> clDice, Betti-0; drift (surface) -> NSD@0.5, volume_delta.
_OMISSION_METRICS = ("cldice", "betti0")
_DRIFT_METRICS = ("nsd05", "volume_delta")


def outcome_tests(p: pd.DataFrame, metric: str | None = None) -> pd.DataFrame:
    """Pre-specified tests Q1/Q2/Q3 (M0-M4), each with its MATCHED metrics.

    (``metric`` is ignored: each question fixes its own metrics.)
    """
    rows = []
    models = set(p["model"].unique())
    scen = set(p["scenario"].unique())
    has = lambda *m: set(m).issubset(models)
    S = lambda model, sc, met: _series(p, model, sc, met)

    def add(question, label, a, b, met):
        res = _paired(a, b, met)
        if res:
            rows.append({"question": question, "outcome": label, "metric": met, **res})

    # Q1: does GT- penalise M0 (trained on clean labels)? GT_star vs GT-
    if has("M0_Star"):
        if {"GT_star", "GT_minus_omission"}.issubset(scen):
            for met in _OMISSION_METRICS:
                add("Q1", "Q1_omission_penalty",
                    S("M0_Star", "GT_star", met), S("M0_Star", "GT_minus_omission", met), met)
        for sc in ("GT_minus_drift_neg", "GT_minus_drift_pos"):
            if {"GT_star", sc}.issubset(scen):
                for met in _DRIFT_METRICS:
                    add("Q1", f"Q1_{sc}_penalty",
                        S("M0_Star", "GT_star", met), S("M0_Star", sc, met), met)

    # Q2: robustness / augmentation (mu = 0): M2 vs M0, scored on the clean GT*
    if has("M0_Star", "M2_Drift_mu0") and "GT_star" in scen:
        for met in ("nsd05", "cldice"):
            add("Q2", "Q2_robustness",
                S("M0_Star", "GT_star", met), S("M2_Drift_mu0", "GT_star", met), met)

    # Q3: learned bias: on the biased GT-, Mk fits better than M0, and deviates on GT*
    for model, sc in [("M3_Drift_muMinus", "GT_minus_drift_neg"),
                      ("M4_Drift_muPlus", "GT_minus_drift_pos")]:
        if has("M0_Star", model) and sc in scen:
            for met in ("nsd05", "cldice"):
                add("Q3", f"Q3_{model}_fits_bias",
                    S("M0_Star", sc, met), S(model, sc, met), met)
            if "GT_star" in scen:  # volume deviation on the clean reference (sign of the learned bias)
                add("Q3", f"Q3_{model}_deviation",
                    S("M0_Star", "GT_star", "volume_delta"),
                    S(model, "GT_star", "volume_delta"), "volume_delta")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------

def make_figures(p: pd.DataFrame, figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    try:
        import report_figures as figs
    except Exception as e:
        print(f"  [figures] cannot import report_figures: {e}")
        return
    for fn in (figs.plot_metric_boxplots, figs.plot_question_summary):
        try:
            fn(p, figures_dir)
        except Exception as e:
            print(f"  [figures] {fn.__name__} failed: {e}")


# ---------------------------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------------------------

def _ledger_summary(results_dir: str) -> str:
    path = os.path.join(results_dir, "ledger.json")
    if not os.path.exists(path):
        return "_No ledger (the orchestrator has not run yet)._"
    try:
        led = json.load(open(path))
    except Exception:
        return "_Unreadable ledger._"
    units = led.get("units", led) if isinstance(led, dict) else {}
    from collections import Counter
    states = Counter(u.get("state", "?") for u in units.values())
    total = sum(states.values()) or 1
    done = states.get("done", 0)
    lines = [f"- **{total}** units: "
             + ", ".join(f"{k}: {v}" for k, v in sorted(states.items()))
             + f"  -> {100 * done / total:.0f}% done"]
    return "\n".join(lines)


def write_status(results_dir: str, df: pd.DataFrame, outcomes: pd.DataFrame,
                 preliminary: bool, fdone: dict) -> None:
    lines = ["# STATUS: PARSE reference-noise study\n"]
    if preliminary:
        lines.append("> **PRELIMINARY RESULTS**: not all folds are finished; the estimates "
                     "will change.\n")
    else:
        lines.append("> **Complete results** (5 folds).\n")

    lines.append("## Fold progress (out of fold)\n")
    if fdone:
        for m in TIER_A:
            if m in fdone:
                bar = "#" * fdone[m] + "." * (ALL_FOLDS - fdone[m])
                lines.append(f"- `{m:<20}` {bar} {fdone[m]}/{ALL_FOLDS} folds")
        for m, n in fdone.items():
            if m not in TIER_A:
                lines.append(f"- `{m:<20}` {n}/{ALL_FOLDS} folds")
    else:
        lines.append("_No OOF prediction collected yet._")

    oof = df[df["eval_kind"] == "oof"]
    n_cases = oof.groupby("model")["case"].nunique().to_dict() if not oof.empty else {}
    lines.append("\n## Coverage (n cases scored out of fold)\n")
    for m, n in sorted(n_cases.items()):
        lines.append(f"- `{m:<20}` n = {n} / 85")

    lines.append("\n## Orchestrator (ledger)\n")
    lines.append(_ledger_summary(results_dir))

    lines.append("\n## Questions Q1/Q2/Q3 (OOF): paired Wilcoxon + 95% bootstrap CI, matched metric\n")
    if outcomes is not None and not outcomes.empty:
        lines.append("| Q | Outcome | Metric | n | delta (b-a) | 95% CI | Cohen's d | p |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in outcomes.iterrows():
            ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]" if pd.notna(r["ci_lo"]) else "-"
            d = f"{r['cohens_d']:+.2f}" if pd.notna(r["cohens_d"]) else "-"
            pv = f"{r['p_value']:.4f}" if pd.notna(r["p_value"]) else "-"
            lines.append(f"| {r.get('question','-')} | {r['outcome']} | {r['metric']} | "
                         f"{int(r['n'])} | {r['delta']:+.3f} | {ci} | {d} | {pv} |")
    else:
        lines.append("_Not enough paired data for the questions (n < 5)._")

    lines.append(f"\n_Generated from `{rs.CSV_NAME}`, "
                 f"commit `{rs.git_commit()}`, {rs.now_iso()}._")

    with open(os.path.join(results_dir, "STATUS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Progress report (statistics, figures, STATUS.md)")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--kind", default="oof", choices=["oof", "test"],
                        help="primary eval_kind for the figures and STATUS.md (default: oof)")
    args = parser.parse_args()

    df = rs.load(args.results_dir)
    if df.empty:
        print(f"[SKIP] {os.path.join(args.results_dir, rs.CSV_NAME)} empty or missing.")
        return

    fdone = folds_done(df)
    present_A = [m for m in TIER_A if m in fdone]
    preliminary = (not present_A) or any(fdone.get(m, 0) < ALL_FOLDS for m in present_A)

    # statistics for both eval kinds, when available
    for kind in ("oof", "test"):
        p = pooled(df, kind)
        if p.empty:
            continue
        full = build_tests(p)
        if not full.empty:
            full.to_csv(os.path.join(args.results_dir, f"metrics_stats_{kind}.csv"), index=False)
        outs = outcome_tests(p)
        if not outs.empty:
            outs.to_csv(os.path.join(args.results_dir, f"outcomes_{kind}.csv"), index=False)
        print(f"[{kind}] {p['case'].nunique()} cases, "
              f"{len(p['model'].unique())} models -> "
              f"metrics_stats_{kind}.csv / outcomes_{kind}.csv")

    # figures and STATUS.md for the primary kind
    p_primary = pooled(df, args.kind)
    if not p_primary.empty:
        make_figures(p_primary, os.path.join(args.results_dir, "figures"))
    outs_primary = outcome_tests(p_primary) if not p_primary.empty else pd.DataFrame()
    write_status(args.results_dir, df, outs_primary, preliminary, fdone)

    banner = "PRELIMINARY" if preliminary else "COMPLETE"
    print(f"\n[{banner}] STATUS.md and figures written to {args.results_dir}/")


if __name__ == "__main__":
    main()
