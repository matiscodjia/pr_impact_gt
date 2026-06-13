#!/usr/bin/env python3
"""Rapport dynamique *anytime* — stats, figures, tableau de bord.

Consomme **uniquement** la table maîtresse (``results/metrics.csv``) et régénère
tout à partir de ce qui existe, à n'importe quel moment (même en cours
d'entraînement). La précision croît avec les folds : en out-of-fold, chaque fold
terminé ajoute ses ~17 cas → ``n`` monte de ~17 à 85.

Produit :
  - ``results/metrics_stats_<kind>.csv``  : tous les tests appariés (Wilcoxon +
    Cohen's d + FDR), réutilise full_statistical_analysis.build_tests.
  - ``results/outcomes_<kind>.csv``       : les comparaisons pré-spécifiées
    (outcomes 1,3,4,6) avec **IC bootstrap** sur le delta.
  - ``results/figures/``                  : figures par outcome (réutilise les
    plot_* d'aggregate_results).
  - ``results/STATUS.md``                 : tableau de bord (maturité, ledger,
    effets/p/IC courants), filigrane PRELIMINARY tant que les 5 folds ne sont
    pas tous terminés.

Usage
-----
    python scripts/report.py --results_dir results [--kind oof|test]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import results_store as rs
from full_statistical_analysis import build_tests

ALL_FOLDS = 5
TIER_A = ["M0_Star", "M1_Omission", "M2_Drift_mu0"]


# ─────────────────────────────────────────────────────────────────
# Préparation des données
# ─────────────────────────────────────────────────────────────────

def pooled(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Vue (model, scenario, case, métriques) pour un eval_kind.

    En OOF, chaque cas est prédit par un seul fold → une ligne par
    (model, scenario, case). On déduplique par sécurité (keep last).
    """
    sub = df[df["eval_kind"] == kind].copy()
    if sub.empty:
        return sub
    sub = sub.drop_duplicates(subset=["model", "scenario", "case"], keep="last")
    return sub[["model", "scenario", "case"] + rs.METRICS]


def folds_done(df: pd.DataFrame) -> dict:
    """Nombre de folds OOF terminés par modèle."""
    oof = df[df["eval_kind"] == "oof"]
    if oof.empty:
        return {}
    return oof.groupby("model")["fold"].nunique().astype(int).to_dict()


# ─────────────────────────────────────────────────────────────────
# Statistiques appariées + IC bootstrap
# ─────────────────────────────────────────────────────────────────

def _bootstrap_ci(deltas: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                  seed: int = 42) -> tuple[float, float]:
    """IC percentile (1-alpha) de la moyenne des deltas appariés."""
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def _paired(a: pd.Series, b: pd.Series, metric: str) -> dict | None:
    """Comparaison appariée a vs b (séries indexées par cas) : delta, d, p, IC."""
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


# Métrique appariée par scénario de bruit (cf. experiments_plan.md) :
#   omission (topologique) → clDice, betti0 ; drift (surface) → nsd05, volume_delta.
_OMISSION_METRICS = ("cldice", "betti0")
_DRIFT_METRICS = ("nsd05", "volume_delta")


def outcome_tests(p: pd.DataFrame, metric: str | None = None) -> pd.DataFrame:
    """Tests pré-spécifiés Q1/Q2/Q3 (M0–M4), chacun avec sa métrique APPARIÉE.

    (Le paramètre ``metric`` est ignoré : chaque question fixe ses propres métriques.)
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

    # ── Q1 — la GT⁻ pénalise-t-elle M0 (entraîné propre) ? GT_star vs GT⁻ ──
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

    # ── Q2 — robustesse/augmentation (μ=0) : M2 vs M0, évalués sur GT* propre ──
    if has("M0_Star", "M2_Drift_mu0") and "GT_star" in scen:
        for met in ("nsd05", "cldice"):
            add("Q2", "Q2_robustness",
                S("M0_Star", "GT_star", met), S("M2_Drift_mu0", "GT_star", met), met)

    # ── Q3 — biais appris : sur la GT⁻ biaisée, Mk colle mieux que M0 ; et dévie sur GT* ──
    for model, sc in [("M3_Drift_muMinus", "GT_minus_drift_neg"),
                      ("M4_Drift_muPlus", "GT_minus_drift_pos")]:
        if has("M0_Star", model) and sc in scen:
            for met in ("nsd05", "cldice"):
                add("Q3", f"Q3_{model}_fits_bias",
                    S("M0_Star", sc, met), S(model, sc, met), met)
            if "GT_star" in scen:  # déviation de volume sur le vrai (signe du biais appris)
                add("Q3", f"Q3_{model}_deviation",
                    S("M0_Star", "GT_star", "volume_delta"),
                    S(model, "GT_star", "volume_delta"), "volume_delta")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# Figures (réutilise aggregate_results)
# ─────────────────────────────────────────────────────────────────

def make_figures(p: pd.DataFrame, figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    try:
        import aggregate_results as agg
    except Exception as e:
        print(f"  [figures] import aggregate_results impossible: {e}")
        return
    for fn in (agg.plot_metric_boxplots, agg.plot_question_summary):
        try:
            fn(p, figures_dir)
        except Exception as e:
            print(f"  [figures] {fn.__name__} a échoué: {e}")


# ─────────────────────────────────────────────────────────────────
# Tableau de bord
# ─────────────────────────────────────────────────────────────────

def _ledger_summary(results_dir: str) -> str:
    path = os.path.join(results_dir, "ledger.json")
    if not os.path.exists(path):
        return "_Ledger absent (orchestrateur pas encore lancé)._"
    try:
        led = json.load(open(path))
    except Exception:
        return "_Ledger illisible._"
    units = led.get("units", led) if isinstance(led, dict) else {}
    from collections import Counter
    states = Counter(u.get("state", "?") for u in units.values())
    total = sum(states.values()) or 1
    done = states.get("done", 0)
    lines = [f"- **{total}** unités — "
             + ", ".join(f"{k}: {v}" for k, v in sorted(states.items()))
             + f"  → {100 * done / total:.0f}% terminé"]
    return "\n".join(lines)


def write_status(results_dir: str, df: pd.DataFrame, outcomes: pd.DataFrame,
                 preliminary: bool, fdone: dict) -> None:
    lines = ["# STATUS — Étude de robustesse PARSE\n"]
    if preliminary:
        lines.append("> ⚠️ **RÉSULTATS PRÉLIMINAIRES** — tous les folds ne sont pas "
                     "terminés ; les estimations s'affineront.\n")
    else:
        lines.append("> ✅ **Résultats complets** (5 folds).\n")

    lines.append("## Avancement des folds (out-of-fold)\n")
    if fdone:
        for m in TIER_A:
            if m in fdone:
                bar = "█" * fdone[m] + "░" * (ALL_FOLDS - fdone[m])
                lines.append(f"- `{m:<20}` {bar} {fdone[m]}/{ALL_FOLDS} folds")
        for m, n in fdone.items():
            if m not in TIER_A:
                lines.append(f"- `{m:<20}` {n}/{ALL_FOLDS} folds")
    else:
        lines.append("_Aucune prédiction OOF collectée pour l'instant._")

    oof = df[df["eval_kind"] == "oof"]
    n_cases = oof.groupby("model")["case"].nunique().to_dict() if not oof.empty else {}
    lines.append("\n## Couverture (n cas évalués en OOF)\n")
    for m, n in sorted(n_cases.items()):
        lines.append(f"- `{m:<20}` n = {n} / 85")

    lines.append("\n## Orchestrateur (ledger)\n")
    lines.append(_ledger_summary(results_dir))

    lines.append("\n## Questions Q1/Q2/Q3 (OOF) — Wilcoxon apparié + IC bootstrap 95% — métrique appariée\n")
    if outcomes is not None and not outcomes.empty:
        lines.append("| Q | Outcome | Métrique | n | Δ (b−a) | IC 95% | Cohen's d | p |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in outcomes.iterrows():
            ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]" if pd.notna(r["ci_lo"]) else "—"
            d = f"{r['cohens_d']:+.2f}" if pd.notna(r["cohens_d"]) else "—"
            pv = f"{r['p_value']:.4f}" if pd.notna(r["p_value"]) else "—"
            lines.append(f"| {r.get('question','—')} | {r['outcome']} | {r['metric']} | "
                         f"{int(r['n'])} | {r['delta']:+.3f} | {ci} | {d} | {pv} |")
    else:
        lines.append("_Pas assez de données appariées pour les questions (n<5)._")

    lines.append(f"\n_Généré à partir de `{rs.CSV_NAME}` — "
                 f"commit `{rs.git_commit()}`, {rs.now_iso()}._")

    with open(os.path.join(results_dir, "STATUS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rapport dynamique anytime")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--kind", default="oof", choices=["oof", "test"],
                        help="eval_kind primaire pour figures/STATUS (défaut oof)")
    args = parser.parse_args()

    df = rs.load(args.results_dir)
    if df.empty:
        print(f"[SKIP] {os.path.join(args.results_dir, rs.CSV_NAME)} vide ou absent.")
        return

    fdone = folds_done(df)
    present_A = [m for m in TIER_A if m in fdone]
    preliminary = (not present_A) or any(fdone.get(m, 0) < ALL_FOLDS for m in present_A)

    # ── Stats sur les deux eval_kinds disponibles ──
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
        print(f"[{kind}] {p['case'].nunique()} cas, "
              f"{len(p['model'].unique())} modèles → "
              f"metrics_stats_{kind}.csv / outcomes_{kind}.csv")

    # ── Figures + STATUS sur le kind primaire ──
    p_primary = pooled(df, args.kind)
    if not p_primary.empty:
        make_figures(p_primary, os.path.join(args.results_dir, "figures"))
    outs_primary = outcome_tests(p_primary) if not p_primary.empty else pd.DataFrame()
    write_status(args.results_dir, df, outs_primary, preliminary, fdone)

    banner = "PRÉLIMINAIRE" if preliminary else "COMPLET"
    print(f"\n[{banner}] STATUS.md + figures générés dans {args.results_dir}/")


if __name__ == "__main__":
    main()
