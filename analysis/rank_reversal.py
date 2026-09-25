#!/usr/bin/env python3
"""P0.1 -- Test de retournement de classement (LE résultat).

Lit ``results/metrics.csv`` (tier A : M0_Star, M1_Omission, M2_Drift_mu0 x GT_star,
GT_minus_omission, GT_minus_drift_neg, GT_minus_drift_pos, n=85 cas, OOF) et produit :

  (a) results/inter_model_by_reference.csv -- contrastes appariés par paire de modèles,
      pour chaque scénario et chaque métrique.
  (b) results/rank_reversal_tests.csv -- test d'interaction (star vs. chaque scénario
      dégradé) : LE test propre du retournement de classement.
  (c) results/rank_matrix.csv -- matrice de rangs (1=meilleur) par (métrique, scénario).
  (d) results/rank_stability_bootstrap.csv -- stabilité bootstrap du classement.

Conventions
-----------
- ``volume_delta`` est signé (biais directionnel, sans "meilleur" naturel) ; pour le
  classement/l'interaction on utilise sa magnitude ``volume_delta_abs = |volume_delta|``
  (biais plus petit = meilleur), toujours dans la même liste de métriques que les 5
  autres. Le signe brut reste disponible dans metrics.csv pour toute analyse dirigée
  (cf. Q1 dans report.py).
- Point estimé apparié = MÉDIANE des différences par cas (colonnes ``delta_median`` /
  ``interaction_median``), cohérent avec le test de signe Wilcoxon. Cohen's d suit la
  convention standard (moyenne/écart-type des différences), donc peut différer
  légèrement en magnitude de la médiane -- c'est attendu, ce sont deux résumés
  différents de la même distribution de différences.
- Dans (b), ``delta_s1``/``delta_s2`` sont les MOYENNES appariées (A-B) à chaque
  scénario -- volontairement différentes de ``interaction_median`` : ce sont ces
  moyennes qui correspondent aux valeurs de Table 5.5 du rapport (moyennes par cas) et
  donc à la définition même du "retournement" telle que posée dans le brief (l'ordre
  des moyennes M0/M1/M2 s'inverse). On a mesuré empiriquement que la MÉDIANE appariée
  peut être exactement 0 (nombreuses quasi-égalités cas par cas) alors que la MOYENNE
  bascule nettement -- HD95(M0,M1) sous star : médiane des diffs = 0.0 mais moyenne =
  -0.12 -- donc ``is_reversal`` doit se baser sur les moyennes pour capturer le
  retournement réel, pas sur la médiane qui masquerait le signal par ex-aequo.
- IC 95% bootstrap BCa, 10000 réechantillonnages sur les CAS (pas sur les métriques),
  seed=42.
- FDR Benjamini-Hochberg appliqué UNE SEULE FOIS sur l'union des tests (a)+(b) (colonne
  ``family`` pour tracer le sous-groupe), q=0.05.
- ``is_reversal`` (b) : critère strict de croisement -- le signe de
  ``m(A|s1)-m(B|s1)`` diffère de celui de ``m(A|s2)-m(B|s2)`` (calculés séparément,
  pas seulement déduit du signe de l'interaction), ET le test d'interaction est
  significatif après FDR. C'est plus strict que "signe de l'interaction opposé au
  signe à s1" seul (qui peut n'être qu'une atténuation sans franchissement de zéro) ;
  documenté ici pour que RESULTS_P0.md puisse distinguer les deux.

Usage
-----
    python analysis/rank_reversal.py --results_dir results --seed 42 --n_boot 10000
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import results_store as rs  # noqa: E402
from report import pooled  # noqa: E402
from stats_utils import bca_median_diff, bh_fdr, paired_cohens_d, wilcoxon_p_safe  # noqa: E402

MODELS = ["M0_Star", "M1_Omission", "M2_Drift_mu0"]
SCENARIOS = ["GT_star", "GT_minus_omission", "GT_minus_drift_neg", "GT_minus_drift_pos"]
RAW_METRICS = ["cldice", "hd95", "nsd", "nsd05", "betti0", "volume_delta"]
# Métrique utilisée pour classement/interaction : volume_delta -> magnitude du biais.
ANALYSIS_METRICS = ["cldice", "hd95", "nsd", "nsd05", "betti0", "volume_delta_abs"]
DIRECTION = {  # "higher" = plus grand est meilleur ; "lower" = plus petit est meilleur
    "cldice": "higher", "nsd": "higher", "nsd05": "higher",
    "hd95": "lower", "betti0": "lower", "volume_delta_abs": "lower",
}


def load_pooled(results_dir: str) -> pd.DataFrame:
    df = rs.load(results_dir)
    if df.empty:
        raise SystemExit(f"[FATAL] {results_dir}/metrics.csv vide ou absent -- rien à analyser.")
    p = pooled(df, "oof")
    missing_models = set(MODELS) - set(p["model"].unique())
    missing_scen = set(SCENARIOS) - set(p["scenario"].unique())
    if missing_models or missing_scen:
        print(f"[WARN] modèles manquants: {missing_models or 'aucun'}, "
              f"scénarios manquants: {missing_scen or 'aucun'}")
    p = p[p["model"].isin(MODELS) & p["scenario"].isin(SCENARIOS)].copy()
    p["volume_delta_abs"] = p["volume_delta"].abs()
    return p


def build_wide(p: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Table indexée par cas, colonnes (model,scenario) -- aligne automatiquement par cas."""
    return p.pivot_table(index="case", columns=["model", "scenario"], values=metric)


def _series(wide: pd.DataFrame, model: str, scenario: str) -> pd.Series:
    return wide[(model, scenario)].dropna()


# ─────────────────────────────────────────────────────────────────
# (a) Contrastes inter-modèles par référence
# ─────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────
# (b) Test d'interaction -- LE test du retournement
# ─────────────────────────────────────────────────────────────────

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

                # Moyennes (pas médianes) : ce sont elles qui définissent le
                # retournement au sens de Table 5.5 -- voir docstring du module.
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


# ─────────────────────────────────────────────────────────────────
# (c) Matrice de rangs
# ─────────────────────────────────────────────────────────────────

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
            row["order_signature"] = "<".join(order)  # meilleur -> pire
            rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# (d) Stabilité bootstrap du classement
# ─────────────────────────────────────────────────────────────────

def _order_of(means: dict, metric: str) -> tuple:
    ascending = DIRECTION[metric] == "lower"
    return tuple(sorted(MODELS, key=lambda m: means[m], reverse=not ascending))


def rank_stability_bootstrap(p: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for metric in ANALYSIS_METRICS:
        wide = build_wide(p, metric)
        # cas communs aux 3 modèles sous GT_star (référence pour l'ordre observé)
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

            # pour tau : ré-échantillonnage joint des cas communs à star ET scenario
            # (mêmes identités de cas -> même tirage d'indices sur le sous-ensemble commun)
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


# ─────────────────────────────────────────────────────────────────
# FDR combiné + écriture
# ─────────────────────────────────────────────────────────────────

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
    print("RETOURNEMENTS DE CLASSEMENT SIGNIFICATIFS (p_FDR < 0.05, croisement de signe strict)")
    print("=" * 100)
    if sig.empty:
        print("  Aucun.")
        return
    print(f"  {'métrique':<18} {'paire':<28} {'star -> scénario':<24} "
          f"{'Δ_star':>10} {'Δ_scénario':>10} {'p_FDR':>10} {'d':>8}")
    for _, r in sig.iterrows():
        pair = f"{r['model_a']} vs {r['model_b']}"
        transition = f"star -> {r['scenario_2']}"
        print(f"  {r['metric']:<18} {pair:<28} {transition:<24} "
              f"{r['delta_s1']:>10.4f} {r['delta_s2']:>10.4f} {r['p_fdr']:>10.4g} {r['d']:>8.3f}")


def main():
    ap = argparse.ArgumentParser(description="P0.1 -- Test de retournement de classement")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    p = load_pooled(args.results_dir)
    print(f"[DATA] {p['case'].nunique()} cas, modèles={sorted(p['model'].unique())}, "
          f"scénarios={sorted(p['scenario'].unique())}")

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

    print(f"[OK] inter_model_by_reference.csv ({len(df_a)} lignes)")
    print(f"[OK] rank_reversal_tests.csv ({len(df_b)} lignes, "
          f"{int(df_b['is_reversal'].sum())} retournement(s))")
    print(f"[OK] rank_matrix.csv ({len(df_c)} lignes)")
    print(f"[OK] rank_stability_bootstrap.csv ({len(df_d)} lignes)")

    print_reversals(df_b)


if __name__ == "__main__":
    main()
