#!/usr/bin/env python3
"""Vérifications ciblées sur les 8 retournements de P0.1, demandées après relecture :

  (0)  L'interaction sur volume_delta SIGNÉ (pas |ΔV|) est-elle significative ?
       Si non (ou marginale), les 4 lignes ``volume_delta_abs`` de
       rank_reversal_tests.csv sont un artefact de la valeur absolue -- même
       pathologie que Betti0 (décalage additif quasi identique entre modèles,
       la valeur absolue fabrique un croisement de zéro qui n'existe pas dans le
       signal signé).
  (0b) MDD (80% puissance, approximation normale standard : MDD = (z_.975+z_.80)
       * sd(interaction)/sqrt(n)) pour les 2 lignes volume_delta_abs sous
       drift_neg : Δ_scénario (0.0002, 0.0003) est-il sous le seuil de résolution ?
  (extra) Pour les 2 lignes NSD@2 (M0 vs M1, M0 vs M2, sous omission) : combien de
       cas partagent le même signe d'interaction ? p_FDR identique à 2.54e-14 fait
       soupçonner un plancher Wilcoxon (signe unanime -> statistique saturée).

Ne réécrit PAS rank_reversal.py (qui reste sur volume_delta_abs pour le classement/
la matrice de rangs, où une magnitude directionnelle n'a pas de sens) -- ce script
est un contrôle de robustesse séparé, sur les mêmes données, mêmes fonctions
(stats_utils), même seed.

Sortie : results/reversal_robustness_checks.csv

Usage
-----
    python analysis/verify_reversal_robustness.py --results_dir results --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import results_store as rs  # noqa: E402
from rank_reversal import MODELS, build_wide, load_pooled  # noqa: E402
from stats_utils import bca_median_diff, bh_fdr, paired_cohens_d, wilcoxon_p_safe  # noqa: E402

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
    print("\n=== (0) Interaction sur volume_delta SIGNÉ (vs |ΔV|) ===")
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
                  f"Δ_star={delta_s1:+.4f} Δ_scn={delta_s2:+.4f} "
                  f"d={d_eff:+.3f} p_raw={p_raw:.4g}")
    return pd.DataFrame(rows)


def check_mdd_driftneg(p, seed, power=0.80, alpha=0.05):
    print("\n=== (0b) MDD (80% puissance) -- volume_delta_abs, drift_neg ===")
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
        print(f"  {model_a} vs {model_b}: observé={observed:+.5f}  MDD80%={mdd:.5f}  "
              f"{'SOUS LE SEUIL -> undecided' if below_mdd else 'au-dessus du seuil'}")
    return pd.DataFrame(rows)


def check_sign_floor_nsd(p):
    print("\n=== (extra) Plancher Wilcoxon sur NSD@2 (omission) : signe unanime ? ===")
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
        print(f"  {model_a} vs {model_b}: {n_pos}/{n} positifs, {n_neg}/{n} négatifs, "
              f"{n_zero}/{n} nuls -> {'signe UNANIME (plancher Wilcoxon)' if unanimous else 'pas unanime'}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Vérifications de robustesse post-relecture")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    p = load_pooled(args.results_dir)

    df_signed = check_signed_volume_delta(p, args.n_boot, args.seed)
    df_signed["p_fdr"] = bh_fdr(df_signed["p_raw"].values)  # famille propre (4 tests)

    df_mdd = check_mdd_driftneg(p, args.seed)
    df_sign = check_sign_floor_nsd(p)

    for df in (df_signed, df_mdd, df_sign):
        df["git_commit"] = rs.git_commit()
        df["seed"] = args.seed
        df["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out_path = os.path.join(args.results_dir, "reversal_robustness_checks.csv")
    combined = pd.concat([df_signed, df_mdd, df_sign], ignore_index=True, sort=False)
    combined.to_csv(out_path, index=False)
    print(f"\n[OK] {out_path} ({len(combined)} lignes)")


if __name__ == "__main__":
    main()
