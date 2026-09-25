#!/usr/bin/env python3
"""Q3 -- Un réseau apprend-il un biais systématique d'annotation ?

PLAN D'ANALYSE FIXÉ AVANT LES DONNÉES. Ce fichier est commité avant qu'aucune prédiction
de M3/M4 n'existe (entraînement non lancé au 2026-09-25) ; toute analyse ajoutée après
coup doit être étiquetée exploratoire.

Modèles
-------
M0 : entraîné sur GT* (étude, graine 0, results/metrics.csv).
M3 : entraîné sur GT⁻ drift μ=−0,5 figée (Dataset103), labels = labelsTr_GT_minus_drift_neg.
M4 : entraîné sur GT⁻ drift μ=+0,5 figée (Dataset104), labels = labelsTr_GT_minus_drift_pos.
Même architecture, mêmes plans nnU-Net (y compris normalisation), mêmes plis, 500 époques :
seuls les labels d'entraînement diffèrent (scripts/cluster/prepare_q3.sh). Chaque Mk est
évalué out-of-fold, apparié à M0 sur les MÊMES cas.

Pourquoi le contraste principal est M4 contre M3, et pas Mk contre M0
------------------------------------------------------------------------
Constat fait AVANT toute donnée M3/M4 (results/metrics.csv, 85 cas) : M2, entraîné sur le
bruit de bord « sans biais » μ=0, prédit déjà 5,4 % de volume de moins que M0 sur GT*
(IC BCa [−6,0 ; −4,9] %, 85/85 cas) -- plus que tout le biais de la référence drift−
(−4,3 %). Entraîner sur des bords bruités fait donc sous-segmenter, biais ou pas. Comparé à
M0, M3 sous-segmenterait « pour de bonnes raisons » même sans rien apprendre de μ. M3 et M4
partagent tout (famille, r, p, réalisation figée, régime) sauf le SIGNE de μ : leur
différence annule cet effet commun et ne garde que la réponse du réseau au biais.

Hypothèses principales (3 tests, correction de Holm)
----------------------------------------------------
H1 -- le biais est appris (volume). Par cas i, ΔV = (|P| − |GT*|)/|GT*| (signé) :
    diff_i = ΔV(M4|GT*)_i − ΔV(M3|GT*)_i ;  réf_i = ΔV(GT⁻+|GT*)_i − ΔV(GT⁻−|GT*)_i (≈ +0,115)
    Test : Wilcoxon apparié bilatéral sur diff (sens prédit : > 0). Taille d'effet : fraction
    apprise λ± = moyenne(diff) / moyenne(réf), IC BCa 95 % (bootstrap des cas, numérateur et
    dénominateur tirés ensemble). λ± = 0 : le réseau ignore le signe du biais ; λ± = 1 : il
    reproduit tout l'écart de volume entre les deux annotateurs.
    Soutenue ssi p_Holm < 0,05, moyenne(diff) > 0 ET borne basse de l'IC de λ± > 0.
H2 -- le biais appris renverse le classement (conséquence d'évaluation), un test par Mk.
    Métrique : NSD@0.5 (instrument apparié au mécanisme de bord, comme dans l'étude).
    Paire (M0, Mk), passage GT* → GT⁻_k (drift− pour M3, drift+ pour M4), interaction par cas
    ι = [m(M0|GT⁻)−m(Mk|GT⁻)] − [m(M0|GT*)−m(Mk|GT*)]. Renversement ssi p_Holm < 0,05
    (Wilcoxon sur ι), M0 devant sur GT* ET changement de signe de moyenne(M0−Mk) -- même
    règle que les 4 renversements de l'étude. Ici la comparaison à M0 est la bonne : la
    question est « contre quelle référence le modèle propre perd-il ? », pas le mécanisme.

Secondaire (pré-spécifié, BH avec l'exploratoire, interprété à la lumière du constat M2)
------------------------------------------------------------------------------------------
- λ par sens contre M0 : [ΔV(Mk) − ΔV(M0)] / ΔV(GT⁻_k|GT*), et même quantité pour M2 (contrôle).

Exploratoire (Benjamini-Hochberg dans cette famille, jamais promu en résultat principal)
-------------------------------------------------------------------------------------------
- interaction H2 sur HD95, NSD@2, clDice, Betti-0, |ΔV| ;
- ΔV(Mk|GT⁻_k) : à quel point le volume de Mk colle à la référence biaisée (0 = parfait) ;
- ΔV(Mk) − ΔV(M2) par sens (descriptif : M2 est à la volée, Mk figé -- deux différences).

Ce que l'analyse ne dit pas : n = cas des plis disponibles (34 pour 2 plis, 85 pour 5) ;
un seul entraînement par régime et par pli (pas de variance de graine pour M3/M4) ; M0 est
celui de l'étude (graine 0).

Entrées  : results/metrics.csv (M0, M2), results_seeds/*/metrics.csv (M3, M4, poussés depuis
           le cluster), results/reference_severity.csv (biais de chaque référence vs GT*).
Sorties  : results/q3_learned_bias.csv (+ tableau imprimé)
Usage    : python analysis/q3_learned_bias.py
           python analysis/q3_learned_bias.py --selftest     # données synthétiques, λ connu
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_utils import _bca_bounds, bh_fdr, paired_cohens_d, wilcoxon_p_safe  # noqa: E402

MODELS = {"M3_Drift_muMinus": "GT_minus_drift_neg", "M4_Drift_muPlus": "GT_minus_drift_pos"}
M0, M2 = "M0_Star", "M2_Drift_mu0"
H2_METRIC = "nsd05"
EXPLORATORY = ["hd95", "nsd", "cldice", "betti0", "volume_delta_abs"]
HIGHER_BETTER = {"nsd05": True, "nsd": True, "cldice": True, "hd95": False, "betti0": False,
                 "volume_delta_abs": False}
KEY = ["model", "scenario", "case"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def ratio_of_means(num, den, n_boot=10000, seed=42):
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))
    boot = num[idx].mean(1) / den[idx].mean(1)
    jack = (num.sum() - num) / (den.sum() - den)          # ratio des moyennes, un cas retiré
    est = num.mean() / den.mean()
    lo, hi = _bca_bounds(est, boot, jack)
    return est, lo, hi


def load(extra_glob):
    frames = [pd.read_csv("results/metrics.csv")]
    for f in sorted(glob.glob(extra_glob)):
        d = pd.read_csv(f)
        frames.append(d[d.model.isin(MODELS)])
    df = pd.concat(frames, ignore_index=True)
    df = df[df.eval_kind == "oof"] if "eval_kind" in df else df
    df["volume_delta_abs"] = df.volume_delta.abs()
    df = df.drop_duplicates(KEY, keep="last")               # un CSV par dossier : doublons possibles
    sev = pd.read_csv("results/reference_severity.csv")[["case", "scenario", "dV_ref_vs_star"]]
    return df, sev


def wide(df, metric):
    return df.pivot_table(index="case", columns=["model", "scenario"], values=metric)


def h1_row(df, sev, model_a, ref_a, model_b, ref_b, label):
    """λ = moyenne(ΔV(a|GT*) − ΔV(b|GT*)) / moyenne(biais(réf_a) − biais(réf_b)) ; réf None = GT* (biais 0)."""
    v = wide(df, "volume_delta")
    need = [(model_a, "GT_star"), (model_b, "GT_star")]
    if not all(c in v for c in need):
        return None
    def bias(r):
        return sev[sev.scenario == r].set_index("case").dV_ref_vs_star if r else 0.0
    d = pd.DataFrame({"a": v[need[0]], "b": v[need[1]]}).dropna()
    d["ref"] = (bias(ref_a) - bias(ref_b)).reindex(d.index)   # ref_a jamais None ici
    d = d.dropna()
    diff, den = (d.a - d.b).to_numpy(), d.ref.to_numpy()
    lam, lo, hi = ratio_of_means(diff, den)
    return {"test": label, "model": f"{model_a} vs {model_b}", "reference": f"{ref_a} vs {ref_b or 'GT_star'}",
            "metric": "volume_delta", "n": len(d), "mean_diff": diff.mean(), "mean_ref_bias": den.mean(),
            "lambda": lam, "lambda_ci_low": lo, "lambda_ci_high": hi, "d": paired_cohens_d(diff),
            "p_raw": wilcoxon_p_safe(d.a, d.b), "sign_as_predicted": bool(np.sign(diff.mean()) == np.sign(den.mean()))}


def h2_row(df, model, ref_scen, metric, label):
    w = wide(df, metric)
    cols = [(M0, "GT_star"), (model, "GT_star"), (M0, ref_scen), (model, ref_scen)]
    if not all(c in w for c in cols):
        return None
    d = w[cols].dropna()
    d_star = d[cols[0]] - d[cols[1]]
    d_ref = d[cols[2]] - d[cols[3]]
    inter = (d_ref - d_star).to_numpy()
    sign = 1 if HIGHER_BETTER[metric] else -1               # >0 : M0 devant
    return {"test": label, "model": model, "reference": ref_scen, "metric": metric, "n": len(d),
            "delta_star": d_star.mean(), "delta_ref": d_ref.mean(), "interaction": inter.mean(),
            "d": paired_cohens_d(inter), "p_raw": wilcoxon_p_safe(d_ref, d_star),
            "m0_ahead_on_star": bool(sign * d_star.mean() > 0),
            "sign_flip": bool(np.sign(d_star.mean()) != np.sign(d_ref.mean()))}


def analyse(df, sev):
    (m3, r3), (m4, r4) = MODELS.items()
    prim = [("H1_bias_learned", h1_row(df, sev, m4, r4, m3, r3, "H1_bias_learned")),
            ("H2_reversal " + m3, h2_row(df, m3, r3, H2_METRIC, "H2_reversal")),
            ("H2_reversal " + m4, h2_row(df, m4, r4, H2_METRIC, "H2_reversal"))]
    missing = [name for name, r in prim if r is None]
    prim = [r for _, r in prim if r]
    expl = []
    for model, ref in MODELS.items():
        expl.append(h1_row(df, sev, model, ref, M0, None, "sec_lambda_vs_M0"))
        expl.append(h1_row(df, sev, M2, ref, M0, None, "sec_control_M2_lambda_vs_M0"))
        v = wide(df, "volume_delta")
        if (model, "GT_star") in v and (M2, "GT_star") in v:
            x = (v[(model, "GT_star")] - v[(M2, "GT_star")]).dropna()
            expl.append({"test": "sec_dV_minus_M2", "model": f"{model} vs {M2}", "reference": "GT_star",
                         "metric": "volume_delta", "n": len(x), "mean_diff": x.mean()})
        if (model, ref) in v:
            x = v[(model, ref)].dropna()
            expl.append({"test": "expl_volume_vs_biased_ref", "model": model, "reference": ref,
                         "metric": "volume_delta", "n": len(x), "mean_diff": x.mean()})
        for m in EXPLORATORY:
            expl.append(h2_row(df, model, ref, m, "expl_interaction"))
    expl = [r for r in expl if r]
    out = pd.DataFrame(prim + expl)
    if prim:
        k = len(prim)
        out.loc[:k - 1, "p_adj"] = holm(out.p_raw[:k])
        out.loc[:k - 1, "family"] = "primary_holm"
        ok = out.p_adj[:k] < 0.05
        h1 = out.test[:k] == "H1_bias_learned"
        col = lambda c: out[c][:k].fillna(False).astype(bool) if c in out else pd.Series(False, index=out.index[:k])
        low = out["lambda_ci_low"][:k] > 0 if "lambda_ci_low" in out else False
        out.loc[:k - 1, "supported"] = np.where(h1, ok & col("sign_as_predicted") & low,
                                                ok & col("sign_flip") & col("m0_ahead_on_star"))
    ex = ~out.family.eq("primary_holm") & out.p_raw.notna() if "family" in out else out.p_raw.notna()
    if ex.any():
        out.loc[ex, "p_adj"] = bh_fdr(out.loc[ex, "p_raw"].to_numpy())
        out.loc[ex, "family"] = "secondary_exploratory_bh"
    out["family"] = out["family"].fillna("descriptive") if "family" in out else "descriptive"
    return out, missing


def selftest():
    """λ connu injecté dans des M3/M4 synthétiques construits à partir de M0 : on doit le retrouver."""
    df, sev = load("__none__")
    rng = np.random.default_rng(0)
    base = df[df.model == M0].copy()
    synth = []
    for (model, ref), lam in zip(MODELS.items(), (0.6, 0.6)):   # même fraction des deux côtés
        s = sev[sev.scenario == ref].set_index("case").dV_ref_vs_star
        for scen in ("GT_star", ref):
            b = base[base.scenario == scen].copy()
            b["model"] = model
            b["volume_delta"] = b.volume_delta + lam * b.case.map(s) + rng.normal(0, 0.005, len(b))
            b["nsd05"] = b.nsd05 + (-0.02 if scen == "GT_star" else 0.02)   # renversement construit
            synth.append(b)
    df = pd.concat([df] + synth, ignore_index=True)
    df["volume_delta_abs"] = df.volume_delta.abs()
    out, _ = analyse(df, sev)
    p = out[out.family == "primary_holm"]
    print(p[["test", "model", "n", "lambda", "lambda_ci_low", "lambda_ci_high", "delta_star", "delta_ref",
             "p_adj", "supported"]].round(4).to_string(index=False))
    lam = float(p[p.test == "H1_bias_learned"]["lambda"].iloc[0])
    assert abs(lam - 0.6) < 0.05, lam
    assert p.supported.all() and len(p) == 3, "les 3 hypothèses construites vraies doivent être soutenues"
    print("selftest OK : λ± injecté 0,6 retrouvé, renversements construits détectés")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds-glob", default="results_seeds/*/metrics.csv")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    df, sev = load(args.seeds_glob)
    have = sorted(set(df.model) & set(MODELS))
    if not have:
        raise SystemExit("aucune ligne M3/M4 : pousser results_seeds/M*/metrics.csv depuis le cluster")
    for m in have:
        sc = sorted(df[df.model == m].scenario.unique())
        folds = sorted(df[df.model == m].fold.unique())
        print(f"{m} : plis {folds}, {df[df.model == m].case.nunique()} cas, scénarios {sc}")
        if MODELS[m] not in sc:
            print(f"  !! pas de score contre {MODELS[m]} : H2 impossible "
                  "(références GT⁻ absentes du cluster au moment du calcul ?)")
    out, missing = analyse(df, sev)
    out.to_csv("results/q3_learned_bias.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["family", "test", "model", "metric", "n", "lambda", "lambda_ci_low", "lambda_ci_high",
            "mean_diff", "delta_star", "delta_ref", "interaction", "d", "p_adj", "supported"]
    print(out[[c for c in cols if c in out]].round(4).to_string(index=False))
    if missing:
        print(f"tests principaux non calculables pour : {missing}")


if __name__ == "__main__":
    main()
