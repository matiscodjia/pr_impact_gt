#!/usr/bin/env python3
"""Q3 -- Does a network learn a systematic annotation bias?

ANALYSIS PLAN FIXED BEFORE THE DATA. This file was committed before any M3/M4 prediction
existed (training not started on 2026-09-25); any analysis added afterwards must be
labelled exploratory. (Translated to English on 2026-10-12 without changing its content;
the original wording is in commit 3630e72.)

Models
------
M0 : trained on GT* (the study, seed 0, results/metrics.csv).
M3 : trained on the frozen GT- drift mu = -0.5 (Dataset103), labels = labelsTr_GT_minus_drift_neg.
M4 : trained on the frozen GT- drift mu = +0.5 (Dataset104), labels = labelsTr_GT_minus_drift_pos.
Same architecture, same nnU-Net plans (normalisation included), same folds, 500 epochs:
only the training labels differ (cluster/prepare_q3.sh). Each Mk is evaluated out of fold,
paired with M0 on the SAME cases.

Why the primary contrast is M4 against M3, and not Mk against M0
-----------------------------------------------------------------
Observed BEFORE any M3/M4 data (results/metrics.csv, 85 cases): M2, trained on the
"unbiased" boundary noise mu = 0, already predicts 5.4 % less volume than M0 on GT*
(BCa CI [-6.0; -4.9] %, 85/85 cases) -- more than the whole bias of the drift- reference
(-4.3 %). Training on noisy boundaries therefore makes a network under-segment, bias or
not. Compared with M0, M3 would under-segment "for the right reasons" even without
learning anything about mu. M3 and M4 share everything (family, r, p, frozen realisation,
regime) except the SIGN of mu: their difference cancels this common effect and keeps only
the network's response to the bias.

Primary hypotheses (3 tests, Holm correction)
---------------------------------------------
H1 -- the bias is learned (volume). Per case i, dV = (|P| - |GT*|) / |GT*| (signed):
    diff_i = dV(M4|GT*)_i - dV(M3|GT*)_i ;  ref_i = dV(GT-+|GT*)_i - dV(GT--|GT*)_i (~ +0.115)
    Test: two-sided paired Wilcoxon on diff (predicted direction: > 0). Effect size: learned
    fraction lambda+- = mean(diff) / mean(ref), 95 % BCa CI (cases resampled, numerator and
    denominator drawn together). lambda+- = 0: the network ignores the sign of the bias;
    lambda+- = 1: it reproduces the whole volume gap between the two annotators.
    Supported iff p_Holm < 0.05, mean(diff) > 0 AND the lower CI bound of lambda+- > 0.
H2 -- the learned bias reverses the ranking (evaluation consequence), one test per Mk.
    Metric: NSD@0.5 (the instrument matched to the boundary mechanism, as in the study).
    Pair (M0, Mk), move GT* -> GT-_k (drift- for M3, drift+ for M4), per-case interaction
    iota = [m(M0|GT-) - m(Mk|GT-)] - [m(M0|GT*) - m(Mk|GT*)]. Reversal iff p_Holm < 0.05
    (Wilcoxon on iota), M0 ahead on GT* AND the sign of mean(M0 - Mk) changes -- the same
    rule as the study's 4 reversals. Here the comparison with M0 is the right one: the
    question is "against which reference does the clean model lose?", not the mechanism.

Secondary (pre-specified, BH together with the exploratory family, read in the light of
the M2 observation)
------------------------------------------------------------------------------------------
- lambda per direction against M0: [dV(Mk) - dV(M0)] / dV(GT-_k|GT*), and the same quantity
  for M2 (control).

Exploratory (Benjamini-Hochberg within this family, never promoted to a main result)
-------------------------------------------------------------------------------------
- H2 interaction on HD95, NSD@2, clDice, Betti-0, |dV|;
- dV(Mk|GT-_k): how closely the volume of Mk matches the biased reference (0 = perfect);
- dV(Mk) - dV(M2) per direction (descriptive: M2 is on-the-fly, Mk frozen -- two differences).

What the analysis cannot say: n = cases of the available folds (34 for 2 folds, 85 for 5);
one training per regime and per fold (no seed variance for M3/M4); M0 is the study's
(seed 0).

Inputs  : results/metrics.csv (M0, M2), results/cluster_runs/*/metrics.csv (M3, M4, scored on
          the GPU node), results/reference_severity.csv (bias of each reference vs GT*).
Output  : results/q3_learned_bias.csv (+ printed table)

    python analysis/q3_learned_bias.py
    python analysis/q3_learned_bias.py --selftest     # synthetic data with a known lambda
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

from prgt.stats import bca_bounds, bh_fdr, paired_cohens_d, wilcoxon_p_safe

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
    jack = (num.sum() - num) / (den.sum() - den)          # ratio of means with one case left out
    est = num.mean() / den.mean()
    lo, hi = bca_bounds(est, boot, jack)
    return est, lo, hi


def load(extra_glob):
    frames = [pd.read_csv("results/metrics.csv")]
    for f in sorted(glob.glob(extra_glob)):
        d = pd.read_csv(f)
        frames.append(d[d.model.isin(MODELS)])
    df = pd.concat(frames, ignore_index=True)
    df = df[df.eval_kind == "oof"] if "eval_kind" in df else df
    df["volume_delta_abs"] = df.volume_delta.abs()
    df = df.drop_duplicates(KEY, keep="last")               # one CSV per folder: duplicates are possible
    sev = pd.read_csv("results/reference_severity.csv")[["case", "scenario", "dV_ref_vs_star"]]
    return df, sev


def wide(df, metric):
    return df.pivot_table(index="case", columns=["model", "scenario"], values=metric)


def h1_row(df, sev, model_a, ref_a, model_b, ref_b, label):
    """lambda = mean(dV(a|GT*) - dV(b|GT*)) / mean(bias(ref_a) - bias(ref_b)); ref None = GT* (bias 0)."""
    v = wide(df, "volume_delta")
    need = [(model_a, "GT_star"), (model_b, "GT_star")]
    if not all(c in v for c in need):
        return None
    def bias(r):
        return sev[sev.scenario == r].set_index("case").dV_ref_vs_star if r else 0.0
    d = pd.DataFrame({"a": v[need[0]], "b": v[need[1]]}).dropna()
    d["ref"] = (bias(ref_a) - bias(ref_b)).reindex(d.index)   # ref_a is never None here
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
    sign = 1 if HIGHER_BETTER[metric] else -1               # > 0: M0 ahead
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
    """Inject a known lambda into synthetic M3/M4 built from M0; the analysis must recover it."""
    df, sev = load("__none__")
    rng = np.random.default_rng(0)
    base = df[df.model == M0].copy()
    synth = []
    for (model, ref), lam in zip(MODELS.items(), (0.6, 0.6)):   # same fraction in both directions
        s = sev[sev.scenario == ref].set_index("case").dV_ref_vs_star
        for scen in ("GT_star", ref):
            b = base[base.scenario == scen].copy()
            b["model"] = model
            b["volume_delta"] = b.volume_delta + lam * b.case.map(s) + rng.normal(0, 0.005, len(b))
            b["nsd05"] = b.nsd05 + (-0.02 if scen == "GT_star" else 0.02)   # built-in reversal
            synth.append(b)
    df = pd.concat([df] + synth, ignore_index=True)
    df["volume_delta_abs"] = df.volume_delta.abs()
    out, _ = analyse(df, sev)
    p = out[out.family == "primary_holm"]
    print(p[["test", "model", "n", "lambda", "lambda_ci_low", "lambda_ci_high", "delta_star", "delta_ref",
             "p_adj", "supported"]].round(4).to_string(index=False))
    lam = float(p[p.test == "H1_bias_learned"]["lambda"].iloc[0])
    assert abs(lam - 0.6) < 0.05, lam
    assert p.supported.all() and len(p) == 3, "the 3 hypotheses built to be true must be supported"
    print("selftest OK: injected lambda+- = 0.6 recovered, built-in reversals detected")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds-glob", default="results/cluster_runs/*/metrics.csv")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    df, sev = load(args.seeds_glob)
    have = sorted(set(df.model) & set(MODELS))
    if not have:
        raise SystemExit("no M3/M4 rows: copy results_seeds/M*/metrics.csv from the GPU node to results/cluster_runs/")
    for m in have:
        sc = sorted(df[df.model == m].scenario.unique())
        folds = sorted(df[df.model == m].fold.unique())
        print(f"{m}: folds {folds}, {df[df.model == m].case.nunique()} cases, scenarios {sc}")
        if MODELS[m] not in sc:
            print(f"  !! no score against {MODELS[m]}: H2 cannot be computed "
                  "(GT- references missing on the node when the scores were computed?)")
    out, missing = analyse(df, sev)
    out.to_csv("results/q3_learned_bias.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["family", "test", "model", "metric", "n", "lambda", "lambda_ci_low", "lambda_ci_high",
            "mean_diff", "delta_star", "delta_ref", "interaction", "d", "p_adj", "supported"]
    print(out[[c for c in cols if c in out]].round(4).to_string(index=False))
    if missing:
        print(f"primary tests that cannot be computed: {missing}")


if __name__ == "__main__":
    main()
