#!/usr/bin/env python3
"""Règle MDD appliquée à TOUS les retournements rapportés (cohérence interne).

Question (relecture simulée n°4) : pour écarter deux contrastes |ΔV| sous drift⁻, l'étude
exige que l'écart APRÈS croisement dépasse la différence minimale détectable ; les
retournements « précoces » de la grille de sévérité (p = 0,1-0,2) sont eux aussi des
croisements minuscules. Même règle partout, sinon double standard.

Règle (identique à verify_reversal_robustness.py, check (0b)) :
    MDD = (z_0.975 + z_0.80) · sd(ι) / sqrt(n)          (80 % de puissance, bilatéral 5 %)
    croisement résolu  ⇔  signe(Δ_GT*) ≠ signe(Δ_GT⁻)  ET  |Δ_GT⁻| ≥ MDD
    retournement strict ⇔ croisement résolu ET Wilcoxon(ι) p brut < 0,05
avec Δ = moyenne par cas de m(M0) − m(Mk), ι = Δ_GT⁻ − Δ_GT* par cas.

Sources (métriques par cas déjà calculées, aucun recalcul d'image) :
    grid        results/dose_response_fine_metrics.csv + dose_response_p05_metrics.csv
    draws       results/dose_response_draw*_metrics.csv (p = 0,3)
    structured  results/structured_omission_metrics.csv
    lcc         results/metrics_lcc.csv (postprocessing == lcc)

p bruts : hors de la famille FDR des 126 tests de l'étude.
Sortie : results/mdd_crossing.csv
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_utils import wilcoxon_p_safe  # noqa: E402

Z = norm.ppf(0.975) + norm.ppf(0.80)
OTHERS = ("M1_Omission", "M2_Drift_mu0")


def _long(path, source, ref_map=None):
    d = pd.read_csv(path)
    if ref_map:
        d = d[d.ref.isin(ref_map)].assign(ref=lambda x: x.ref.map(ref_map))
    return d.assign(source=source)


def load():
    frames = [_long("results/dose_response_fine_metrics.csv", "grid"),
              _long("results/dose_response_p05_metrics.csv", "grid", {"GT_star": "GT_star", "p0.5": "p0.5"})]
    for f in sorted(glob.glob("results/dose_response_draw*_metrics.csv")):
        frames.append(_long(f, "draw" + re.search(r"draw(\d+)", f).group(1)))
    frames.append(_long("results/structured_omission_metrics.csv", "structured"))
    lcc = pd.read_csv("results/metrics_lcc.csv")
    lcc = lcc[(lcc.postprocessing == "lcc") & lcc.scenario.isin(["GT_star", "GT_minus_omission"])]
    frames.append(lcc.rename(columns={"scenario": "ref"})[["case", "model", "ref", "hd95", "nsd"]]
                  .assign(source="lcc"))
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["source", "case", "model", "ref"], keep="first")


def main():
    df = load()
    rows = []
    for (source, metric), g in df.melt(id_vars=["source", "case", "model", "ref"], value_vars=["hd95", "nsd"],
                                        var_name="metric").groupby(["source", "metric"]):
        w = g.pivot_table(index=["case", "ref"], columns="model", values="value").reset_index()
        star = w[w.ref == "GT_star"].set_index("case")
        for other in OTHERS:
            d_star = star["M0_Star"] - star[other]
            for ref in sorted(set(w.ref) - {"GT_star"}):
                d_ref = w[w.ref == ref].set_index("case")["M0_Star"] - w[w.ref == ref].set_index("case")[other]
                common = d_ref.index.intersection(d_star.index)
                a, b = d_ref.loc[common].values, d_star.loc[common].values
                iota = a - b
                mdd = Z * np.std(iota, ddof=1) / np.sqrt(len(iota))
                flip = bool(np.sign(b.mean()) != np.sign(a.mean()))
                p = wilcoxon_p_safe(a, b)
                rows.append({"source": source, "ref": ref, "metric": metric, "pair": f"M0_vs_{other}",
                             "delta_star": b.mean(), "delta_ref": a.mean(), "iota": iota.mean(),
                             "mdd": mdd, "sign_flip": flip,
                             "crossing_resolved": flip and abs(a.mean()) >= mdd,
                             "p_raw": p, "strict_reversal": flip and abs(a.mean()) >= mdd and p < 0.05,
                             "n": len(iota)})
    out = pd.DataFrame(rows)
    out.to_csv("results/mdd_crossing.csv", index=False)
    pd.set_option("display.width", 220)
    grid = out[out.source == "grid"].sort_values(["metric", "pair", "ref"])
    print(grid.drop(columns=["source", "n"]).round(4).to_string(index=False))
    rest = out[out.source != "grid"].copy()
    rest["group"] = rest.source.where(~rest.source.str.startswith("draw"), "draws")
    rest.loc[rest.source == "structured", "group"] = "structured:" + rest.ref.str.replace(r"_\d+$", "", regex=True)
    print(rest.groupby(["group", "metric", "pair"]).agg(
        n=("sign_flip", "size"), flips=("sign_flip", "sum"), resolved=("crossing_resolved", "sum"),
        strict=("strict_reversal", "sum"), min_margin_over_mdd=("delta_ref", "size")).to_string())
    rest["ratio"] = rest.delta_ref.abs() / rest.mdd
    print(rest.groupby(["group", "metric", "pair"]).ratio.agg(["min", "max"]).round(2).to_string())


if __name__ == "__main__":
    main()
