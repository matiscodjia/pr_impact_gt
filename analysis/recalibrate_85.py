#!/usr/bin/env python3
"""Recalibration du bruit sur les 85 cas (critique R2b : calibration mesurée sur n = 8).

Deux mesures d'accord GT- vs GT*, avec les quatre instruments de ``calibrate_noise.py``
(clDice, NSD@0.5mm, HD95, Betti0) :

  A. ``study``  -- les références RÉELLEMENT utilisées pour l'évaluation
     (``labelsTr_GT_minus_{omission,drift_neg,drift_pos}``, graine 42 + index du cas).
     C'est le chiffre qui compte : l'accord de la référence contre laquelle on a classé.
  B. ``grid``   -- la grille de ``results/noise_calibration.csv`` rejouée à l'identique
     (``generate(..., seed=0)``) sur 85 cas au lieu des 8 premiers, pour vérifier que le
     point retenu reste dans la fenêtre quand l'échantillon grandit.

Pour chaque configuration : moyenne, IC BCa 95 % de la moyenne, écart-type, médiane,
part des cas dans la fenêtre [0.85, 0.90] sur la métrique appariée, et moyenne sur les
8 premiers cas (le sous-échantillon de la calibration d'origine).

Sorties : results/calibration_85_per_case.csv, results/calibration_85_summary.csv

Usage
-----
    python analysis/recalibrate_85.py --workers 5
    python analysis/recalibrate_85.py --stats-only
    python analysis/recalibrate_85.py --limit 2 --workers 2      # essai rapide
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

from stats_utils import bca_mean_diff  # noqa: E402

RAW = "nnUNet_data/nnUNet_raw/Dataset100_PARSE"
WINDOW = (0.85, 0.90)          # fenêtre supposée (IAA_CLDICE de calibrate_noise.py)
MARGIN = 8
STUDY = {"omission": ("distal_omission", 2, 0.3, 0.0, "clDice"),
         "drift_neg": ("boundary_drift", 1, 0.5, -0.5, "NSD@0.5"),
         "drift_pos": ("boundary_drift", 1, 0.5, 0.5, "NSD@0.5")}
GRID = ([("distal_omission", r, p, 0.0, "clDice") for r, p in ((1, 0.5), (2, 0.3), (3, 0.3))]
        + [("boundary_drift", 1, 0.5, mu, "NSD@0.5") for mu in (-1.0, -0.5, 0.0, 0.5, 1.0)])
N_ORIG = 8                     # taille de la calibration d'origine (les 8 premiers cas triés)


def _bbox(mask, margin):
    sl = []
    for ax in range(3):
        nz = np.where(mask.any(axis=tuple(i for i in range(3) if i != ax)))[0]
        sl.append(slice(max(0, int(nz[0]) - margin), int(nz[-1]) + 1 + margin))
    return tuple(sl)


def _agreement(deg, star, spacing):
    from cross_evaluate import compute_betti0, compute_cldice, compute_hd95, compute_nsd
    return {"clDice": compute_cldice(deg, star),
            "NSD@0.5": compute_nsd(deg, star, spacing, tolerance=0.5),
            "HD95": compute_hd95(deg, star, spacing),
            "Betti0": compute_betti0(deg, star)}


def _task(args):
    import nibabel as nib
    from degradations import generate

    idx, case = args
    nii = nib.load(f"{RAW}/labelsTr/{case}")
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    star_full = np.asanyarray(nii.dataobj) > 0
    rows = []

    # A. références de l'étude : boîte de l'union (le drift+ peut déborder de GT*)
    stored = {k: np.asanyarray(nib.load(f"{RAW}/labelsTr_GT_minus_{k}/{case}").dataobj) > 0
              for k in STUDY}
    union = star_full.copy()
    for m in stored.values():
        union |= m
    ub = _bbox(union, 2)
    for k, (fam, r, p, mu, axis) in STUDY.items():
        rows.append({"case": case, "idx": idx, "source": "study", "config": k, "family": fam,
                     "r": r, "p": p, "mu": mu, "axis": axis,
                     **_agreement(stored[k][ub], star_full[ub], spacing)})

    # B. grille de calibration, protocole identique à calibrate_noise.py (boîte serrée, seed=0)
    bb = _bbox(star_full, 0)
    star = star_full[bb].astype(np.uint8)
    for fam, r, p, mu, axis in GRID:
        deg = generate(star.copy(), fam, r, p, seed=0, spacing=spacing, mu=mu)
        rows.append({"case": case, "idx": idx, "source": "grid",
                     "config": f"{fam}_r{r}_p{p}_mu{mu:+.1f}", "family": fam,
                     "r": r, "p": p, "mu": mu, "axis": axis,
                     **_agreement(deg, star, spacing)})
    return rows


def compute(args):
    cases = [os.path.basename(f) for f in sorted(glob.glob(f"{RAW}/labelsTr/*.nii.gz"))]
    tasks = list(enumerate(cases))
    if args.limit:
        tasks = tasks[:args.limit]
    with Pool(args.workers, maxtasksperchild=4) as pool:
        out = pool.map(_task, tasks, chunksize=1)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv("results/calibration_85_per_case.csv", index=False)
    return df


def summarise(df):
    rows = []
    keys = ["source", "config", "family", "r", "p", "mu", "axis"]
    for key, g in df.groupby(keys, sort=False):
        info = dict(zip(keys, key))
        v = g[info["axis"]].to_numpy(float)
        mean, lo, hi = bca_mean_diff(v, seed=42)
        first = g[g.idx < N_ORIG]
        rows.append({**info, "n": len(g),
                     "matched_mean": mean, "matched_ci_low": lo, "matched_ci_high": hi,
                     "matched_sd": v.std(ddof=1), "matched_median": float(np.median(v)),
                     "frac_cases_in_window": float(((v >= WINDOW[0]) & (v <= WINDOW[1])).mean()),
                     "mean_in_window": bool(WINDOW[0] <= mean <= WINDOW[1]),
                     "matched_mean_first8": first[info["axis"]].mean(),
                     **{f"{m}_mean": g[m].mean() for m in ("clDice", "NSD@0.5", "HD95", "Betti0")}})
    out = pd.DataFrame(rows)
    out.to_csv("results/calibration_85_summary.csv", index=False)
    pd.set_option("display.width", 240)
    print(out.drop(columns=["family", "r", "p", "mu"]).round(4).to_string(index=False))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="n'évaluer que les N premiers cas (essai)")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()
    df = pd.read_csv("results/calibration_85_per_case.csv") if args.stats_only else compute(args)
    summarise(df)


if __name__ == "__main__":
    main()
