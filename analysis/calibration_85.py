#!/usr/bin/env python3
"""Noise calibration measured on all 85 cases (the original calibration used n = 8).

Two measures of agreement between GT- and GT*, with the four instruments of
pipeline/calibrate_noise.py (clDice, NSD@0.5mm, HD95, Betti-0):

  A. ``study`` -- the references ACTUALLY used for evaluation
     (labelsTr_GT_minus_{omission,drift_neg,drift_pos}, seed 42 + case index). This is the
     number that matters: the agreement of the reference the models were ranked against.
  B. ``grid``  -- the grid of results/noise_calibration.csv replayed identically
     (apply_noise(..., seed=0)) on 85 cases instead of the first 8, to check that the
     chosen operating point stays in the window as the sample grows.

For each configuration: mean, 95 % BCa interval of the mean, standard deviation, median,
fraction of cases inside the window [0.85, 0.90] on the matched metric, and the mean over
the first 8 cases (the subsample of the original calibration).

Outputs: results/calibration_85_per_case.csv, results/calibration_85_summary.csv

    python analysis/calibration_85.py --workers 5
    python analysis/calibration_85.py --stats-only
    python analysis/calibration_85.py --limit 2 --workers 2      # quick trial
"""

from __future__ import annotations

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from prgt.metrics import union_bbox
from prgt.stats import bca_mean_diff

RAW = "nnUNet_data/nnUNet_raw/Dataset100_PARSE"
WINDOW = (0.85, 0.90)          # calibration window (IAA_CLDICE in pipeline/calibrate_noise.py)
MARGIN = 8
STUDY = {"omission": ("distal_omission", 2, 0.3, 0.0, "clDice"),
         "drift_neg": ("boundary_drift", 1, 0.5, -0.5, "NSD@0.5"),
         "drift_pos": ("boundary_drift", 1, 0.5, 0.5, "NSD@0.5")}
GRID = ([("distal_omission", r, p, 0.0, "clDice") for r, p in ((1, 0.5), (2, 0.3), (3, 0.3))]
        + [("boundary_drift", 1, 0.5, mu, "NSD@0.5") for mu in (-1.0, -0.5, 0.0, 0.5, 1.0)])
N_ORIG = 8                     # size of the original calibration (first 8 cases, sorted)


def _agreement(deg, star, spacing):
    from prgt.metrics import compute_betti0, compute_cldice, compute_hd95, compute_nsd
    return {"clDice": compute_cldice(deg, star),
            "NSD@0.5": compute_nsd(deg, star, spacing, tolerance=0.5),
            "HD95": compute_hd95(deg, star, spacing),
            "Betti0": compute_betti0(deg, star)}


def _task(args):
    import nibabel as nib
    from prgt.noise import apply_noise

    idx, case = args
    nii = nib.load(f"{RAW}/labelsTr/{case}")
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    star_full = np.asanyarray(nii.dataobj) > 0
    rows = []

    # A. the study's references, cropped to the union box (drift+ can extend beyond GT*)
    stored = {k: np.asanyarray(nib.load(f"{RAW}/labelsTr_GT_minus_{k}/{case}").dataobj) > 0
              for k in STUDY}
    union = star_full.copy()
    for m in stored.values():
        union |= m
    ub = union_bbox([union], 2)
    for k, (fam, r, p, mu, axis) in STUDY.items():
        rows.append({"case": case, "idx": idx, "source": "study", "config": k, "family": fam,
                     "r": r, "p": p, "mu": mu, "axis": axis,
                     **_agreement(stored[k][ub], star_full[ub], spacing)})

    # B. calibration grid, same protocol as pipeline/calibrate_noise.py (tight box, seed 0)
    bb = union_bbox([star_full], 0)
    star = star_full[bb].astype(np.uint8)
    for fam, r, p, mu, axis in GRID:
        deg = apply_noise(star.copy(), fam, r, p, seed=0, spacing=spacing, mu=mu)
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
    ap.add_argument("--limit", type=int, default=0, help="only evaluate the first N cases (trial)")
    ap.add_argument("--stats-only", action="store_true", help="re-read the per-case CSV")
    args = ap.parse_args()
    df = pd.read_csv("results/calibration_85_per_case.csv") if args.stats_only else compute(args)
    summarise(df)


if __name__ == "__main__":
    main()
