#!/usr/bin/env python3
"""Connected-component counts behind the Betti-0 error (paper: "A caution on Betti-0").

The paper states that, under omission, the reference changes (GT- is fragmented) but the
Betti-0 error cannot show it between models: if every prediction has more components than
both references, |Delta N| = N(P) - N(ref), and N(ref) cancels exactly in any difference
between models. This script counts N(GT*), N(GT-) and N(P) directly (26-connectivity, as
in prgt.metrics.compute_betti0) instead of inferring them, and checks that:

  1. N(GT*) = 1 for every case;
  2. N(P) > max(N(GT*), N(GT-)) for every case and every model;
  3. the recomputed |N(P) - N(ref)| equals the ``betti0`` column of results/metrics.csv;
  4. the between-model Betti-0 interaction is zero case by case.

Outputs: results/betti0_counts.csv (one row per case), summary on stdout.

    python analysis/betti0_counts.py --workers 3
    python analysis/betti0_counts.py --stats-only
"""

from __future__ import annotations

import argparse
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

from prgt.folds import case_fold_map

RAW = "nnUNet_data/nnUNet_raw/Dataset100_PARSE"
RES = "nnUNet_data/nnUNet_results/Dataset100_PARSE"
TRAINERS = {"M0_Star": "nnUNetTrainerStd",
            "M1_Omission": "nnUNetTrainerDegradedOmissionOnly",
            "M2_Drift_mu0": "nnUNetTrainerDriftMu0"}
REFS = {"GT_star": "labelsTr",
        "GT_minus_omission": "labelsTr_GT_minus_omission",
        "GT_minus_drift_neg": "labelsTr_GT_minus_drift_neg",
        "GT_minus_drift_pos": "labelsTr_GT_minus_drift_pos"}
OUT = "results/betti0_counts.csv"


def _n_components(path):
    import nibabel as nib
    from scipy.ndimage import generate_binary_structure, label
    mask = np.asanyarray(nib.load(path).dataobj) > 0.5
    return int(label(mask, structure=generate_binary_structure(3, 3))[1])


def _task(args):
    case, fold = args
    row = {"case": case, "fold": fold}
    for ref, sub in REFS.items():
        row[f"N_{ref}"] = _n_components(f"{RAW}/{sub}/{case}")
    for m, t in TRAINERS.items():
        row[f"N_{m}"] = _n_components(f"{RES}/{t}__nnUNetPlans__3d_fullres/fold_{fold}/validation/{case}")
    return row


def compute(workers):
    folds = case_fold_map()
    with Pool(workers, maxtasksperchild=4) as pool:
        rows = pool.map(_task, sorted(folds.items()), chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    return df


def check(df):
    models, refs = list(TRAINERS), list(REFS)
    print(f"n = {len(df)} cases")
    print(f"1. N(GT*) = 1 everywhere: {bool((df.N_GT_star == 1).all())} "
          f"(values: {sorted(df.N_GT_star.unique())})")
    for ref in refs[1:]:
        v = df[f"N_{ref}"]
        print(f"   N({ref}): mean {v.mean():.1f}, median {v.median():.0f}, "
              f"range {v.min()}-{v.max()}")
    for m in models:
        v = df[f"N_{m}"]
        print(f"   N({m}): mean {v.mean():.1f}, median {v.median():.0f}, range {v.min()}-{v.max()}")

    print("2. N(P) > N(ref), number of cases out of", len(df))
    for m in models:
        print(f"   {m:<13}" + "  ".join(f"{r}: {int((df[f'N_{m}'] > df[f'N_{r}']).sum())}" for r in refs))

    met = pd.read_csv("results/metrics.csv")
    met = met[met.eval_kind == "oof"]
    bad = 0
    for m in models:
        for r in refs:
            got = met[(met.model == m) & (met.scenario == r)].set_index("case").betti0
            mine = (df[f"N_{m}"] - df[f"N_{r}"]).abs().set_axis(df.case)
            bad += int((got.reindex(mine.index) != mine).sum())
    print(f"3. |N(P) - N(ref)| != metrics.csv betti0: {bad} rows out of {len(models) * len(refs) * len(df)}")

    print("4. zero Betti-0 interaction (cases out of", len(df), ")")
    for a, b in itertools.combinations(models, 2):
        for r in refs[1:]:
            d_ref = ((df[f"N_{a}"] - df[f"N_{r}"]).abs() - (df[f"N_{b}"] - df[f"N_{r}"]).abs())
            d_star = ((df[f"N_{a}"] - df.N_GT_star).abs() - (df[f"N_{b}"] - df.N_GT_star).abs())
            print(f"   {a}/{b} {r:<19} {int(((d_ref - d_star) == 0).sum())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--stats-only", action="store_true", help="re-read results/betti0_counts.csv")
    args = ap.parse_args()
    df = pd.read_csv(OUT) if args.stats_only else compute(args.workers)
    check(df)


if __name__ == "__main__":
    main()
