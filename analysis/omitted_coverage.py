#!/usr/bin/env python3
"""How much of the omitted regions do the predictions cover? (premise of the mechanism)

Proposed mechanism for the reversal under omission: M0 (trained on GT*) predicts thin
structures that the omission reference GT- no longer contains. Two direct measures, with
no new inference:

  (1) COVERAGE: for each omitted component (26-connected component of Z = GT* minus GT-),
      the fraction of its voxels covered by each model's prediction, aggregated by volume
      (voxel-weighted) and per case. Does M0 cover more than M1 and M2?
  (2) NEGATIVE-CONTROL FEASIBILITY: how many omitted components are predicted by NONE of
      the three models (within d voxels)? A control that "omits what no model predicts"
      only makes sense if such components are not negligible in size.

Findings (results/omitted_coverage_summary.csv): the negative control is not feasible
(unpredicted components are almost all < 10 voxels); M0 covers about 11 points more of the
omitted volume than M1/M2 in every case.

Outputs: results/omitted_coverage_components.csv (components, strict overlap),
         results/omitted_coverage_summary.csv (size bins x dilation),
         results/omitted_coverage_per_case.csv (volume coverage per case and model).

    python analysis/omitted_coverage.py --workers 4
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

from prgt.folds import case_fold_map

RAW = "nnUNet_data/nnUNet_raw/Dataset100_PARSE"
RES = "nnUNet_data/nnUNet_results/Dataset100_PARSE"
TRAINERS = {"M0": "nnUNetTrainerStd", "M1": "nnUNetTrainerDegradedOmissionOnly",
            "M2": "nnUNetTrainerDriftMu0"}
SIZE_BINS = [(1, 10), (10, 100), (100, 1000), (1000, 10**9)]


def _load(path):
    import nibabel as nib
    return np.asanyarray(nib.load(path).dataobj) > 0


def _case(args):
    from scipy.ndimage import (binary_dilation, find_objects,
                               generate_binary_structure, label)
    case, fold, dilations = args
    conn = generate_binary_structure(3, 3)
    star = _load(f"{RAW}/labelsTr/{case}")
    z = star & ~_load(f"{RAW}/labelsTr_GT_minus_omission/{case}")
    if not z.any():
        return []
    dmax = max(dilations)
    box = tuple(slice(max(s.start - dmax - 2, 0), s.stop + dmax + 2)
                for s in find_objects(z.astype(np.uint8))[0])
    comp, n = label(z[box], structure=conn)
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    preds = {m: _load(f"{RES}/{t}__nnUNetPlans__3d_fullres/fold_{fold}/validation/{case}")[box]
             for m, t in TRAINERS.items()}
    out = {"case": case, "comp": np.arange(1, n + 1), "vox": sizes[1:]}
    for d in dilations:
        hit_any = np.zeros(n + 1, bool)
        for m, p in preds.items():
            pd_ = binary_dilation(p, structure=conn, iterations=d) if d > 0 else p
            cnt = np.bincount(comp[pd_ & (comp > 0)], minlength=n + 1)
            out[f"cover_{m}_d{d}"] = cnt[1:] / sizes[1:]
            hit_any |= cnt > 0
        out[f"none_d{d}"] = ~hit_any[1:]
    return [pd.DataFrame(out)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dilations", type=int, nargs="+", default=[0, 1, 3])
    args = ap.parse_args()

    folds = case_fold_map()
    tasks = [(c, int(f), args.dilations) for c, f in sorted(folds.items())]
    with Pool(args.workers, maxtasksperchild=2) as pool:
        parts = [d for res in pool.map(_case, tasks, chunksize=1) for d in res]
    df = pd.concat(parts, ignore_index=True)

    # components, strict overlap (d = 0)
    keep = ["case", "comp", "vox"] + [f"cover_{m}_d0" for m in TRAINERS] + ["none_d0"]
    df[keep].to_csv("results/omitted_coverage_components.csv", index=False)

    rows = []
    for d in args.dilations:
        for lo, hi in SIZE_BINS:
            s = df[(df.vox >= lo) & (df.vox < hi)]
            per = s[s[f"none_d{d}"]].groupby("case").size().reindex(df.case.unique()).fillna(0)
            rows.append({"dilation_vox": d, "size_lo": lo, "size_hi": hi, "n_components": len(s),
                         "n_voxels": int(s.vox.sum()), "n_unpredicted": int(s[f"none_d{d}"].sum()),
                         "frac_unpredicted": float(s[f"none_d{d}"].mean()),
                         "cases_with_ge1": int((per >= 1).sum()),
                         **{f"vol_coverage_{m}": float(np.average(s[f"cover_{m}_d{d}"], weights=s.vox))
                            for m in TRAINERS}})
    summ = pd.DataFrame(rows)
    summ.to_csv("results/omitted_coverage_summary.csv", index=False)

    # volume coverage per case (d = 0) and the M0 - Mk difference
    for m in TRAINERS:
        df[f"v_{m}"] = df[f"cover_{m}_d0"] * df.vox
    per = df.groupby("case")[["vox"] + [f"v_{m}" for m in TRAINERS]].sum()
    for m in TRAINERS:
        per[f"cov_{m}"] = per[f"v_{m}"] / per.vox
    per[["vox"] + [f"cov_{m}" for m in TRAINERS]].to_csv("results/omitted_coverage_per_case.csv")

    pd.set_option("display.width", 220)
    print(summ.round(3).to_string(index=False))
    for m in ("M1", "M2"):
        d = per.cov_M0 - per[f"cov_{m}"]
        print(f"coverage M0 - {m}: mean {d.mean():+.4f}, M0 > {m} in {(d > 0).sum()}/{len(d)} cases")


if __name__ == "__main__":
    main()
