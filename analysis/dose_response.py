#!/usr/bin/env python3
"""Dose-response of the HD95 / NSD@2mm reversal in the omission rate p.

The mechanism "M0 predicts branches that the reference removed" predicts that the
interaction

    iota(p) = (m(M0|GT-_p) - m(M_k|GT-_p)) - (m(M0|GT*) - m(M_k|GT*))

grows with p. GT- is regenerated with the SAME per-case seed as the study's reference
(seed_base = 42 + index of the case in the sorted labelsTr) and only p changes. Since the
selection is ``rng.random(n) < p``, the omissions are NESTED as p grows:
GT-(0.5) is contained in GT-(0.3), itself in GT-(0.1) -- checked here (``nested_ok``).

Validation: the regenerated p = 0.3 must equal, voxel for voxel, the reference already on
disk (labelsTr_GT_minus_omission); otherwise the index/seed convention is wrong and the
other p are not comparable (``matches_stored``). The noise is computed on the bounding box
of GT* (+ margin): exact, because the opening (ball r) and the component labelling do not
depend on empty background.

No new inference: the out-of-fold predictions are read as they are.

Outputs: results/dose_response<tag>_metrics.csv (per case x model x reference),
         results/dose_response<tag>.csv (interaction per p and per pair),
         results/dose_response<tag>_checks.csv (per-case validations).

Several draws of GT- (variance of the noise realisation): ``--seed S --tag _drawS`` with a
different base seed (S = 42 is the study; use 1042, 2042, ...: per-case seeds are
S + index, so a step of 1000 avoids any overlap between draws).

    python analysis/dose_response.py --workers 4
    python analysis/dose_response.py --stats-only      # re-aggregate without recomputing
    python analysis/dose_response.py --ps 0.3 --seed 1042 --tag _draw1042 --workers 32
    python analysis/dose_response.py --ps 0.1 0.15 0.2 0.25 0.3 --tag _fine --ref-cldice
"""

from __future__ import annotations

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from prgt.folds import case_fold_map
from prgt.metrics import SurfaceFeatures, union_bbox
from prgt.stats import bca_mean_diff, paired_cohens_d, wilcoxon_p_safe

RAW = "nnUNet_data/nnUNet_raw/Dataset100_PARSE"
RES = "nnUNet_data/nnUNet_results/Dataset100_PARSE"
TRAINERS = {"M0_Star": "nnUNetTrainerStd",
            "M1_Omission": "nnUNetTrainerDegradedOmissionOnly",
            "M2_Drift_mu0": "nnUNetTrainerDriftMu0"}
R_OPEN = 2
MARGIN = 8            # >= r of the opening, plus a surface margin
STORED_P = 0.3


def _task(args):
    import nibabel as nib
    from prgt.metrics import compute_cldice, hd95_from_features, nsd_from_features
    from prgt.noise import apply_noise_configs

    idx, case, fold, ps, seed, ref_cldice = args
    nii = nib.load(f"{RAW}/labelsTr/{case}")
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    star_full = np.asanyarray(nii.dataobj) > 0
    stored_path = f"{RAW}/labelsTr_GT_minus_omission/{case}"      # missing => no cross-check
    stored = np.asanyarray(nib.load(stored_path).dataobj) > 0 if os.path.exists(stored_path) else None
    preds_full = {m: np.asanyarray(nib.load(
        f"{RES}/{t}__nnUNetPlans__3d_fullres/fold_{fold}/validation/{case}").dataobj) > 0
        for m, t in TRAINERS.items()}

    # GT* box to apply the noise; union box (with the predictions) to score
    bb = union_bbox([star_full], MARGIN)
    star = star_full[bb]
    refs = {}
    for p in ps:
        deg = apply_noise_configs(
            star[np.newaxis].astype(np.float32),
            [{"family": "distal_omission", "r": R_OPEN, "p": float(p)}],
            seed_base=seed + idx, spacing=spacing)[0] > 0
        full = np.zeros_like(star_full)
        full[bb] = deg
        refs[p] = full

    info = {"case": case, "matches_stored": bool(np.array_equal(refs[STORED_P], stored))
            if (STORED_P in refs and stored is not None) else None}
    ordered = sorted(ps)
    info["nested_ok"] = all(not (refs[b] & ~refs[a]).any()          # GT-(b) within GT-(a) for b > a
                            for a, b in zip(ordered[:-1], ordered[1:]))
    for p in ps:
        info[f"omitted_vox_p{p}"] = int((star_full & ~refs[p]).sum())
        if ref_cldice:                   # severity of the reference itself (window [0.85, 0.90])
            info[f"cldice_ref_p{p}"] = compute_cldice(refs[p][bb], star)

    union = star_full.copy()
    for m in preds_full.values():
        union |= m
    ub = union_bbox([union], 2)
    star_c = star_full[ub]
    refs_c = {p: refs[p][ub] for p in ps}

    rows = []
    for m, pred in preds_full.items():
        pf = SurfaceFeatures(pred[ub], spacing)
        for name, g in [("GT_star", star_c)] + [(f"p{p}", refs_c[p]) for p in ps]:
            rows.append({"case": case, "model": m, "ref": name,
                         "hd95": hd95_from_features(pf, g), "nsd": nsd_from_features(pf, g, 2.0)})
    return rows, info


def _path(args, stem):
    return f"results/dose_response{args.tag}{stem}.csv"


def compute(args):
    folds = case_fold_map()
    cases = [os.path.basename(f) for f in sorted(glob.glob(f"{RAW}/labelsTr/*.nii.gz"))]
    tasks = [(i, c, int(folds[c]), args.ps, args.seed, args.ref_cldice) for i, c in enumerate(cases)]
    if args.limit:                       # quick trial: first N cases only
        tasks = tasks[:args.limit]
    with Pool(args.workers, maxtasksperchild=2) as pool:
        out = pool.map(_task, tasks, chunksize=1)
    df = pd.DataFrame([r for rows, _ in out for r in rows])
    info = pd.DataFrame([i for _, i in out])
    df.to_csv(_path(args, "_metrics"), index=False)
    info.to_csv(_path(args, "_checks"), index=False)
    if args.seed == 42 and info.matches_stored.notna().any():   # the stored reference uses seed 42
        print(f"regenerated p=0.3 == stored reference: {int(info.matches_stored.sum())}/{len(info)} cases")
    print(f"nested omissions: {int(info.nested_ok.sum())}/{len(info)} cases")
    if args.ref_cldice:
        for p in args.ps:
            c = info[f"cldice_ref_p{p}"]
            print(f"clDice(GT-_p{p}, GT*) = {c.mean():.4f}  ({(c.between(0.85, 0.90)).mean():.0%} of cases in [0.85, 0.90])")
    return df


def stats(df, seed, args):
    rows = []
    refs = [r for r in df.ref.unique() if r != "GT_star"]
    for metric in ("hd95", "nsd"):
        w = df.pivot_table(index=["case", "ref"], columns="model", values=metric).reset_index()
        star = w[w.ref == "GT_star"].set_index("case")
        for other in ("M1_Omission", "M2_Drift_mu0"):
            d_star = star["M0_Star"] - star[other]
            for ref in sorted(refs, key=lambda r: float(r[1:])):
                g = w[w.ref == ref].set_index("case")
                d_ref = g["M0_Star"] - g[other]
                iota = (d_ref - d_star).dropna()
                mean, lo, hi = bca_mean_diff(iota.values, seed=seed)
                rows.append({"metric": metric, "pair": f"M0_vs_{other}", "p_omission": float(ref[1:]),
                             "delta_star": d_star.mean(), "delta_ref": d_ref.mean(),
                             "sign_flip": bool(np.sign(d_star.mean()) != np.sign(d_ref.mean())),
                             "interaction_mean": mean, "ci_low": lo, "ci_high": hi,
                             "d": paired_cohens_d(iota.values),
                             "p_raw": wilcoxon_p_safe(d_ref.loc[iota.index].values,
                                                      d_star.loc[iota.index].values),
                             "n": len(iota)})
    out = pd.DataFrame(rows)
    out.to_csv(_path(args, ""), index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stats-only", action="store_true", help="re-read the per-case CSV")
    ap.add_argument("--tag", default="", help="suffix of the output files, e.g. _draw1042")
    ap.add_argument("--limit", type=int, default=0, help="only evaluate the first N cases (trial)")
    ap.add_argument("--ref-cldice", action="store_true",
                    help="also measure clDice(GT-_p, GT*) per case (in *_checks.csv)")
    args = ap.parse_args()
    df = (pd.read_csv(_path(args, "_metrics")) if args.stats_only else compute(args))
    stats(df, args.seed, args)


if __name__ == "__main__":
    main()
