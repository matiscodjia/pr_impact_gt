#!/usr/bin/env python3
"""Couverture des régions omises par les prédictions (prémisse du mécanisme, critique R3).

Le mécanisme proposé pour le renversement sous omission : M0 (entraîné sur GT*) prédit
des structures fines que la référence GT- (omission) ne contient plus. Deux mesures
directes, sans ré-inférence :

  (1) COUVERTURE : pour chaque composante omise (composante connexe 26-voisins de
      Z = GT* \\ GT-), la part de ses voxels recouverts par la prédiction de chaque modèle,
      agrégée en volume (pondérée par voxels) et par cas. Test : M0 en couvre-t-il plus ?
  (2) CONTRÔLE NÉGATIF (faisabilité) : combien de composantes omises ne sont prédites par
      AUCUN des trois modèles (à d voxels près) ? Un contrôle « omettre ce qu'aucun modèle
      ne prédit » n'a de sens que s'il en reste de taille non négligeable.

Résultats (voir results/omitted_coverage_summary.csv) : le contrôle négatif est
inexploitable (les composantes non prédites sont quasi toutes < 10 voxels) ; la couverture
de M0 dépasse celle de M1/M2 d'environ 11 points dans chaque cas.

Sorties : results/omitted_coverage_components.csv (composantes, recouvrement strict),
          results/omitted_coverage_summary.csv (strates de taille x dilatation),
          results/omitted_coverage_per_case.csv (couverture volumique par cas et modèle).

Usage
-----
    python analysis/omitted_coverage.py --workers 4
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fold_map import case_fold_map  # noqa: E402

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

    # composantes, recouvrement strict (d=0)
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

    # couverture volumique par cas (d=0) et différence M0 - Mk
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
        print(f"couverture M0 - {m}: moyenne {d.mean():+.4f}, M0 > {m} dans {(d > 0).sum()}/{len(d)} cas")


if __name__ == "__main__":
    main()
