#!/usr/bin/env python3
"""STRUCTURED omission: does the reversal survive if the omission is not independent?

``distal_omission`` removes each thin structure independently (Bernoulli(p) per component),
whereas an annotator omits in a spatially structured way: they stop at a given generation,
or skip a whole region. This can change where the distant false positives fall, and so HD95.

The candidates are EXACTLY those of the operator (26-connected components of GT* minus
opening(GT*, ball r=2)) and, per case, the omitted volume is the SAME as in the study's
reference (seed 42); only WHICH components are removed changes:

  depth       -- by generation: the depth of a component is the geodesic distance (mm,
                 inside GT*) from the root to its most proximal point; the root is the
                 thickest voxel of GT* (maximum of the inner distance map, pulmonary
                 trunk). The deepest components go first. Deterministic.
  territory_s -- by territory: k-means (K = 16) on the centroids of the candidates; whole
                 territories are removed in a random order (seed s).

Both stop at the cumulative volume closest to the volume omitted by the study. The
study's independent reference (``indep``, read from disk) is the control: its interaction
must reproduce dose_response.csv at p = 0.3.

No new inference: the out-of-fold predictions are read as they are.

Outputs: results/structured_omission_metrics.csv (case x model x reference),
         results/structured_omission_refs.csv (omitted volume, clDice(GT-, GT*) per reference),
         results/structured_omission.csv (interaction per reference and pair),
         results/structured_omission_summary.csv (aggregate over the territory draws).

    python analysis/structured_omission.py --workers 6
    python analysis/structured_omission.py --limit 2 --workers 2   # trial
    python analysis/structured_omission.py --stats-only
"""

from __future__ import annotations

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from dose_response import RAW, RES, TRAINERS
from prgt.folds import case_fold_map
from prgt.metrics import SurfaceFeatures, union_bbox
from prgt.stats import bca_mean_diff, paired_cohens_d, wilcoxon_p_safe

R_OPEN = 2
MARGIN = 8
K_TERRITORIES = 16
OUT = "results/structured_omission"


def _geodesic_depth(star, spacing):
    """Geodesic distance (mm) inside ``star`` from its thickest voxel."""
    from scipy.ndimage import distance_transform_edt
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    idx = np.full(star.shape, -1, np.int64)
    coords = np.argwhere(star)
    idx[star] = np.arange(len(coords))
    rows, cols, w = [], [], []
    sp = np.asarray(spacing, float)
    offsets = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dz, dy, dx) > (0, 0, 0)]                     # half of the 26-neighbourhood
    for off in offsets:
        src = tuple(slice(max(0, -o), s - max(0, o)) for o, s in zip(off, star.shape))
        dst = tuple(slice(max(0, o), s - max(0, -o)) for o, s in zip(off, star.shape))
        a, b = idx[src], idx[dst]
        ok = (a >= 0) & (b >= 0)
        rows.append(a[ok])
        cols.append(b[ok])
        w.append(np.full(int(ok.sum()), float(np.linalg.norm(np.asarray(off) * sp))))
    n = len(coords)
    g = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n, n)).tocsr()
    inner = distance_transform_edt(star, sampling=spacing)
    root = int(idx[np.unravel_index(int(np.argmax(inner)), star.shape)])
    dist = dijkstra(g, directed=False, indices=root)
    depth = np.full(star.shape, np.inf)
    depth[star] = dist
    return depth


def _closest_prefix(order, sizes, target):
    """Shortest prefix of ``order`` whose cumulative volume is closest to ``target``."""
    cum = np.cumsum(sizes[order])
    k = int(np.argmin(np.abs(np.concatenate([[0], cum]) - target)))
    return order[:k]


def _task(args):
    import nibabel as nib
    from scipy.cluster.vq import kmeans2
    from scipy.ndimage import binary_opening, center_of_mass, label, minimum
    from prgt.metrics import compute_cldice, hd95_from_features, nsd_from_features
    from prgt.noise import CONNECTIVITY_26, ball

    case, fold, seeds = args
    nii = nib.load(f"{RAW}/labelsTr/{case}")
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    star_full = np.asanyarray(nii.dataobj) > 0
    indep_full = np.asanyarray(nib.load(f"{RAW}/labelsTr_GT_minus_omission/{case}").dataobj) > 0
    preds_full = {m: np.asanyarray(nib.load(
        f"{RES}/{t}__nnUNetPlans__3d_fullres/fold_{fold}/validation/{case}").dataobj) > 0
        for m, t in TRAINERS.items()}

    bb = union_bbox([star_full], MARGIN)
    star = star_full[bb]
    target = int((star_full & ~indep_full).sum())

    removed = star & ~binary_opening(star, structure=ball(R_OPEN))   # same candidates as the operator
    comp, n = label(removed, structure=CONNECTIVITY_26)
    sizes = np.bincount(comp.ravel(), minlength=n + 1)[1:]
    labels = np.arange(1, n + 1)

    def _ref(chosen):
        out = np.zeros_like(star_full)
        out[bb] = star & ~np.isin(comp, chosen)
        return out

    refs = {"indep": indep_full}
    depth = _geodesic_depth(star, spacing)
    comp_depth = np.asarray(minimum(depth, comp, labels))
    refs["depth"] = _ref(labels[_closest_prefix(np.argsort(-comp_depth, kind="stable"), sizes, target)])

    cent = np.array(center_of_mass(removed, comp, labels)) * np.asarray(spacing)
    k = min(K_TERRITORIES, n)
    for s in seeds:
        rng = np.random.default_rng(s)
        _, terr = kmeans2(cent, k, minit="++", rng=rng)
        terr_ids = rng.permutation(np.unique(terr))
        terr_sizes = np.array([sizes[terr == t].sum() for t in terr_ids])
        kept = _closest_prefix(np.arange(len(terr_ids)), terr_sizes, target)
        refs[f"territory_{s}"] = _ref(labels[np.isin(terr, terr_ids[kept])])

    ref_rows = []
    for name, g in refs.items():
        ref_rows.append({"case": case, "ref": name, "target_vox": target,
                         "omitted_vox": int((star_full & ~g).sum()),
                         "cldice_vs_star": compute_cldice(g[bb], star),
                         "n_components_ref": int(label(g[bb], structure=CONNECTIVITY_26)[1])})

    union = star_full.copy()
    for m in preds_full.values():
        union |= m
    ub = union_bbox([union], 2)
    rows = []
    for m, pred in preds_full.items():
        pf = SurfaceFeatures(pred[ub], spacing)
        for name, g in [("GT_star", star_full)] + list(refs.items()):
            gc = g[ub]
            rows.append({"case": case, "model": m, "ref": name,
                         "hd95": hd95_from_features(pf, gc), "nsd": nsd_from_features(pf, gc, 2.0)})
    return rows, ref_rows


def compute(args):
    folds = case_fold_map()
    cases = [os.path.basename(f) for f in sorted(glob.glob(f"{RAW}/labelsTr/*.nii.gz"))]
    tasks = [(c, int(folds[c]), args.seeds) for c in cases]
    if args.limit:
        tasks = tasks[:args.limit]
    with Pool(args.workers, maxtasksperchild=2) as pool:
        out = pool.map(_task, tasks, chunksize=1)
    df = pd.DataFrame([r for rows, _ in out for r in rows])
    refs = pd.DataFrame([r for _, rr in out for r in rr])
    df.to_csv(f"{OUT}_metrics.csv", index=False)
    refs.to_csv(f"{OUT}_refs.csv", index=False)
    return df


def stats(df, seed):
    rows = []
    refs = [r for r in df.ref.unique() if r != "GT_star"]
    for metric in ("hd95", "nsd"):
        w = df.pivot_table(index=["case", "ref"], columns="model", values=metric).reset_index()
        star = w[w.ref == "GT_star"].set_index("case")
        for other in ("M1_Omission", "M2_Drift_mu0"):
            d_star = star["M0_Star"] - star[other]
            for ref in refs:
                g = w[w.ref == ref].set_index("case")
                d_ref = g["M0_Star"] - g[other]
                iota = (d_ref - d_star).dropna()
                mean, lo, hi = bca_mean_diff(iota.values, seed=seed)
                rows.append({"metric": metric, "pair": f"M0_vs_{other}", "ref": ref,
                             "delta_star": d_star.mean(), "delta_ref": d_ref.mean(),
                             "sign_flip": bool(np.sign(d_star.mean()) != np.sign(d_ref.mean())),
                             "interaction_mean": mean, "ci_low": lo, "ci_high": hi,
                             "d": paired_cohens_d(iota.values),
                             "p_raw": wilcoxon_p_safe(d_ref.loc[iota.index].values,
                                                      d_star.loc[iota.index].values),
                             "n": len(iota)})
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}.csv", index=False)

    out["variant"] = out.ref.str.replace(r"_\d+$", "", regex=True)
    summ = []
    for (metric, pair, variant), g in out.groupby(["metric", "pair", "variant"]):
        summ.append({"metric": metric, "pair": pair, "variant": variant, "n_refs": len(g),
                     "n_sign_flip": int(g.sign_flip.sum()),
                     "interaction_mean": g.interaction_mean.mean(),
                     "interaction_sd": g.interaction_mean.std(ddof=1) if len(g) > 1 else np.nan,
                     "interaction_min": g.interaction_mean.min(),
                     "interaction_max": g.interaction_mean.max(),
                     "d_min": g.d.min(), "d_max": g.d.max(), "p_raw_max": g.p_raw.max()})
    summ = pd.DataFrame(summ)
    summ.to_csv(f"{OUT}_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print(summ.round(4).to_string(index=False))
    if os.path.exists(f"{OUT}_refs.csv"):
        r = pd.read_csv(f"{OUT}_refs.csv")
        r["variant"] = r.ref.str.replace(r"_\d+$", "", regex=True)
        r["vol_ratio"] = r.omitted_vox / r.target_vox
        print(r.groupby("variant")[["vol_ratio", "cldice_vs_star", "n_components_ref"]]
              .agg(["mean", "min", "max"]).round(3).to_string())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(11)))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42, help="seed of the BCa bootstrap")
    ap.add_argument("--stats-only", action="store_true", help="re-read the per-case CSV")
    ap.add_argument("--limit", type=int, default=0, help="only evaluate the first N cases (trial)")
    args = ap.parse_args()
    df = pd.read_csv(f"{OUT}_metrics.csv") if args.stats_only else compute(args)
    stats(df, args.seed)


if __name__ == "__main__":
    main()
