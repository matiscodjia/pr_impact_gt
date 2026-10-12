#!/usr/bin/env python3
"""Omission STRUCTURÉE : le renversement survit-il si l'omission n'est pas indépendante ?

Question (relecture simulée, Q1) : ``distal_omission`` retire chaque structure fine
indépendamment (Bernoulli(p) par composante), alors qu'un annotateur omet de façon
spatialement structurée -- il s'arrête à une génération, ou saute une région entière.
Cela peut changer la répartition des faux positifs lointains, donc HD95.

On garde EXACTEMENT les mêmes candidates que l'opérateur (composantes 26-connexes de
GT* \\ ouverture(GT*, boule r=2)) et, par cas, le MÊME volume omis que la référence de
l'étude (graine 42) ; seul change QUELLES composantes sont retirées :

  depth       -- par génération : profondeur d'une composante = distance géodésique
                 (mm, dans GT*) de la racine à son point le plus proximal ; racine = voxel
                 le plus épais de GT* (maximum de la carte de distance intérieure, tronc
                 de l'AP). On retire les plus profondes d'abord. Déterministe.
  territory_s -- par territoire : k-means (K=16) sur les centroïdes des candidates ;
                 on retire des territoires entiers dans un ordre aléatoire (graine s).

Dans les deux cas on s'arrête au volume cumulé le plus proche du volume omis par
l'étude. La référence indépendante de l'étude (``indep``, relue sur disque) sert de
témoin : son interaction doit redonner dose_response.csv à p=0.3.

Aucune ré-inférence : prédictions OOF lues telles quelles.

Sorties : results/structured_omission_metrics.csv (cas x modèle x référence),
          results/structured_omission_refs.csv (volume omis, clDice(GT-, GT*) par référence),
          results/structured_omission.csv (interaction par référence et par paire),
          results/structured_omission_summary.csv (agrégat des tirages territory).

Usage
-----
    python analysis/structured_omission.py --workers 6
    python analysis/structured_omission.py --limit 2 --workers 2   # essai
    python analysis/structured_omission.py --stats-only
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

from dose_response import RAW, RES, TRAINERS, _bbox, _Feat  # noqa: E402
from fold_map import case_fold_map  # noqa: E402
from stats_utils import bca_mean_diff, paired_cohens_d, wilcoxon_p_safe  # noqa: E402

R_OPEN = 2
MARGIN = 8
K_TERRITORIES = 16
OUT = "results/structured_omission"


def _geodesic_depth(star, spacing):
    """Distance géodésique (mm) dans ``star`` depuis son voxel le plus épais."""
    from scipy.ndimage import distance_transform_edt
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    idx = np.full(star.shape, -1, np.int64)
    coords = np.argwhere(star)
    idx[star] = np.arange(len(coords))
    rows, cols, w = [], [], []
    sp = np.asarray(spacing, float)
    offsets = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dz, dy, dx) > (0, 0, 0)]                     # demi-voisinage 26-connexe
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
    """Plus court préfixe de ``order`` dont le volume cumulé est le plus proche de ``target``."""
    cum = np.cumsum(sizes[order])
    k = int(np.argmin(np.abs(np.concatenate([[0], cum]) - target)))
    return order[:k]


def _task(args):
    import nibabel as nib
    from scipy.cluster.vq import kmeans2
    from scipy.ndimage import binary_opening, center_of_mass, label, minimum
    from cross_evaluate import _hd95_cached, _nsd_cached, compute_cldice
    from degradations import _CONN3, _ball

    case, fold, seeds = args
    nii = nib.load(f"{RAW}/labelsTr/{case}")
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    star_full = np.asanyarray(nii.dataobj) > 0
    indep_full = np.asanyarray(nib.load(f"{RAW}/labelsTr_GT_minus_omission/{case}").dataobj) > 0
    preds_full = {m: np.asanyarray(nib.load(
        f"{RES}/{t}__nnUNetPlans__3d_fullres/fold_{fold}/validation/{case}").dataobj) > 0
        for m, t in TRAINERS.items()}

    bb = _bbox(star_full, MARGIN)
    star = star_full[bb]
    target = int((star_full & ~indep_full).sum())

    removed = star & ~binary_opening(star, structure=_ball(R_OPEN))   # mêmes candidates que l'opérateur
    comp, n = label(removed, structure=_CONN3)
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
                         "n_components_ref": int(label(g[bb], structure=_CONN3)[1])})

    union = star_full.copy()
    for m in preds_full.values():
        union |= m
    ub = _bbox(union, 2)
    rows = []
    for m, pred in preds_full.items():
        pf = _Feat(pred[ub], spacing)
        for name, g in [("GT_star", star_full)] + list(refs.items()):
            gc = g[ub]
            rows.append({"case": case, "model": m, "ref": name,
                         "hd95": _hd95_cached(pf, gc), "nsd": _nsd_cached(pf, gc, 2.0)})
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
    ap.add_argument("--seed", type=int, default=42, help="graine du bootstrap BCa")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="n'évaluer que les N premiers cas (essai)")
    args = ap.parse_args()
    df = pd.read_csv(f"{OUT}_metrics.csv") if args.stats_only else compute(args)
    stats(df, args.seed)


if __name__ == "__main__":
    main()
