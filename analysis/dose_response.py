#!/usr/bin/env python3
"""Dose-réponse en p (taux d'omission) du renversement HD95 / NSD@2mm.

Question (critique R3 de la relecture) : le mécanisme « M0 prédit des branches que la
référence a retirées » prédit que l'interaction

    iota(p) = (m(M0|GT-_p) - m(M_k|GT-_p)) - (m(M0|GT*) - m(M_k|GT*))

croît avec p. On régénère GT- avec la MÊME graine par cas que la référence de l'étude
(seed_base = 42 + index du cas dans l'ordre trié de labelsTr) et on change seulement p.
La sélection étant ``rng.random(n) < p``, les omissions sont EMBOÎTÉES quand p croît :
GT-(0.5) ⊂ GT-(0.3) ⊂ GT-(0.1) -- vérifié dans ce script (``nested_ok``).

Validation : le p=0.3 régénéré doit être identique voxel à voxel au jeu déjà sur disque
(``labelsTr_GT_minus_omission``) ; sinon la convention d'index/graine est fausse et les
autres p ne sont pas comparables (``matches_stored``). Le calcul est fait sur la boîte
englobante de GT* (+ marge) : exact, car l'ouverture (boule r) et l'étiquetage des
composantes ne dépendent pas du fond vide.

Aucune ré-inférence : les prédictions OOF sont lues telles quelles.

Sorties : results/dose_response<tag>_metrics.csv (par cas x modèle x référence),
          results/dose_response<tag>.csv (interaction par p et par paire),
          results/dose_response<tag>_checks.csv (validations par cas).

Tirages multiples de GT- (variance de la réalisation du bruit) : ``--seed S --tag _drawS``
avec une graine de base qui change (S = 42 est l'étude ; utiliser 1042, 2042... : les
graines par cas valent S + index, un pas de 1000 évite tout recouvrement entre tirages).

Usage
-----
    python analysis/dose_response.py --workers 4
    python analysis/dose_response.py --stats-only      # ré-agrège sans recalculer
    python analysis/dose_response.py --ps 0.3 --seed 1042 --tag _draw1042 --workers 32
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

from stats_utils import bca_mean_diff, paired_cohens_d, wilcoxon_p_safe  # noqa: E402
from fold_map import case_fold_map  # noqa: E402

RAW = "nnUNet_data/nnUNet_raw/Dataset100_PARSE"
RES = "nnUNet_data/nnUNet_results/Dataset100_PARSE"
TRAINERS = {"M0_Star": "nnUNetTrainerStd",
            "M1_Omission": "nnUNetTrainerDegradedOmissionOnly",
            "M2_Drift_mu0": "nnUNetTrainerDriftMu0"}
R_OPEN = 2
MARGIN = 8            # >= r pour l'ouverture, + marge de surface
STORED_P = 0.3


class _Feat:
    """Sous-ensemble léger de PredFeatures (sans squelette : inutile pour HD95/NSD)."""

    def __init__(self, pred, spacing):
        from scipy.ndimage import binary_erosion, distance_transform_edt
        self.spacing = spacing
        self.bool = pred
        self.any = bool(pred.any())
        self.dt = distance_transform_edt(~pred, sampling=spacing)
        self.surf = pred ^ binary_erosion(pred, structure=np.ones((3, 3, 3)))
        self.dt_from_surf = distance_transform_edt(~self.surf, sampling=spacing)


def _bbox(mask, margin):
    sl = []
    for ax in range(3):
        nz = np.where(mask.any(axis=tuple(i for i in range(3) if i != ax)))[0]
        sl.append(slice(max(0, int(nz[0]) - margin), int(nz[-1]) + 1 + margin))
    return tuple(sl)


def _task(args):
    import nibabel as nib
    from cross_evaluate import _hd95_cached, _nsd_cached
    from degradations import apply_degradation_pipeline

    idx, case, fold, ps, seed = args
    nii = nib.load(f"{RAW}/labelsTr/{case}")
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    star_full = np.asanyarray(nii.dataobj) > 0
    stored_path = f"{RAW}/labelsTr_GT_minus_omission/{case}"      # absent => pas de validation croisée
    stored = np.asanyarray(nib.load(stored_path).dataobj) > 0 if os.path.exists(stored_path) else None
    preds_full = {m: np.asanyarray(nib.load(
        f"{RES}/{t}__nnUNetPlans__3d_fullres/fold_{fold}/validation/{case}").dataobj) > 0
        for m, t in TRAINERS.items()}

    # boîte de GT* pour dégrader ; boîte union (avec prédictions) pour évaluer
    bb = _bbox(star_full, MARGIN)
    star = star_full[bb]
    refs = {}
    for p in ps:
        deg = apply_degradation_pipeline(
            star[np.newaxis].astype(np.float32),
            [{"family": "distal_omission", "r": R_OPEN, "p": float(p)}],
            seed_base=seed + idx, spacing=spacing)[0] > 0
        full = np.zeros_like(star_full)
        full[bb] = deg
        refs[p] = full

    info = {"case": case, "matches_stored": bool(np.array_equal(refs[STORED_P], stored))
            if (STORED_P in refs and stored is not None) else None}
    ordered = sorted(ps)
    info["nested_ok"] = all(not (refs[b] & ~refs[a]).any()          # GT-(b) ⊂ GT-(a) pour b>a
                            for a, b in zip(ordered[:-1], ordered[1:]))
    for p in ps:
        info[f"omitted_vox_p{p}"] = int((star_full & ~refs[p]).sum())

    union = star_full.copy()
    for m in preds_full.values():
        union |= m
    ub = _bbox(union, 2)
    star_c = star_full[ub]
    refs_c = {p: refs[p][ub] for p in ps}

    rows = []
    for m, pred in preds_full.items():
        pf = _Feat(pred[ub], spacing)
        for name, g in [("GT_star", star_c)] + [(f"p{p}", refs_c[p]) for p in ps]:
            rows.append({"case": case, "model": m, "ref": name,
                         "hd95": _hd95_cached(pf, g), "nsd": _nsd_cached(pf, g, 2.0)})
    return rows, info


def _path(args, stem):
    return f"results/dose_response{args.tag}{stem}.csv"


def compute(args):
    folds = case_fold_map()
    cases = [os.path.basename(f) for f in sorted(glob.glob(f"{RAW}/labelsTr/*.nii.gz"))]
    tasks = [(i, c, int(folds[c]), args.ps, args.seed) for i, c in enumerate(cases)]
    if args.limit:                       # essai rapide : les N premiers cas seulement
        tasks = tasks[:args.limit]
    with Pool(args.workers, maxtasksperchild=2) as pool:
        out = pool.map(_task, tasks, chunksize=1)
    df = pd.DataFrame([r for rows, _ in out for r in rows])
    info = pd.DataFrame([i for _, i in out])
    df.to_csv(_path(args, "_metrics"), index=False)
    info.to_csv(_path(args, "_checks"), index=False)
    if args.seed == 42 and info.matches_stored.notna().any():   # réf. stockée = graine 42
        print(f"validation p=0.3 == référence stockée : {int(info.matches_stored.sum())}/{len(info)} cas")
    print(f"omissions emboîtées : {int(info.nested_ok.sum())}/{len(info)} cas")
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
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--tag", default="", help="suffixe des fichiers de sortie, ex. _draw1042")
    ap.add_argument("--limit", type=int, default=0, help="n'évaluer que les N premiers cas (essai)")
    args = ap.parse_args()
    df = (pd.read_csv(_path(args, "_metrics")) if args.stats_only else compute(args))
    stats(df, args.seed, args)


if __name__ == "__main__":
    main()
