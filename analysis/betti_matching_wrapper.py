#!/usr/bin/env python3
"""Wrapper P1.1 -- Betti-matching (Stucki et al., github.com/nstucki/Betti-Matching-3D).

Convention vérifiée avant tout usage (README du dépôt) : le "Betti matching error" est
une ERREUR, pas une similarité -- 0 = correspondance topologique parfaite, plus petit
est meilleur, même sens que l'ancien |ΔN| qu'il raffine ("a refinement of the
well-established Betti number error, by counting the features in both images that do
not spatially correspond to a feature in the other image"). Direction = "lower",
cohérent avec `analysis/rank_reversal.py::DIRECTION["betti0"]`.

Filtration : la librairie opère sur des volumes à valeurs réelles (persistance sur
complexe cubique, filtration par sous-niveaux) -- des masques binaires bruts sont
dégénérés pour cet usage (toutes les composantes réelles deviennent des intervalles
"essentiels", exclus par défaut). On utilise donc la transformée de distance SIGNÉE de
chaque masque (négative à l'intérieur du foreground, positive à l'extérieur) : à
l'ensemble de sous-niveau τ correspond exactement le foreground érodé/dilaté de |τ|,
et le seuil τ=0 redonne le masque original. Une composante distale isolée dans le
foreground "meurt" (fusionne avec le reste) exactement à la distance qui la sépare du
reste de la structure -- la persistance encode directement une information de distance,
pas seulement un décompte, ce qui est le point même de préférer Betti-matching à |ΔN|.

Coût : la lib parallélise nativement un batch via `compute_matching(list, list)`
(std::async) -- à utiliser pour scanner plusieurs cas plutôt que budgétiser un
`multiprocessing.Pool` par-dessus (double parallélisme = contention).

Usage
-----
    from betti_matching import betti_matching_error
    d = betti_matching_error(pred_path, ref_path)  # dict: bm_error_dim0, n_matched_dim0, ...
"""

from __future__ import annotations

import os
import sys
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import block_reduce

_BM_BUILD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "external",
    "Betti-Matching-3D", "build",
)
sys.path.insert(0, os.path.abspath(_BM_BUILD_DIR))
import betti_matching as _bm  # noqa: E402


def signed_distance_transform(mask: np.ndarray) -> np.ndarray:
    """SDT : négative dans le foreground, positive dans le fond, 0 sur le bord."""
    mask = mask.astype(bool)
    if not mask.any():
        return distance_transform_edt(~mask).astype(np.float64)
    if mask.all():
        return -distance_transform_edt(mask).astype(np.float64)
    return (distance_transform_edt(~mask) - distance_transform_edt(mask)).astype(np.float64)


def betti_matching_error(pred: np.ndarray, ref: np.ndarray, dim: int = 0,
                         persistence_threshold: float = 2.0) -> dict:
    """Betti-matching error (dimension `dim`, defaut 0 = composantes connexes).

    `pred`/`ref` : masques booléens (mêmes conventions que `evaluate_pair_cached` --
    `pred` joue le rôle "input1"/prédiction, `ref` le rôle "input2"/référence).

    IMPORTANT (découvert lors du test de coût, pas documenté dans le README) : la
    persistance homologie cubique sur la SDT d'un masque voxelisé produit une MASSE de
    paires "unmatched" de persistance quasi nulle (médiane observée ~0.38 voxel) --
    du bruit de discrétisation de la surface (rugosité voxel par voxel), pas des
    composantes anatomiques. Sur un cas réel (PARSE_0027, M0 vs GT*) : 1232 non-appariés
    bruts côté prédiction, dont seulement **45** ont une persistance > 2 voxels ; côté
    référence (GT* propre, N=1 attendu), 1747 bruts mais **0** au-delà de 2 voxels --
    cohérent avec la référence propre étant réellement mono-composante. Un seuil de
    persistance est donc OBLIGATOIRE, pas optionnel, pour que le compte ait un sens
    anatomique. `persistence_threshold=2.0` (voxels) est le défaut retenu : il ramène le
    compte prédiction (45) dans l'ordre de grandeur de N(P)=44-81 déjà mesuré dans le
    rapport (Table tab:betti) avec l'ancien |ΔN|, et ramène le compte référence à 0,
    cohérent avec N(GT*)=1. À valider/ajuster si les résultats sur l'échantillon de 20
    cas montrent un désaccord systématique.
    """
    sdt_pred = signed_distance_transform(pred)
    sdt_ref = signed_distance_transform(ref)
    result = _bm.compute_matching(sdt_pred, sdt_ref)

    n_unmatched_pred_raw = int(result.num_unmatched_input1[dim])
    n_unmatched_ref_raw = int(result.num_unmatched_input2[dim])
    n_matched = int(result.num_matched[dim])

    bb1 = np.asarray(result.input1_unmatched_birth_coordinates[dim])
    bd1 = np.asarray(result.input1_unmatched_death_coordinates[dim])
    bb2 = np.asarray(result.input2_unmatched_birth_coordinates[dim])
    bd2 = np.asarray(result.input2_unmatched_death_coordinates[dim])
    pers_pred = (sdt_pred[tuple(bd1.T)] - sdt_pred[tuple(bb1.T)]) if len(bb1) else np.array([])
    pers_ref = (sdt_ref[tuple(bd2.T)] - sdt_ref[tuple(bb2.T)]) if len(bb2) else np.array([])

    n_unmatched_pred = int((pers_pred > persistence_threshold).sum())
    n_unmatched_ref = int((pers_ref > persistence_threshold).sum())

    return {
        "bm_error": n_unmatched_pred + n_unmatched_ref,  # 0 = correspondance parfaite
        "bm_unmatched_pred": n_unmatched_pred,
        "bm_unmatched_ref": n_unmatched_ref,
        "bm_n_matched": n_matched,
        "bm_error_raw_unfiltered": n_unmatched_pred_raw + n_unmatched_ref_raw,
        "bm_persistence_threshold": persistence_threshold,
    }


def _union_bbox(masks: list[np.ndarray], margin: int) -> tuple:
    any_fg = None
    for m in masks:
        any_fg = m if any_fg is None else (any_fg | m)
    if any_fg is None or not any_fg.any():
        return tuple(slice(0, s) for s in masks[0].shape)
    sl = []
    for ax in range(any_fg.ndim):
        idx = np.any(any_fg, axis=tuple(i for i in range(any_fg.ndim) if i != ax))
        nz = np.where(idx)[0]
        lo = max(0, int(nz[0]) - margin)
        hi = min(any_fg.shape[ax], int(nz[-1]) + 1 + margin)
        sl.append(slice(lo, hi))
    return tuple(sl)


def betti_matching_error_from_paths(pred_path: str, ref_path: str, dim: int = 0,
                                    crop_margin: int | None = 15,
                                    persistence_threshold: float = 2.0,
                                    downsample: int = 1) -> dict:
    """`crop_margin` : marge (voxels) autour de l'union pred/ref avant calcul (None =
    pleine résolution, testé >20min/cas à 512x512x300, inutilisable). Le crop ne peut
    PAS tronquer le foreground (union pred+ref inclus en entier par construction, comme
    pour les autres métriques) ; seule la marge de fond change, et les fusions de
    composantes qui nous intéressent (distales, quelques dizaines de voxels au plus) se
    produisent bien en-deçà d'une marge de 15 -- documenté comme approximation
    justifiée, pas une exactitude à la précision machine comme pour le crop HD95/NSD.

    `downsample` : facteur de max-pooling appliqué APRÈS le crop, avant la SDT (même
    convention que `visualize_degradations.py`, `block_reduce(..., np.max)` -- préserve
    les structures fines, une brique de voxels est foreground si au moins un l'est).
    NÉCESSAIRE en pratique, pas une simple option de confort : un seul appel
    `compute_matching` sur un volume croppé à ~(395,287,286) mesure un pic mémoire de
    **13.8 Go** (`/usr/bin/time -l`, un seul thread) -- sur une machine à 24 Go de RAM,
    ça interdit toute parallélisation (déjà provoqué un OOM kill à 3 workers) et rend
    même le run séquentiel des 85 cas risqué. `downsample=2` réduit le nombre de voxels
    d'un facteur ~8 ; à valider par le test de cohérence sur l'échantillon de 20 cas
    (le seuil de persistance, en voxels *downsamplés*, doit probablement être réduit en
    conséquence -- une distance de 2 voxels downsamplés correspond à ~4 voxels natifs)."""
    # dataobj préserve le dtype disque (uint8) -- get_fdata() upcaste en float64 (8x la
    # mémoire) avant même le crop ; avec plusieurs workers concurrents sur des volumes
    # 512x512x300, c'est ce qui a déclenché un OOM kill lors du premier test à 3 workers.
    pred = np.asarray(nib.load(pred_path).dataobj) > 0
    ref = np.asarray(nib.load(ref_path).dataobj) > 0
    if crop_margin is not None:
        bbox = _union_bbox([pred, ref], margin=crop_margin)
        pred, ref = pred[bbox], ref[bbox]
    if downsample > 1:
        bs = (downsample, downsample, downsample)
        pred = block_reduce(pred, bs, np.max)
        ref = block_reduce(ref, bs, np.max)
    t0 = time.time()
    out = betti_matching_error(pred, ref, dim=dim, persistence_threshold=persistence_threshold)
    out["_seconds"] = time.time() - t0
    out["_shape"] = pred.shape
    out["_downsample"] = downsample
    return out
