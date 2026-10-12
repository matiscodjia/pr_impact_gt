"""Per-case segmentation metrics for thin tubular structures (pulmonary arteries).

All metrics compare a prediction with one reference mask of the same case and geometry.
Masks are thresholded at 0.5; distances are in millimetres (``spacing`` = voxel size).

    clDice        centreline Dice (topology-aware overlap), higher is better
    HD95          95th percentile of the symmetric voxel-to-mask distances (mm), lower is better
    NSD@tau       normalised surface Dice at tolerance tau mm (tau = 2 and 0.5), higher is better
    Betti-0 error |#components(pred) - #components(ref)|, 26-connectivity, lower is better
    volume delta  signed relative volume difference (|pred| - |ref|) / |ref|

Two ways to compute them:

* ``compute_*`` functions: one prediction, one reference, everything computed from scratch.
* ``PredictionFeatures`` + ``score_prediction``: when the same prediction is scored against
  several references (GT* and each GT-), the prediction-only quantities (skeleton, distance
  maps, surface, components) are computed once. The results are numerically identical to
  the ``compute_*`` functions and the scoring time is roughly halved.

``union_bbox`` crops a case to the bounding box of its foreground before scoring. The crop
is exact for every metric here: none of them depends on empty background beyond a margin
of one voxel, and the arteries fill ~0.3 % of a 512^3 volume.
"""

import numpy as np

# 26-connectivity for connected components. The scipy default (6-connectivity, faces only)
# splits a thin vascular tree into dozens or hundreds of spurious components.
_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)


# ---------------------------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------------------------

def union_bbox(masks, margin=2):
    """Bounding box (with ``margin`` voxels) of the union of boolean masks, as a slice tuple.

    Returns the full volume if every mask is empty.
    """
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


# ---------------------------------------------------------------------------------------------
# Stand-alone metrics
# ---------------------------------------------------------------------------------------------

def compute_cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    from skimage.morphology import skeletonize

    pred_bool = pred > 0.5
    gt_bool = gt > 0.5

    if not pred_bool.any() and not gt_bool.any():
        return 1.0
    if not pred_bool.any() or not gt_bool.any():
        return 0.0

    skel_pred = skeletonize(pred_bool)
    skel_gt = skeletonize(gt_bool)

    # Topological precision: fraction of the predicted skeleton inside the reference volume.
    tprec = np.logical_and(skel_pred, gt_bool).sum() / max(skel_pred.sum(), 1)
    # Topological sensitivity: fraction of the reference skeleton inside the predicted volume.
    tsens = np.logical_and(skel_gt, pred_bool).sum() / max(skel_gt.sum(), 1)

    if tprec + tsens == 0:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))


def compute_nsd(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1, 1, 1), tolerance: float = 2.0) -> float:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    pred_bool = pred > 0.5
    gt_bool = gt > 0.5

    if not pred_bool.any() and not gt_bool.any():
        return 1.0
    if not pred_bool.any() or not gt_bool.any():
        return 0.0

    struct = np.ones((3, 3, 3))
    surf_pred = pred_bool ^ binary_erosion(pred_bool, structure=struct)
    surf_gt = gt_bool ^ binary_erosion(gt_bool, structure=struct)

    dt_from_surf_gt = distance_transform_edt(~surf_gt, sampling=spacing)
    dt_from_surf_pred = distance_transform_edt(~surf_pred, sampling=spacing)

    pred_within = (dt_from_surf_gt[surf_pred] <= tolerance).sum()
    gt_within = (dt_from_surf_pred[surf_gt] <= tolerance).sum()

    return float((pred_within + gt_within) / (surf_pred.sum() + surf_gt.sum()))


def compute_betti0(pred: np.ndarray, gt: np.ndarray) -> int:
    from scipy.ndimage import label

    _, n_pred = label(pred > 0.5, structure=_CONNECTIVITY_26)
    _, n_gt = label(gt > 0.5, structure=_CONNECTIVITY_26)
    return abs(n_pred - n_gt)


def compute_hd95(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1, 1, 1)) -> float:
    from scipy.ndimage import distance_transform_edt

    pred_bool = pred > 0.5
    gt_bool = gt > 0.5

    if not pred_bool.any() or not gt_bool.any():
        return np.inf

    dt_pred = distance_transform_edt(~pred_bool, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_bool, sampling=spacing)

    all_distances = np.concatenate([dt_pred[gt_bool], dt_gt[pred_bool]])
    return float(np.percentile(all_distances, 95))


def compute_volume_delta(pred: np.ndarray, gt: np.ndarray) -> float:
    """Signed relative volume difference (|pred| - |gt|) / |gt|.

    The only directional metric here: < 0 means under-segmentation, > 0 over-segmentation.
    It is monotone in the drift bias mu of ``boundary_drift``, whereas NSD and HD95 are
    unsigned and cannot tell mu = -0.5 from mu = +0.5.
    """
    n_gt = float((gt > 0.5).sum())
    if n_gt == 0:
        return 0.0
    return float((pred > 0.5).sum() - n_gt) / n_gt


# ---------------------------------------------------------------------------------------------
# Scoring one prediction against several references
# ---------------------------------------------------------------------------------------------

class SurfaceFeatures:
    """Prediction-only quantities needed by HD95 and NSD (no skeleton, no components).

    Lighter than ``PredictionFeatures`` for analyses that only report distance metrics.
    ``pred`` must already be boolean.
    """

    __slots__ = ("spacing", "mask", "nonempty", "dist_to_mask", "surface", "dist_to_surface")

    def __init__(self, pred: np.ndarray, spacing: tuple):
        from scipy.ndimage import binary_erosion, distance_transform_edt

        self.spacing = spacing
        self.mask = pred
        self.nonempty = bool(pred.any())
        self.dist_to_mask = distance_transform_edt(~pred, sampling=spacing)
        self.surface = pred ^ binary_erosion(pred, structure=np.ones((3, 3, 3)))
        self.dist_to_surface = distance_transform_edt(~self.surface, sampling=spacing)


class PredictionFeatures:
    """Prediction-only quantities, reusable against any reference of the same case."""

    __slots__ = ("spacing", "mask", "nonempty", "skeleton", "dist_to_mask", "surface",
                 "dist_to_surface", "n_components")

    def __init__(self, pred: np.ndarray, spacing: tuple = (1, 1, 1)):
        from scipy.ndimage import binary_erosion, distance_transform_edt, label
        from skimage.morphology import skeletonize

        self.spacing = spacing
        self.mask = pred > 0.5
        self.nonempty = bool(self.mask.any())
        if not self.nonempty:
            self.skeleton = self.dist_to_mask = self.surface = self.dist_to_surface = None
            self.n_components = 0
            return
        struct = np.ones((3, 3, 3))
        self.skeleton = skeletonize(self.mask)
        self.dist_to_mask = distance_transform_edt(~self.mask, sampling=spacing)
        self.surface = self.mask ^ binary_erosion(self.mask, structure=struct)
        self.dist_to_surface = distance_transform_edt(~self.surface, sampling=spacing)
        self.n_components = label(self.mask, structure=_CONNECTIVITY_26)[1]


def cldice_from_features(pf: PredictionFeatures, gt_bool: np.ndarray) -> float:
    from skimage.morphology import skeletonize

    if not pf.nonempty and not gt_bool.any():
        return 1.0
    if not pf.nonempty or not gt_bool.any():
        return 0.0
    skel_gt = skeletonize(gt_bool)
    tprec = np.logical_and(pf.skeleton, gt_bool).sum() / max(pf.skeleton.sum(), 1)
    tsens = np.logical_and(skel_gt, pf.mask).sum() / max(skel_gt.sum(), 1)
    if tprec + tsens == 0:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))


def hd95_from_features(pf, gt_bool: np.ndarray) -> float:
    """HD95 from ``PredictionFeatures`` or ``SurfaceFeatures``."""
    from scipy.ndimage import distance_transform_edt

    if not pf.nonempty or not gt_bool.any():
        return np.inf
    dt_gt = distance_transform_edt(~gt_bool, sampling=pf.spacing)
    all_distances = np.concatenate([pf.dist_to_mask[gt_bool], dt_gt[pf.mask]])
    return float(np.percentile(all_distances, 95))


def nsd_from_features(pf, gt_bool: np.ndarray, tolerance: float = 2.0) -> float:
    """NSD at ``tolerance`` mm from ``PredictionFeatures`` or ``SurfaceFeatures``."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    if not pf.nonempty and not gt_bool.any():
        return 1.0
    if not pf.nonempty or not gt_bool.any():
        return 0.0
    struct = np.ones((3, 3, 3))
    surf_gt = gt_bool ^ binary_erosion(gt_bool, structure=struct)
    dt_from_surf_gt = distance_transform_edt(~surf_gt, sampling=pf.spacing)
    pred_within = (dt_from_surf_gt[pf.surface] <= tolerance).sum()
    gt_within = (pf.dist_to_surface[surf_gt] <= tolerance).sum()
    return float((pred_within + gt_within) / (pf.surface.sum() + surf_gt.sum()))


def betti0_from_features(pf: PredictionFeatures, gt_bool: np.ndarray) -> int:
    from scipy.ndimage import label

    _, n_gt = label(gt_bool, structure=_CONNECTIVITY_26)
    return abs(pf.n_components - n_gt)


def score_prediction(pf: PredictionFeatures, gt: np.ndarray, tolerance: float = 2.0) -> dict:
    """All metrics of one prediction (given as features) against one reference mask.

    The spacing comes from ``pf``: prediction and reference share the geometry of the case.
    Returns the columns of the master table: cldice, hd95, nsd (at ``tolerance``), nsd05,
    betti0, volume_delta.
    """
    gt_bool = gt > 0.5
    n_gt = float(gt_bool.sum())
    return {
        "cldice": cldice_from_features(pf, gt_bool),
        "hd95": hd95_from_features(pf, gt_bool),
        "nsd": nsd_from_features(pf, gt_bool, tolerance),
        "nsd05": nsd_from_features(pf, gt_bool, 0.5),   # sub-voxel: sensitive to boundary drift
        "betti0": betti0_from_features(pf, gt_bool),
        "volume_delta": (float(pf.mask.sum()) - n_gt) / n_gt if n_gt else 0.0,
    }
