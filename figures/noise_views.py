"""2D and 3D views of what the noise operator does to a mask (technical-report figures).

For each noise family (prgt.noise), the degraded mask is compared with the original and
the difference is drawn in colour:

    RED   -> REMOVED voxels                 removed = GT* and not GT-
    BLUE  -> ADDED voxels (boundary drift)  added   = GT- and not GT*
    GREY  -> kept voxels                    kept    = GT* and GT-

Outputs per case:
    degradation_2d_<id>.png        grid families x {original | degraded | difference}
    degradation_3d_<id>.png        3D surface rendering, one view per family
    drift_mu_sweep_2d/3d_<id>.png  sweep of the drift bias mu of boundary_drift (--mu-sweep)

The helpers (load_mask, render_2d, render_3d, render_mu_sweep_2d/3d, add_mesh, set_theme)
are reused by report_noise_panels.py and report_clean_gt.py.

    python figures/noise_views.py                      # auto: 3 cases of labelsTr
    python figures/noise_views.py --input my.nii.gz --r 2 --p 0.5
    python figures/noise_views.py --families distal_omission boundary_drift
    python figures/noise_views.py --no-3d              # 2D only (fast)

    # drift bias mu of boundary_drift: under-segmentation <-> jitter <-> over-segmentation
    python figures/noise_views.py --mu-sweep -1 -0.5 0 0.5 1 --r 2 --p 1.0
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.measure import block_reduce, marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from prgt.metrics import compute_cldice, compute_nsd
from prgt.noise import FAMILIES, apply_noise

# Palette and theme
# Semantic colours (constant: readable on light and dark backgrounds).
_C_REMOVED = (0.85, 0.12, 0.14)  # red: removed voxels
_C_ADDED = (0.12, 0.40, 0.85)    # blue: added voxels (over-segmentation)

# Style variables driven by the theme (reassigned by set_theme()).
# Defaults = DARK theme (screen).
_BG = "#0e1117"                  # figure and axes background
_C_KEPT = (0.62, 0.62, 0.66)     # grey: kept
_TEXT = "white"                  # titles, main labels, legends
_SUBTEXT = "#d0d0d0"             # panel subtitles
_XLABEL = "#bfc7d5"              # axis labels (inter-annotator agreement)
_SPINE = "#333"                  # axes frame
_BOX_FC = "#1c1f26"              # annotation box background
_BOX_EC = "#444"                 # annotation box edge
_EMPTY = 0.0                     # empty pixels of the RGB overlays (0 = black, 1 = white)
_KEPT_ALPHA = 0.10               # opacity of the kept 3D mesh

# LIGHT theme overrides (article page): darker grey to stay readable on white, dark text,
# slightly more opaque translucent mesh.
_THEME_LIGHT = dict(
    _BG="white", _C_KEPT=(0.50, 0.50, 0.54), _TEXT="black", _SUBTEXT="#333",
    _XLABEL="#444", _SPINE="#bbb", _BOX_FC="#f2f2f2", _BOX_EC="#bbb",
    _EMPTY=1.0, _KEPT_ALPHA=0.16,
)


def set_theme(light: bool) -> None:
    """Switch the theme: dark (default) or light (white page).

    Reassigns the style globals read by every rendering function.
    """
    if light:
        globals().update(_THEME_LIGHT)

_FAMILY_LABEL = {
    "distal_omission": "distal_omission\n(fine-branch omission)",
    "boundary_drift": "boundary_drift\n(drift + surface jitter)",
    "boundary_jitter": "boundary_jitter\n(surface jitter ±r)",
    "distal_truncation": "distal_truncation\n(terminal truncation)",
    "homogeneous_morpho": "homogeneous_morpho\n(uniform erosion -- CONTROL)",
}


def _agreement(degraded, original, spacing):
    """Agreement GT- vs GT* on the bounding box of the structure: (clDice, NSD@0.5mm).

    clDice is the topological agreement (skeleton); NSD@0.5mm the surface agreement
    (sub-voxel tolerance, the only one sensitive to boundary drift). The crop speeds up
    the skeletonisation.
    """
    sl = bbox(original, margin=4)
    deg_c, orig_c = degraded[sl], original[sl]
    return compute_cldice(deg_c, orig_c), compute_nsd(deg_c, orig_c, spacing, tolerance=0.5)


# Geometry and I/O

def load_mask(path: str) -> tuple[np.ndarray, tuple]:
    """Binary mask (uint8) and voxel spacing (mm) of a NIfTI label file."""
    img = nib.load(path)
    data = (img.get_fdata() > 0.5).astype(np.uint8)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return data, spacing


def bbox(mask: np.ndarray, margin: int = 6) -> tuple:
    """Bounding box of the non-zero voxels plus ``margin``, as a slice tuple."""
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return tuple(slice(0, s) for s in mask.shape)
    lo = np.maximum(coords.min(0) - margin, 0)
    hi = np.minimum(coords.max(0) + margin + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _diff(original: np.ndarray, degraded: np.ndarray) -> tuple:
    o = original > 0.5
    d = degraded > 0.5
    removed = o & ~d
    added = d & ~o
    kept = o & d
    return kept, removed, added


def _best_slice(change: np.ndarray, axis: int = 2) -> int:
    """Slice index along ``axis`` with the largest number of changed voxels."""
    other = tuple(a for a in range(3) if a != axis)
    counts = change.sum(axis=other)
    return int(np.argmax(counts)) if counts.any() else change.shape[axis] // 2


# 2D rendering

def _overlay_rgb(kept_s, removed_s, added_s) -> np.ndarray:
    """RGB image (H, W, 3) from the boolean slice masks."""
    h, w = kept_s.shape
    rgb = np.full((h, w, 3), _EMPTY, np.float32)
    rgb[kept_s] = _C_KEPT
    rgb[removed_s] = _C_REMOVED
    rgb[added_s] = _C_ADDED
    return rgb


def _gray_rgb(mask_s, color=None) -> np.ndarray:
    if color is None:
        color = _C_KEPT
    h, w = mask_s.shape
    rgb = np.full((h, w, 3), _EMPTY, np.float32)
    rgb[mask_s] = color
    return rgb


def render_2d(original, spacing, families, r, p, seed, out_path, case_id, dpi=220, mu=0.0):
    nrows = len(families)
    fig, axes = plt.subplots(nrows, 3, figsize=(13.5, 4.4 * nrows),
                             facecolor=_BG, squeeze=False)
    aspect = spacing[1] / spacing[0] if spacing[0] > 0 else 1.0

    for row, family in enumerate(families):
        degraded = apply_noise(original.copy(), family, r, p, seed, spacing, mu=mu)
        kept, removed, added = _diff(original, degraded)
        change = removed | added
        z = _best_slice(change, axis=2)

        sl = bbox(kept | removed | added, margin=8)
        sl2 = (sl[0], sl[1], z)

        o_s = original[sl2] > 0.5
        d_s = degraded[sl2] > 0.5
        kept_s, removed_s, added_s = kept[sl2], removed[sl2], added[sl2]

        cl = compute_cldice(degraded, original)
        n_rem, n_add = int(removed.sum()), int(added.sum())
        frac_rem = 100.0 * n_rem / max(int((original > 0.5).sum()), 1)

        panels = [
            ("Original (GT*)", _gray_rgb(o_s)),
            (f"Degraded (GT⁻)  r={r} p={p}", _gray_rgb(d_s)),
            ("Diff  •  red=removed  blue=added", _overlay_rgb(kept_s, removed_s, added_s)),
        ]
        for col, (title, rgb) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower",
                      interpolation="nearest", aspect=aspect)
            ax.set_facecolor(_BG)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(_SPINE)
            if col == 0:
                ax.set_ylabel(_FAMILY_LABEL.get(family, family), color=_TEXT,
                              fontsize=12, fontweight="bold", labelpad=12)
            ax.set_title(title, color=_SUBTEXT, fontsize=11, pad=8)

        # numbers on the difference panel
        axes[row, 2].text(
            0.02, 0.02,
            f"clDice(GT⁻,GT*)={cl:.3f}\n−{n_rem} vx ({frac_rem:.1f}%)   +{n_add} vx",
            transform=axes[row, 2].transAxes, color=_TEXT, fontsize=9,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc=_BOX_FC, ec=_BOX_EC, alpha=0.85),
        )

    legend = [
        Patch(facecolor=_C_REMOVED, label="Removed (amputation)"),
        Patch(facecolor=_C_ADDED, label="Added (local dilation)"),
        Patch(facecolor=_C_KEPT, label="Kept"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False,
               labelcolor=_TEXT, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Degradation effects -- {case_id}  (axial slice, max-change slice)",
                 color=_TEXT, fontsize=15, fontweight="bold", y=1.015)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"  2D -> {out_path}  ({dpi} dpi)")


# 3D rendering

def add_mesh(ax, vol_bool, spacing, color, alpha):
    """Add the marching-cubes surface of a binary volume (padded to close the surfaces)."""
    if vol_bool.sum() < 4:
        return False
    padded = np.pad(vol_bool.astype(np.float32), 1, mode="constant")
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing)
    except (RuntimeError, ValueError):
        return False
    verts -= np.array(spacing)  # undo the padding offset
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)
    return True


def render_3d(original, spacing, families, r, p, seed, out_path, case_id, ds_factor, dpi=220, mu=0.0):
    ncols = len(families)
    fig = plt.figure(figsize=(7.0 * ncols, 7.2), facecolor=_BG)

    for i, family in enumerate(families):
        degraded = apply_noise(original.copy(), family, r, p, seed, spacing, mu=mu)
        kept, removed, added = _diff(original, degraded)

        sl = bbox(kept | removed | added, margin=4)
        kept_c, removed_c, added_c = kept[sl], removed[sl], added[sl]

        # downsample by max-pooling (keeps thin structures)
        if ds_factor > 1:
            bs = (ds_factor, ds_factor, ds_factor)
            kept_c = block_reduce(kept_c, bs, np.max)
            removed_c = block_reduce(removed_c, bs, np.max)
            added_c = block_reduce(added_c, bs, np.max)
        sp = tuple(s * ds_factor for s in spacing)

        ax = fig.add_subplot(1, ncols, i + 1, projection="3d")
        ax.set_facecolor(_BG)
        # kept structure: translucent, in the background
        add_mesh(ax, kept_c, sp, _C_KEPT, alpha=_KEPT_ALPHA)
        add_mesh(ax, removed_c, sp, _C_REMOVED, alpha=0.55)
        add_mesh(ax, added_c, sp, _C_ADDED, alpha=0.65)

        dims = np.array(kept_c.shape) * np.array(sp)
        ax.set_box_aspect(dims)
        ax.view_init(elev=18, azim=-60)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_axis_off()
        n_rem, n_add = int(removed.sum()), int(added.sum())
        ax.set_title(f"{family}\n−{n_rem} vx   +{n_add} vx",
                     color=_TEXT, fontsize=12, fontweight="bold", pad=2)

    legend = [
        Patch(facecolor=_C_REMOVED, label="Removed"),
        Patch(facecolor=_C_ADDED, label="Added"),
        Patch(facecolor=_C_KEPT, label="Kept (translucent)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               labelcolor=_TEXT, fontsize=12)
    fig.suptitle(f"Degradations in 3D — {case_id}  (ds×{ds_factor})",
                 color=_TEXT, fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"  3D -> {out_path}  ({dpi} dpi)")


# Sweep of the drift bias mu (boundary_drift)
# Shows the central phenomenon: mu = 0 -> unbiased jitter (averages out, augmentation);
# mu != 0 -> learnable systematic bias (under- or over-segmentation). The inter-annotator
# agreement (clDice + NSD@0.5mm) is printed to show the noise stays realistic.

def _drift_regime(mu: float) -> str:
    if mu < 0:
        return "under-segmentation (bias -)"
    if mu > 0:
        return "over-segmentation (bias +)"
    return "pure jitter (zero bias)"


def render_mu_sweep_2d(original, spacing, r, p, mus, seed, out_path, case_id, dpi=220):
    degraded = [apply_noise(original.copy(), "boundary_drift", r, p, seed, spacing, mu=m) for m in mus]
    diffs = [_diff(original, d) for d in degraded]
    change_all = np.zeros(original.shape, bool)
    for _, removed, added in diffs:
        change_all |= removed | added
    z = _best_slice(change_all, axis=2)
    sl = bbox(original, margin=8)
    aspect = spacing[1] / spacing[0] if spacing[0] > 0 else 1.0

    fig, axes = plt.subplots(1, len(mus), figsize=(5.0 * len(mus), 6.2),
                             facecolor=_BG, squeeze=False)
    for j, (mu, (kept, removed, added)) in enumerate(zip(mus, diffs)):
        sl2 = (sl[0], sl[1], z)
        ax = axes[0, j]
        ax.imshow(np.transpose(_overlay_rgb(kept[sl2], removed[sl2], added[sl2]), (1, 0, 2)),
                  origin="lower", interpolation="nearest", aspect=aspect)
        ax.set_facecolor(_BG); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(_SPINE)
        ax.set_title(f"μ = {mu:+.1f}\n{_drift_regime(mu)}",
                     color=_TEXT, fontsize=13, fontweight="bold", pad=8)
        cldice, nsd = _agreement(degraded[j], original, spacing)
        ax.set_xlabel(f"inter-observer agreement\nclDice {cldice:.3f}  ·  NSD@0.5 {nsd:.3f}",
                      color=_XLABEL, fontsize=10, labelpad=10)

    _drift_legend_and_save(fig, out_path, case_id, r, dpi, dim="2D")
    print(f"  mu sweep 2D -> {out_path}  ({dpi} dpi)")


def render_mu_sweep_3d(original, spacing, r, p, mus, seed, out_path, case_id, ds_factor, dpi=220):
    fig = plt.figure(figsize=(6.0 * len(mus), 7.0), facecolor=_BG)
    for j, mu in enumerate(mus):
        degraded = apply_noise(original.copy(), "boundary_drift", r, p, seed, spacing, mu=mu)
        kept, removed, added = _diff(original, degraded)
        sl = bbox(original, margin=4)
        layers = [kept[sl], removed[sl], added[sl]]
        if ds_factor > 1:
            layers = [block_reduce(v, (ds_factor,) * 3, np.max) for v in layers]
        sp = tuple(s * ds_factor for s in spacing)

        ax = fig.add_subplot(1, len(mus), j + 1, projection="3d")
        ax.set_facecolor(_BG)
        for vol, color, alpha in zip(layers, (_C_KEPT, _C_REMOVED, _C_ADDED), (_KEPT_ALPHA, 0.55, 0.65)):
            add_mesh(ax, vol, sp, color, alpha)
        ax.set_box_aspect(np.array(layers[0].shape) * np.array(sp))
        ax.view_init(elev=16, azim=-62)
        ax.set_axis_off()
        cldice, nsd = _agreement(degraded, original, spacing)
        ax.set_title(f"μ = {mu:+.1f}  ·  {_drift_regime(mu)}\nclDice {cldice:.3f} · NSD@0.5 {nsd:.3f}",
                     color=_TEXT, fontsize=12, fontweight="bold", pad=2)

    _drift_legend_and_save(fig, out_path, case_id, r, dpi, dim=f"3D (ds×{ds_factor})")
    print(f"  mu sweep 3D -> {out_path}  ({dpi} dpi)")


def _drift_legend_and_save(fig, out_path, case_id, r, dpi, dim):
    legend = [
        Patch(facecolor=_C_REMOVED, label="Removed (boundary receded)"),
        Patch(facecolor=_C_ADDED, label="Added (boundary advanced)"),
        Patch(facecolor=_C_KEPT, label="Kept"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               labelcolor=_TEXT, fontsize=12)
    fig.suptitle(
        f"boundary_drift — bias mu sweep — {case_id}  [{dim}, r={r}]\n"
        "mu=0 unbiased (cancels to augmentation, Q2)   ·   μ≠0 systematic bias "
        "learnable (Q3)   .   inter-observer agreement preserved",
        color=_TEXT, fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


# Command line

def _resolve_inputs(args) -> list[str]:
    if args.input:
        return [args.input]
    pats = [
        "nnUNet_data/nnUNet_raw/Dataset100_PARSE/labelsTr/*.nii.gz",
        "data/train/labels/*.nii.gz",
    ]
    for pat in pats:
        files = sorted(glob.glob(pat))
        if files:
            return files[: args.n_cases]
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="single .nii.gz file (default: first cases of labelsTr)")
    ap.add_argument("--n-cases", type=int, default=3, help="number of cases in auto mode")
    ap.add_argument("--families", nargs="+",
                    default=["distal_omission", "boundary_drift",
                             "distal_truncation", "homogeneous_morpho"],
                    choices=list(FAMILIES),
                    help="families to show (boundary_jitter = alias of boundary_drift with mu=0)")
    ap.add_argument("--r", type=float, default=2.0, help="parameter r (radius / amplitude)")
    ap.add_argument("--p", type=float, default=0.5, help="parameter p (severity)")
    ap.add_argument("--mu", type=float, default=0.0, help="boundary drift bias (boundary_drift)")
    ap.add_argument("--seed", type=int, default=0, help="noise seed")
    ap.add_argument("--ds-factor", type=int, default=2, help="3D downsampling (max-pool; 1 = full resolution)")
    ap.add_argument("--dpi", type=int, default=220, help="figure resolution (dots per inch)")
    ap.add_argument("--no-3d", action="store_true", help="2D only (fast)")
    ap.add_argument("--light", action="store_true",
                    help="light theme (white background), for a PDF or an article")
    ap.add_argument("--mu-sweep", nargs="+", type=float, metavar="MU",
                    help="draw the sweep of the drift bias mu of boundary_drift "
                         "(e.g. --mu-sweep -1 -0.5 0 0.5 1); p=1.0 recommended")
    ap.add_argument("--output-dir", default="report/technical/figures")
    args = ap.parse_args()

    set_theme(args.light)
    os.makedirs(args.output_dir, exist_ok=True)
    inputs = _resolve_inputs(args)
    if not inputs:
        print("No mask found. Give --input or check labelsTr/.")
        return

    print(f"families: {args.families}   r={args.r} p={args.p} mu={args.mu} seed={args.seed}")
    for path in inputs:
        case_id = os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")
        original, spacing = load_mask(path)
        if original.sum() == 0:
            print(f"[skip] {case_id}: empty mask")
            continue
        print(f"[{case_id}] shape={original.shape} spacing={spacing} fg={int(original.sum())}")

        # dedicated mode: sweep of the drift bias mu (boundary_drift)
        if args.mu_sweep:
            p_sweep = args.p if args.p < 1.0 else 1.0   # p = 1.0: the whole boundary drifts
            render_mu_sweep_2d(original, spacing, args.r, p_sweep, args.mu_sweep, args.seed,
                               os.path.join(args.output_dir, f"drift_mu_sweep_2d_{case_id}.png"),
                               case_id, dpi=args.dpi)
            if not args.no_3d:
                render_mu_sweep_3d(original, spacing, args.r, p_sweep, args.mu_sweep, args.seed,
                                   os.path.join(args.output_dir, f"drift_mu_sweep_3d_{case_id}.png"),
                                   case_id, args.ds_factor, dpi=args.dpi)
            continue

        render_2d(original, spacing, args.families, args.r, args.p, args.seed,
                  os.path.join(args.output_dir, f"degradation_2d_{case_id}.png"), case_id,
                  dpi=args.dpi, mu=args.mu)
        if not args.no_3d:
            render_3d(original, spacing, args.families, args.r, args.p, args.seed,
                      os.path.join(args.output_dir, f"degradation_3d_{case_id}.png"),
                      case_id, args.ds_factor, dpi=args.dpi, mu=args.mu)


if __name__ == "__main__":
    main()
