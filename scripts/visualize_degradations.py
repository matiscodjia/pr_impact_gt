"""Visualisation 2D + 3D de l'effet des dégradations formelles sur les masques.

Pour chaque famille de bruit (degradations.py), on calcule la différence avec
le masque original et on la rend en couleur :

    ROUGE  → voxels RETIRÉS (zones amputées)            removed = GT*  & ¬GT⁻
    BLEU   → voxels AJOUTÉS (ex. boundary_jitter)       added   = GT⁻  & ¬GT*
    GRIS   → voxels conservés (structure inchangée)     kept    = GT*  & GT⁻

Sorties par cas :
    degradation_2d_<id>.png   — grille familles × {Original | Dégradé | Diff}
    degradation_3d_<id>.png   — rendu surfacique 3D, une vue par famille
    drift_mu_sweep_2d/3d_<id>.png — balayage du biais μ de boundary_drift (mode --mu-sweep)

Usage :
    python scripts/visualize_degradations.py                   # auto : 3 cas de labelsTr
    python scripts/visualize_degradations.py --input mon.nii.gz --r 2 --p 0.5
    python scripts/visualize_degradations.py --families distal_omission boundary_drift
    python scripts/visualize_degradations.py --no-3d           # 2D seulement (rapide)

    # Schéma du biais de dérive μ (boundary_drift) : sous-seg ↔ jitter ↔ sur-seg
    python scripts/visualize_degradations.py --mu-sweep -1 -0.5 0 0.5 1 --r 2 --p 1.0
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.measure import block_reduce, marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from degradations import generate, _REGISTRY  # noqa: E402

try:
    from cross_evaluate import compute_cldice, compute_nsd  # noqa: E402
except Exception:  # pragma: no cover
    compute_cldice = compute_nsd = None

# Palette
_C_KEPT = (0.62, 0.62, 0.66)   # gris
_C_REMOVED = (0.85, 0.12, 0.14)  # rouge
_C_ADDED = (0.12, 0.40, 0.85)    # bleu
_BG = "#0e1117"

_FAMILY_LABEL = {
    "distal_omission": "distal_omission\n(omission branches fines)",
    "boundary_drift": "boundary_drift\n(dérive + jitter de surface)",
    "boundary_jitter": "boundary_jitter\n(jitter de surface ±r)",
    "distal_truncation": "distal_truncation\n(troncature terminale)",
    "homogeneous_morpho": "homogeneous_morpho\n(érosion uniforme — CONTRÔLE)",
}


def _agreement(degraded, original, spacing):
    """Accord inter-annotateur GT⁻ vs GT* sur la bbox de la structure (rapide).

    Renvoie (clDice, NSD@0.5mm) — clDice = accord topologique (squelette),
    NSD@0.5mm = accord géométrique de surface (tolérance sous-voxel, seule sensible
    à la dérive de bord). Calculé sur la bbox pour accélérer le skeletonize."""
    if compute_cldice is None:
        return float("nan"), float("nan")
    sl = _bbox(original, margin=4)
    deg_c, orig_c = degraded[sl], original[sl]
    return compute_cldice(deg_c, orig_c), compute_nsd(deg_c, orig_c, spacing, tolerance=0.5)


# ── Géométrie / I/O ──────────────────────────────────────────────────────────

def _load_mask(path: str) -> tuple[np.ndarray, tuple]:
    img = nib.load(path)
    data = (img.get_fdata() > 0.5).astype(np.uint8)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return data, spacing


def _bbox(mask: np.ndarray, margin: int = 6) -> tuple:
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
    """Index de coupe le long de `axis` maximisant le nombre de voxels modifiés."""
    other = tuple(a for a in range(3) if a != axis)
    counts = change.sum(axis=other)
    return int(np.argmax(counts)) if counts.any() else change.shape[axis] // 2


# ── Rendu 2D ─────────────────────────────────────────────────────────────────

def _overlay_rgb(kept_s, removed_s, added_s) -> np.ndarray:
    """Construit une image RGB (H, W, 3) à partir des masques de coupe booléens."""
    h, w = kept_s.shape
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[kept_s] = _C_KEPT
    rgb[removed_s] = _C_REMOVED
    rgb[added_s] = _C_ADDED
    return rgb


def _gray_rgb(mask_s, color=_C_KEPT) -> np.ndarray:
    h, w = mask_s.shape
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[mask_s] = color
    return rgb


def render_2d(original, spacing, families, r, p, seed, out_path, case_id, dpi=220, mu=0.0):
    nrows = len(families)
    fig, axes = plt.subplots(nrows, 3, figsize=(13.5, 4.4 * nrows),
                             facecolor=_BG, squeeze=False)
    aspect = spacing[1] / spacing[0] if spacing[0] > 0 else 1.0

    for row, family in enumerate(families):
        degraded = generate(original.copy(), family, r, p, seed, spacing, mu=mu)
        kept, removed, added = _diff(original, degraded)
        change = removed | added
        z = _best_slice(change, axis=2)

        sl = _bbox(kept | removed | added, margin=8)
        sl2 = (sl[0], sl[1], z)

        o_s = original[sl2] > 0.5
        d_s = degraded[sl2] > 0.5
        kept_s, removed_s, added_s = kept[sl2], removed[sl2], added[sl2]

        cl = compute_cldice(degraded, original) if compute_cldice else float("nan")
        n_rem, n_add = int(removed.sum()), int(added.sum())
        frac_rem = 100.0 * n_rem / max(int((original > 0.5).sum()), 1)

        panels = [
            ("Original (GT*)", _gray_rgb(o_s)),
            (f"Dégradé (GT⁻)  r={r} p={p}", _gray_rgb(d_s)),
            ("Diff  •  rouge=retiré  bleu=ajouté", _overlay_rgb(kept_s, removed_s, added_s)),
        ]
        for col, (title, rgb) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower",
                      interpolation="nearest", aspect=aspect)
            ax.set_facecolor(_BG)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#333")
            if col == 0:
                ax.set_ylabel(_FAMILY_LABEL.get(family, family), color="white",
                              fontsize=12, fontweight="bold", labelpad=12)
            ax.set_title(title, color="#d0d0d0", fontsize=11, pad=8)

        # Annotation quantitative sur le panneau diff
        axes[row, 2].text(
            0.02, 0.02,
            f"clDice(GT⁻,GT*)={cl:.3f}\n−{n_rem} vx ({frac_rem:.1f}%)   +{n_add} vx",
            transform=axes[row, 2].transAxes, color="white", fontsize=9,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1c1f26", ec="#444", alpha=0.85),
        )

    legend = [
        Patch(facecolor=_C_REMOVED, label="Retiré (amputation)"),
        Patch(facecolor=_C_ADDED, label="Ajouté (dilatation locale)"),
        Patch(facecolor=_C_KEPT, label="Conservé"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False,
               labelcolor="white", fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Effet des dégradations — {case_id}  (coupe axiale, slice de change max)",
                 color="white", fontsize=15, fontweight="bold", y=1.015)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"  2D → {out_path}  ({dpi} dpi)")


# ── Rendu 3D ─────────────────────────────────────────────────────────────────

def _add_mesh(ax, vol_bool, spacing, color, alpha):
    """Marching cubes sur un volume binaire (padding pour fermer les surfaces)."""
    if vol_bool.sum() < 4:
        return False
    padded = np.pad(vol_bool.astype(np.float32), 1, mode="constant")
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing)
    except (RuntimeError, ValueError):
        return False
    verts -= np.array(spacing)  # compenser le padding
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)
    return True


def render_3d(original, spacing, families, r, p, seed, out_path, case_id, ds_factor, dpi=220, mu=0.0):
    ncols = len(families)
    fig = plt.figure(figsize=(7.0 * ncols, 7.2), facecolor=_BG)

    for i, family in enumerate(families):
        degraded = generate(original.copy(), family, r, p, seed, spacing, mu=mu)
        kept, removed, added = _diff(original, degraded)

        sl = _bbox(kept | removed | added, margin=4)
        kept_c, removed_c, added_c = kept[sl], removed[sl], added[sl]

        # Downsample par max-pooling (préserve les structures fines)
        if ds_factor > 1:
            bs = (ds_factor, ds_factor, ds_factor)
            kept_c = block_reduce(kept_c, bs, np.max)
            removed_c = block_reduce(removed_c, bs, np.max)
            added_c = block_reduce(added_c, bs, np.max)
        sp = tuple(s * ds_factor for s in spacing)

        ax = fig.add_subplot(1, ncols, i + 1, projection="3d")
        ax.set_facecolor(_BG)
        # Structure conservée : translucide, en fond
        _add_mesh(ax, kept_c, sp, _C_KEPT, alpha=0.10)
        _add_mesh(ax, removed_c, sp, _C_REMOVED, alpha=0.55)
        _add_mesh(ax, added_c, sp, _C_ADDED, alpha=0.65)

        dims = np.array(kept_c.shape) * np.array(sp)
        ax.set_box_aspect(dims)
        ax.view_init(elev=18, azim=-60)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_axis_off()
        n_rem, n_add = int(removed.sum()), int(added.sum())
        ax.set_title(f"{family}\n−{n_rem} vx   +{n_add} vx",
                     color="white", fontsize=12, fontweight="bold", pad=2)

    legend = [
        Patch(facecolor=_C_REMOVED, label="Retiré"),
        Patch(facecolor=_C_ADDED, label="Ajouté"),
        Patch(facecolor=_C_KEPT, label="Conservé (translucide)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               labelcolor="white", fontsize=12)
    fig.suptitle(f"Dégradations en 3D — {case_id}  (ds×{ds_factor})",
                 color="white", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"  3D → {out_path}  ({dpi} dpi)")


# ── Balayage du biais μ (boundary_drift) ─────────────────────────────────────
# Montre le phénomène central : μ=0 → jitter non biaisé (se compense, augmentation) ;
# μ≠0 → biais systématique apprenable (sous- ou sur-segmentation). L'accord
# inter-annotateur (clDice + NSD@0.5mm) est annoté pour prouver le réalisme.

def _drift_regime(mu: float) -> str:
    if mu < 0:
        return "sous-segmentation (biais −)"
    if mu > 0:
        return "sur-segmentation (biais +)"
    return "jitter pur (biais nul)"


def render_mu_sweep_2d(original, spacing, r, p, mus, seed, out_path, case_id, dpi=220):
    degraded = [generate(original.copy(), "boundary_drift", r, p, seed, spacing, mu=m) for m in mus]
    diffs = [_diff(original, d) for d in degraded]
    change_all = np.zeros(original.shape, bool)
    for _, removed, added in diffs:
        change_all |= removed | added
    z = _best_slice(change_all, axis=2)
    sl = _bbox(original, margin=8)
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
            spine.set_color("#333")
        ax.set_title(f"μ = {mu:+.1f}\n{_drift_regime(mu)}",
                     color="white", fontsize=13, fontweight="bold", pad=8)
        cldice, nsd = _agreement(degraded[j], original, spacing)
        ax.set_xlabel(f"accord inter-annotateur\nclDice {cldice:.3f}  ·  NSD@0.5 {nsd:.3f}",
                      color="#bfc7d5", fontsize=10, labelpad=10)

    _drift_legend_and_save(fig, out_path, case_id, r, dpi, dim="2D")
    print(f"  μ-sweep 2D → {out_path}  ({dpi} dpi)")


def render_mu_sweep_3d(original, spacing, r, p, mus, seed, out_path, case_id, ds_factor, dpi=220):
    fig = plt.figure(figsize=(6.0 * len(mus), 7.0), facecolor=_BG)
    for j, mu in enumerate(mus):
        degraded = generate(original.copy(), "boundary_drift", r, p, seed, spacing, mu=mu)
        kept, removed, added = _diff(original, degraded)
        sl = _bbox(original, margin=4)
        layers = [kept[sl], removed[sl], added[sl]]
        if ds_factor > 1:
            layers = [block_reduce(v, (ds_factor,) * 3, np.max) for v in layers]
        sp = tuple(s * ds_factor for s in spacing)

        ax = fig.add_subplot(1, len(mus), j + 1, projection="3d")
        ax.set_facecolor(_BG)
        for vol, color, alpha in zip(layers, (_C_KEPT, _C_REMOVED, _C_ADDED), (0.08, 0.55, 0.65)):
            _add_mesh(ax, vol, sp, color, alpha)
        ax.set_box_aspect(np.array(layers[0].shape) * np.array(sp))
        ax.view_init(elev=16, azim=-62)
        ax.set_axis_off()
        cldice, nsd = _agreement(degraded, original, spacing)
        ax.set_title(f"μ = {mu:+.1f}  ·  {_drift_regime(mu)}\nclDice {cldice:.3f} · NSD@0.5 {nsd:.3f}",
                     color="white", fontsize=12, fontweight="bold", pad=2)

    _drift_legend_and_save(fig, out_path, case_id, r, dpi, dim=f"3D (ds×{ds_factor})")
    print(f"  μ-sweep 3D → {out_path}  ({dpi} dpi)")


def _drift_legend_and_save(fig, out_path, case_id, r, dpi, dim):
    legend = [
        Patch(facecolor=_C_REMOVED, label="Retiré (bord reculé)"),
        Patch(facecolor=_C_ADDED, label="Ajouté (bord avancé)"),
        Patch(facecolor=_C_KEPT, label="Conservé"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               labelcolor="white", fontsize=12)
    fig.suptitle(
        f"boundary_drift — balayage du biais μ — {case_id}  [{dim}, r={r}]\n"
        "μ=0 non biaisé (se compense → augmentation, Q2)   ·   μ≠0 biais systématique "
        "apprenable (Q3)   ·   accord inter-annotateur conservé",
        color="white", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


# ── Driver ───────────────────────────────────────────────────────────────────

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
    ap.add_argument("--input", help="Fichier .nii.gz unique (sinon auto depuis labelsTr)")
    ap.add_argument("--n-cases", type=int, default=3, help="Nb de cas en mode auto")
    ap.add_argument("--families", nargs="+",
                    default=["distal_omission", "boundary_drift",
                             "distal_truncation", "homogeneous_morpho"],
                    choices=list(_REGISTRY),
                    help="Familles à visualiser (boundary_jitter = alias de boundary_drift μ=0)")
    ap.add_argument("--r", type=float, default=2.0, help="Paramètre r (rayon/amplitude)")
    ap.add_argument("--p", type=float, default=0.5, help="Paramètre p (sévérité)")
    ap.add_argument("--mu", type=float, default=0.0, help="Biais de dérive du bord (boundary_drift)")
    ap.add_argument("--seed", type=int, default=0, help="Seed du bruit")
    ap.add_argument("--ds-factor", type=int, default=2, help="Downsample 3D (max-pool ; 1 = pleine résolution)")
    ap.add_argument("--dpi", type=int, default=220, help="Résolution des figures (points/pouce)")
    ap.add_argument("--no-3d", action="store_true", help="2D seulement (rapide)")
    ap.add_argument("--mu-sweep", nargs="+", type=float, metavar="MU",
                    help="Génère le schéma de balayage du biais μ pour boundary_drift "
                         "(ex. --mu-sweep -1 -0.5 0 0.5 1). p=1.0 conseillé.")
    ap.add_argument("--output-dir", default="result/figures/degradations")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    inputs = _resolve_inputs(args)
    if not inputs:
        print("Aucun masque trouvé. Spécifiez --input ou vérifiez labelsTr/.")
        return

    print(f"Familles : {args.families}   r={args.r} p={args.p} mu={args.mu} seed={args.seed}")
    for path in inputs:
        case_id = os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")
        original, spacing = _load_mask(path)
        if original.sum() == 0:
            print(f"[skip] {case_id} : masque vide")
            continue
        print(f"[{case_id}] shape={original.shape} spacing={spacing} fg={int(original.sum())}")

        # Mode dédié : schéma de balayage du biais μ (boundary_drift)
        if args.mu_sweep:
            p_sweep = args.p if args.p < 1.0 else 1.0   # p=1.0 : tout le bord dérive
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
