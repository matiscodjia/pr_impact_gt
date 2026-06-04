#!/usr/bin/env python3
"""Analyse visuelle du Betti0 entre prédictions et GT.

Pour chaque cas dans PRED_DIR, calcule le Betti0, visualise les composantes
connexes colorées sur des coupes axiales, et vérifie si un seuil de taille
suffirait à expliquer les valeurs élevées.

Usage
-----
    python scripts/analyze_betti0.py \
        --pred_dir /path/to/predictions \
        --gt_dir   $nnUNet_raw/Dataset100_PARSE/labelsTs \
        --out_dir  results/betti0_analysis

    # Analyser un seul cas en détail
    python scripts/analyze_betti0.py \
        --pred_dir /path/to/predictions \
        --gt_dir   $nnUNet_raw/Dataset100_PARSE/labelsTs \
        --out_dir  results/betti0_analysis \
        --case     PARSE_0091.nii.gz
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as _label_cc
from tqdm import tqdm


# Connectivité 26 explicite. Le défaut scipy (structure=None -> connectivité 6,
# faces uniquement) fragmente un arbre vasculaire fin en dizaines/centaines de
# composantes parasites et gonfle artificiellement le Betti0.
_STRUCT_26 = generate_binary_structure(3, 3)


def label_cc(mask: np.ndarray):
    """Étiquetage des composantes connexes en connectivité 26 (3D)."""
    return _label_cc(mask, structure=_STRUCT_26)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _random_colors(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = rng.uniform(0.35, 1.0, size=(n + 1, 3))
    cols[0] = 0.0
    return cols


def _label_to_rgba(labeled_2d: np.ndarray, colors: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    rgba = np.zeros((*labeled_2d.shape, 4))
    for lid in np.unique(labeled_2d):
        if lid == 0:
            continue
        m = labeled_2d == lid
        rgba[m, :3] = colors[min(lid, len(colors) - 1)]
        rgba[m, 3] = alpha
    return rgba


def _component_sizes(labeled_vol: np.ndarray) -> np.ndarray:
    counts = np.bincount(labeled_vol.ravel())
    return counts[1:]


# ── Analyse par cas ───────────────────────────────────────────────────────────

def compute_row(pred_path: str, gt_path: str) -> dict:
    pred = nib.load(pred_path).get_fdata().astype(bool)
    gt   = nib.load(gt_path).get_fdata().astype(bool)
    _, n_pred = label_cc(pred)
    _, n_gt   = label_cc(gt)
    return {
        "case":   os.path.basename(pred_path),
        "n_gt":   n_gt,
        "n_pred": n_pred,
        "betti0": abs(n_pred - n_gt),
        "delta":  n_pred - n_gt,
    }


# ── Figures ──────────────────────────────────────────────────────────────────

def fig_overview(df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    labels = df["case"].str.replace(".nii.gz", "", regex=False)

    axes[0].bar(labels, df["betti0"], color="steelblue")
    axes[0].set_xlabel("Cas")
    axes[0].set_ylabel("Betti0")
    axes[0].set_title("Betti0 par cas")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].axhline(df["betti0"].mean(), color="tomato", linestyle="--",
                    label=f"Moyenne = {df['betti0'].mean():.1f}")
    axes[0].legend()

    x = np.arange(len(df))
    axes[1].bar(x - 0.2, df["n_gt"],   width=0.4, label="GT",   color="green",  alpha=0.75)
    axes[1].bar(x + 0.2, df["n_pred"], width=0.4, label="Pred", color="tomato", alpha=0.75)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_ylabel("Composantes connexes")
    axes[1].set_title("GT vs Pred — nombre de composantes connexes")
    axes[1].legend()

    fig.suptitle("Vue d'ensemble Betti0", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sauvegardé : {path}")


def fig_colored_slices(
    labeled_gt: np.ndarray, n_gt: int,
    labeled_pred: np.ndarray, n_pred: int,
    case_name: str, out_dir: str,
    n_slices: int = 6,
) -> None:
    depth = labeled_gt.shape[2]
    slices = np.linspace(int(depth * 0.15), int(depth * 0.85), n_slices, dtype=int)

    colors_gt   = _random_colors(n_gt)
    colors_pred = _random_colors(n_pred)

    fig, axes = plt.subplots(2, n_slices, figsize=(n_slices * 3, 7))

    for col, sl in enumerate(slices):
        sl_gt   = labeled_gt[:, :, sl]
        sl_pred = labeled_pred[:, :, sl]

        for row, (sl_lab, colors, tag) in enumerate([
            (sl_gt,   colors_gt,   f"GT  z={sl}"),
            (sl_pred, colors_pred, f"Pred  z={sl}"),
        ]):
            rgba = _label_to_rgba(sl_lab, colors)
            axes[row, col].imshow(rgba.transpose(1, 0, 2), origin="lower", interpolation="nearest")
            axes[row, col].set_title(tag, fontsize=8)
            axes[row, col].axis("off")

    betti0 = abs(n_pred - n_gt)
    direction = "pred a trop de cc" if n_pred > n_gt else "pred manque des cc"
    fig.suptitle(
        f"{case_name}  —  GT: {n_gt} cc  |  Pred: {n_pred} cc  |  Betti0 = {betti0}  ({direction})",
        fontsize=11,
    )
    plt.tight_layout()
    stem = case_name.replace(".nii.gz", "")
    path = os.path.join(out_dir, f"{stem}_colored_slices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sauvegardé : {path}")


def fig_component_sizes(
    labeled_gt: np.ndarray, n_gt: int,
    labeled_pred: np.ndarray, n_pred: int,
    case_name: str, out_dir: str,
) -> None:
    sizes_gt   = _component_sizes(labeled_gt)
    sizes_pred = _component_sizes(labeled_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, sizes, label, color in [
        (axes[0], sizes_gt,   f"GT ({n_gt} cc)",    "green"),
        (axes[1], sizes_pred, f"Pred ({n_pred} cc)", "tomato"),
    ]:
        ax.bar(range(1, len(sizes) + 1), np.sort(sizes)[::-1], color=color, alpha=0.8)
        ax.set_xlabel("Composante (triée par taille)")
        ax.set_ylabel("Taille (voxels)")
        ax.set_title(label)
        if sizes.max() > 0:
            ax.set_yscale("log")

    fig.suptitle(f"{case_name} — distribution des tailles de composantes connexes", fontsize=11)
    plt.tight_layout()
    stem = case_name.replace(".nii.gz", "")
    path = os.path.join(out_dir, f"{stem}_component_sizes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sauvegardé : {path}")


def fig_threshold_sweep(
    labeled_pred: np.ndarray, n_pred: int, n_gt: int,
    case_name: str, out_dir: str,
) -> pd.DataFrame:
    thresholds = [1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
    sizes = _component_sizes(labeled_pred)

    rows = []
    for t in thresholds:
        n_kept = int((sizes >= t).sum())
        rows.append({"seuil": t, "n_pred_kept": n_kept, "betti0": abs(n_kept - n_gt)})
    sweep = pd.DataFrame(rows)

    betti0_original = abs(n_pred - n_gt)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sweep["seuil"], sweep["betti0"], "o-", color="steelblue",
            label="Betti0 après seuil")
    ax.axhline(betti0_original, color="tomato", linestyle="--",
               label=f"Betti0 original ({betti0_original})")
    ax.set_xlabel("Taille minimale de composante conservée (voxels)")
    ax.set_ylabel("Betti0")
    ax.set_xscale("log")
    ax.set_title(f"{case_name} — impact d'un seuil de taille sur le Betti0")
    ax.legend()
    plt.tight_layout()

    stem = case_name.replace(".nii.gz", "")
    path = os.path.join(out_dir, f"{stem}_threshold_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sauvegardé : {path}")

    return sweep


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse visuelle du Betti0")
    parser.add_argument("--pred_dir", required=True,
                        help="Dossier des prédictions (.nii.gz)")
    parser.add_argument("--gt_dir", required=True,
                        help="Dossier des GT de test (.nii.gz)")
    parser.add_argument("--out_dir", default="results/betti0_analysis",
                        help="Dossier de sortie pour les figures")
    parser.add_argument("--case", default=None,
                        help="Analyser un seul cas en détail (ex: PARSE_0091.nii.gz). "
                             "Par défaut : le pire cas en Betti0.")
    parser.add_argument("--n_slices", type=int, default=6,
                        help="Nombre de coupes axiales à afficher")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Calcul du Betti0 pour tous les cas ──
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.nii.gz")))
    if not pred_files:
        print(f"ERREUR : aucun .nii.gz dans {args.pred_dir}")
        return

    print(f"Calcul du Betti0 sur {len(pred_files)} cas...")
    rows = []
    for pred_path in tqdm(pred_files, unit="cas"):
        fname = os.path.basename(pred_path)
        gt_path = os.path.join(args.gt_dir, fname)
        if not os.path.exists(gt_path):
            print(f"  [WARN] GT manquant pour {fname}, ignoré")
            continue
        rows.append(compute_row(pred_path, gt_path))

    if not rows:
        print("ERREUR : aucun cas avec GT correspondant trouvé.")
        return

    df = pd.DataFrame(rows).sort_values("betti0", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(args.out_dir, "betti0_summary.csv")
    df.to_csv(csv_path, index=False)

    print("\n── Tableau récapitulatif ─────────────────────────────")
    print(df.to_string(index=False))
    print(f"\nBetti0  moyen : {df['betti0'].mean():.2f}")
    print(f"Betti0  médian: {df['betti0'].median():.2f}")
    print(f"Betti0  max   : {df['betti0'].max()}")
    print(f"n_gt    moyen : {df['n_gt'].mean():.1f}")
    print(f"n_pred  moyen : {df['n_pred'].mean():.1f}")
    print(f"delta   moyen : {df['delta'].mean():+.2f}  "
          f"({'pred a tendance à sur-segmenter' if df['delta'].mean() > 0 else 'pred a tendance à sous-segmenter'})")
    print(f"\nTableau sauvegardé : {csv_path}")

    # ── 2. Figure vue d'ensemble ──
    print("\nGénération des figures...")
    fig_overview(df, args.out_dir)

    # ── 3. Analyse détaillée d'un cas ──
    case_name = args.case if args.case else df.iloc[0]["case"]
    print(f"\nAnalyse détaillée : {case_name}  (Betti0 = {df[df['case'] == case_name]['betti0'].values[0]})")

    pred_nii = nib.load(os.path.join(args.pred_dir, case_name))
    gt_nii   = nib.load(os.path.join(args.gt_dir,   case_name))
    pred = pred_nii.get_fdata().astype(bool)
    gt   = gt_nii.get_fdata().astype(bool)

    labeled_pred, n_pred = label_cc(pred)
    labeled_gt,   n_gt   = label_cc(gt)

    fig_colored_slices(labeled_gt, n_gt, labeled_pred, n_pred,
                       case_name, args.out_dir, args.n_slices)
    fig_component_sizes(labeled_gt, n_gt, labeled_pred, n_pred,
                        case_name, args.out_dir)
    sweep = fig_threshold_sweep(labeled_pred, n_pred, n_gt,
                                case_name, args.out_dir)

    print("\n── Sweep de seuil ────────────────────────────────────")
    print(sweep.to_string(index=False))

    # Trouver le seuil minimal qui ramène Betti0 à 0
    zero_rows = sweep[sweep["betti0"] == 0]
    if not zero_rows.empty:
        t_zero = zero_rows.iloc[0]["seuil"]
        print(f"\n→ Betti0 = 0 dès un seuil de {t_zero} voxels "
              f"(les petites composantes sont des artefacts)")
    else:
        print("\n→ Aucun seuil ne ramène Betti0 à 0 "
              "(l'écart vient de vraies structures manquantes/en trop)")

    print(f"\nToutes les figures dans : {args.out_dir}")


if __name__ == "__main__":
    main()
