"""Noise calibration: which noise parameters match inter-annotator agreement?

Applies each noise family (NO training) to a few PARSE masks and measures the agreement
between GT- and GT* with the instrument MATCHED to the kind of noise:

  - topological noise (distal_omission, distal_truncation) -> clDice, Betti-0
  - surface noise     (boundary_drift)                      -> NSD@0.5mm, HD95

All four metrics are computed for every row: the matched metric gives the verdict
("inside the inter-annotator window"), the others are controls that should stay flat,
which shows that the right axis is measured. Topological window: clDice in [0.85, 0.90].
analysis/calibration_85.py replays this grid on all 85 cases.

Output : results/noise_calibration.csv + annotated console table.

    python pipeline/calibrate_noise.py                      # default grid, 4 cases
    python pipeline/calibrate_noise.py --n-cases 8
    python pipeline/calibrate_noise.py --families distal_omission
    python pipeline/calibrate_noise.py --drift-mu -1 -0.5 0 0.5 1 --drift-r 2
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import nibabel as nib

from prgt.metrics import compute_betti0, compute_cldice, compute_hd95, compute_nsd
from prgt.noise import apply_noise

IAA_CLDICE = (0.85, 0.90)   # topological inter-observer window


def _load_masks(n_cases: int) -> list[tuple[np.ndarray, tuple]]:
    """Load n PARSE masks, cropped to their bounding box (speeds up skeletons and EDTs)."""
    patterns = ["nnUNet_data/nnUNet_raw/Dataset100_PARSE/labelsTr/*.nii.gz",
                "data/train/labels/*.nii.gz"]
    files = next((sorted(glob.glob(p))[:n_cases] for p in patterns if glob.glob(p)), [])
    masks = []
    for f in files:
        img = nib.load(f)
        mask = (img.get_fdata() > 0.5).astype(np.uint8)
        spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
        coords = np.argwhere(mask)
        lo, hi = coords.min(0), coords.max(0) + 1
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        masks.append((mask[sl], spacing))
    return masks


def _measure(masks, family, r, p, mu) -> dict:
    """Mean of the 4 agreement metrics GT- vs GT* over all masks."""
    cldice, nsd05, hd95, betti0 = [], [], [], []
    for mask, spacing in masks:
        degraded = apply_noise(mask.copy(), family, r, p, seed=0, spacing=spacing, mu=mu)
        cldice.append(compute_cldice(degraded, mask))
        nsd05.append(compute_nsd(degraded, mask, spacing, tolerance=0.5))
        hd95.append(compute_hd95(degraded, mask, spacing))
        betti0.append(compute_betti0(degraded, mask))
    return {"clDice": np.mean(cldice), "NSD@0.5": np.mean(nsd05),
            "HD95": np.mean(hd95), "Betti0": np.mean(betti0)}


def _default_grid(args) -> list[dict]:
    """(family, r, p, mu, axis); axis = the matched metric that gives the verdict."""
    grid = []
    if "distal_omission" in args.families:
        grid += [dict(family="distal_omission", r=r, p=p, mu=0.0, axis="clDice")
                 for r in (1, 2, 3) for p in (0.2, 0.3, 0.5)]
    if "distal_truncation" in args.families:
        grid += [dict(family="distal_truncation", r=r, p=p, mu=0.0, axis="clDice")
                 for r in (2, 3, 4) for p in (0.3, 0.5)]
    if "boundary_drift" in args.families:
        grid += [dict(family="boundary_drift", r=args.drift_r, p=args.drift_p, mu=mu,
                      axis="NSD@0.5") for mu in args.drift_mu]
    return grid


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-cases", type=int, default=4)
    ap.add_argument("--families", nargs="+",
                    default=["distal_omission", "distal_truncation", "boundary_drift"])
    ap.add_argument("--drift-mu", nargs="+", type=float, default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    ap.add_argument("--drift-r", type=float, default=2.0)
    ap.add_argument("--drift-p", type=float, default=1.0)
    ap.add_argument("--out", default="results/noise_calibration.csv")
    args = ap.parse_args()

    masks = _load_masks(args.n_cases)
    if not masks:
        print("No mask found (labelsTr/ or data/train/labels/).")
        return
    print(f"{len(masks)} PARSE masks · clDice window {IAA_CLDICE}\n")

    header = f"{'family':<18}{'r':>4}{'p':>6}{'mu':>6} | {'clDice':>8}{'NSD@0.5':>9}{'HD95':>7}{'Betti0':>8}  verdict"
    print(header); print("-" * len(header))
    rows = []
    for cfg in _default_grid(args):
        m = _measure(masks, cfg["family"], cfg["r"], cfg["p"], cfg["mu"])
        in_win = IAA_CLDICE[0] <= m["clDice"] <= IAA_CLDICE[1]
        verdict = "in window (clDice)" if (cfg["axis"] == "clDice" and in_win) else ""
        print(f"{cfg['family']:<18}{cfg['r']:>4.0f}{cfg['p']:>6.1f}{cfg['mu']:>6.1f} | "
              f"{m['clDice']:>8.3f}{m['NSD@0.5']:>9.3f}{m['HD95']:>7.2f}{m['Betti0']:>8.1f}  {verdict}")
        rows.append({**cfg, **{k: round(float(v), 4) for k, v in m.items()}})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {args.out}")
    print("Reading: clDice = topological agreement (omission); NSD@0.5/HD95 = surface "
          "agreement (drift). The OFF-axis metric must stay flat (evidence of dissociation).")


if __name__ == "__main__":
    main()
