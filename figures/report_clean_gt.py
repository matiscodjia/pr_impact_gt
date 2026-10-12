#!/usr/bin/env python3
"""3D rendering of a clean reference GT* (technical report, figure gt_clean_3D.png).

    python figures/report_clean_gt.py [--case PARSE_0001] [--out report/technical/figures]
"""
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

import noise_views as V


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="PARSE_0001")
    ap.add_argument("--labels", default="nnUNet_data/nnUNet_raw/Dataset100_PARSE/labelsTr")
    ap.add_argument("--out", default="report/technical/figures")
    args = ap.parse_args()

    V.set_theme(True)  # light background
    img = nib.load(os.path.join(args.labels, f"{args.case}.nii.gz"))
    gt = np.asarray(img.get_fdata()) > 0
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    ds = 2
    sl = V.bbox(gt, margin=4)
    g = block_reduce(gt[sl], (ds, ds, ds), np.max)
    sp = tuple(s * ds for s in spacing)

    fig = plt.figure(figsize=(7.5, 7.5), facecolor=V._BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(V._BG)
    V.add_mesh(ax, g, sp, (0.80, 0.16, 0.18), alpha=0.97)  # arterial crimson, solid
    ax.set_box_aspect(np.array(g.shape) * np.array(sp))
    ax.view_init(elev=18, azim=-60)
    ax.set_axis_off()
    fig.suptitle(f"GT* : clean annotation ({args.case})", color=V._TEXT,
                 fontsize=15, fontweight="bold", y=0.95)
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "gt_clean_3D.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=V._BG)
    print("saved", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
