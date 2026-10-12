#!/usr/bin/env python3
"""Number of connected components of GT* under 6-, 18- and 26-connectivity.

PARSE is said to contain fragmented annotations, while the paper states N(GT*) = 1 for
every case. The count depends on the connectivity: the study uses 26-connectivity
(prgt.metrics.compute_betti0, betti0_counts.py), in which two voxels that only share a
corner are connected. This script counts N(GT*) under the three connectivities and
measures the size of the components outside the main one.

Output : results/gtstar_connectivity.csv (one row per case) + printed summary.

    python analysis/gtstar_connectivity.py --workers 6
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

RAW = "nnUNet_data/nnUNet_raw/Dataset100_PARSE/labelsTr"
NAMES = {1: "6", 2: "18", 3: "26"}   # scipy connectivity rank -> neighbourhood size


def count(path):
    import nibabel as nib
    from scipy.ndimage import generate_binary_structure, label

    m = np.asanyarray(nib.load(path).dataobj) > 0
    row = {"case": os.path.basename(path), "voxels": int(m.sum())}
    for rank, name in NAMES.items():
        lab, n = label(m, structure=generate_binary_structure(3, rank))
        sizes = np.sort(np.bincount(lab.ravel())[1:])[::-1]
        row[f"n_conn{name}"] = int(n)
        row[f"second_conn{name}"] = int(sizes[1]) if n > 1 else 0
        row[f"outside_main_conn{name}"] = int(sizes[1:].sum())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    files = sorted(glob.glob(f"{RAW}/*.nii.gz"))
    with Pool(args.workers) as pool:
        d = pd.DataFrame(pool.map(count, files))
    d.to_csv("results/gtstar_connectivity.csv", index=False)
    for name in NAMES.values():
        n = d[f"n_conn{name}"]
        frac = d[f"outside_main_conn{name}"] / d.voxels
        print(f"connectivity {name:>2}: N=1 in {(n == 1).sum()}/{len(d)} cases; median N {n.median():.0f}, "
              f"max {n.max()}; largest 2nd component {d[f'second_conn{name}'].max()} vox; "
              f"volume outside the main component: median {frac.median():.4%}, max {frac.max():.4%}")


if __name__ == "__main__":
    main()
