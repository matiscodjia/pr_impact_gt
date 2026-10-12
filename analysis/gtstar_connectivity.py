#!/usr/bin/env python3
"""Nombre de composantes connexes de GT* selon la connexité (6, 18, 26).

Question : PARSE est réputé contenir des annotations déjà fragmentées, alors que le
manuscrit écrit N(GT*) = 1 dans chaque cas. Ce comptage dépend de la connexité : l'étude
utilise la connexité 26 (cross_evaluate.compute_betti0, betti0_counts.py), où deux voxels
qui ne se touchent que par un coin sont connexes. Ce script compte N(GT*) dans les trois
connexités, et mesure la taille des composantes hors de la principale.

Sortie : results/gtstar_connectivity.csv (une ligne par cas) + résumé.
Usage  : python analysis/gtstar_connectivity.py --workers 6
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

RAW = "nnUNet_data/nnUNet_raw/Dataset100_PARSE/labelsTr"
NAMES = {1: "6", 2: "18", 3: "26"}


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
        print(f"connexité {name:>2}: N=1 dans {(n == 1).sum()}/{len(d)} cas ; N médian {n.median():.0f}, "
              f"max {n.max()} ; 2e composante max {d[f'second_conn{name}'].max()} vox ; "
              f"volume hors composante principale : médian {frac.median():.4%}, max {frac.max():.4%}")


if __name__ == "__main__":
    main()
