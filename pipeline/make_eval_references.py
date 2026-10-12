#!/usr/bin/env python3
"""Write the degraded evaluation references GT- to disk.

Unlike the on-the-fly noise of the training trainers, these references are deterministic
and saved, so the evaluation is reproducible: case i of a split gets the seed
``seed + i`` (sorted file order). For every degraded scenario of
``evaluation.scenarios`` and every split present (labelsTr for out-of-fold evaluation,
labelsTs for the test set), the labels are written to ``<split>_<scenario_name>/``.

    python pipeline/make_eval_references.py \\
        --dataset_dir nnUNet_data/nnUNet_raw/Dataset100_PARSE \\
        --config configs/experiment_config.yaml --seed 42
"""

import argparse
import glob
import os
from functools import partial
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import yaml
from tqdm import tqdm

from prgt.noise import apply_noise_configs


def _degrade_one_case(idx_path, output_dir, noise_configs, seed):
    """Degrade one label file (parallel worker). Returns (file name, removed voxels)."""
    idx, lbl_path = idx_path
    fname = os.path.basename(lbl_path)
    nii = nib.load(lbl_path)
    data = nii.get_fdata().astype(np.float32)
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])  # the families work in mm

    before = int((data > 0).sum())
    degraded = apply_noise_configs(
        data[np.newaxis, ...], noise_configs,
        seed_base=seed + idx, spacing=spacing,
    )
    after = int((degraded[0] > 0).sum())

    nib.save(nib.Nifti1Image(degraded[0], nii.affine, nii.header),
             os.path.join(output_dir, fname))
    return fname, before - after


def degrade_labels(
    labels_dir: str,
    output_dir: str,
    noise_configs: list,
    seed: int = 42,
    workers: int = 1,
) -> None:
    """Write degraded copies of every label in ``labels_dir`` (cases in parallel).

    Each case is independent (seed = seed_base + index), so no state is shared.
    """
    os.makedirs(output_dir, exist_ok=True)
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
    print(f"Degrading {len(label_files)} labels ({workers} worker(s))...")
    for i, d in enumerate(noise_configs, 1):
        params = {k: v for k, v in d.items() if k != "family"}
        print(f"  {i}. {d['family']} - {params}")
    print(f"  seed: {seed}")

    work = partial(_degrade_one_case, output_dir=output_dir,
                   noise_configs=noise_configs, seed=seed)
    items = list(enumerate(label_files))
    n_degraded = 0
    if workers > 1:
        with Pool(workers) as pool:
            for fname, delta in tqdm(pool.imap_unordered(work, items),
                                     total=len(items), unit="case"):
                n_degraded += int(delta != 0)
    else:
        for item in tqdm(items, unit="case"):
            _, delta = work(item)
            n_degraded += int(delta != 0)

    print(f"  -> {len(label_files)} labels saved ({n_degraded} with fewer voxels) to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Write the degraded evaluation references")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="nnU-Net raw dataset folder (e.g. nnUNet_raw/Dataset100_PARSE)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="YAML configuration file",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="base seed (case i uses seed + i)"
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
        help="parallel processes (default: half of the cores); 1 = sequential",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Out-of-fold evaluation also needs the TRAINING labels degraded (the cases predicted
    # in validation by the folds), not only the test set: one labels{Tr,Ts}_<scenario>
    # folder per split present.
    splits = [s for s in ("labelsTr", "labelsTs")
              if os.path.isdir(os.path.join(args.dataset_dir, s))]
    if not splits:
        print(f"ERROR: neither labelsTr nor labelsTs in {args.dataset_dir}.")
        print("Check that the labels were copied during the conversion.")
        return

    for scenario in config["evaluation"]["scenarios"]:
        name = scenario["name"]
        deg = scenario.get("degradation")

        if deg is None:
            print(f"\n[{name}] no degradation (GT*): skipped")
            continue

        for split in splits:
            output_dir = os.path.join(args.dataset_dir, f"{split}_{name}")
            print(f"\n[{name}/{split}] writing degraded labels...")
            degrade_labels(
                labels_dir=os.path.join(args.dataset_dir, split),
                output_dir=output_dir,
                noise_configs=deg,
                seed=args.seed,
                workers=args.workers,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
