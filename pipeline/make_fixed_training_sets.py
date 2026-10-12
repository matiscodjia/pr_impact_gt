#!/usr/bin/env python3
"""Build a nnU-Net dataset with the same images and FROZEN degraded training labels.

The new dataset contains:
- imagesTr / imagesTs : symbolic links to Dataset100_PARSE (no copy)
- labelsTr            : deterministic noise (one seed per case), generated once
- labelsTs            : symbolic links to Dataset100_PARSE/labelsTs (evaluation unchanged)

The noise pipeline is read from ``fixed_datasets[dataset_id]`` in the configuration; the
families are applied in the order of the list, so adding one needs no code change. Case i
(sorted order) gets the seed ``seed + i``, the same convention as make_eval_references.py,
so with the same configuration the training labels equal the evaluation references.

    python pipeline/make_fixed_training_sets.py --dataset_id 103 \\
        --source_dir $nnUNet_raw/Dataset100_PARSE \\
        --output_dir $nnUNet_raw/Dataset103_PARSE_DriftMuMinus \\
        --config configs/experiment_config.yaml

Afterwards, do NOT run nnUNetv2_plan_and_preprocess on the new dataset: it would compute
the CT intensity normalisation under the degraded mask, and the network would differ from
M0 by more than its labels. Reuse the plans of Dataset100 instead (cluster/prepare_q3.sh).
"""

import argparse
import glob
import json
import os
import sys
from functools import partial
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import yaml
from tqdm import tqdm

from prgt.noise import apply_noise_configs


def _symlink(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def _degrade_label(
    src_path: str,
    dst_path: str,
    noise_configs: list,
    seed: int,
) -> dict:
    """Degrade one NIfTI label with a fixed seed; returns voxel counts before and after."""
    nii = nib.load(src_path)
    data = nii.get_fdata().astype(np.float32)
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])  # the families work in mm

    n_voxels_before = int((data > 0).sum())
    degraded = apply_noise_configs(
        data[np.newaxis, ...], noise_configs,
        seed_base=seed, spacing=spacing,
    )[0]
    n_voxels_after = int((degraded > 0).sum())

    nib.save(nib.Nifti1Image(degraded, nii.affine, nii.header), dst_path)
    return {"before": n_voxels_before, "after": n_voxels_after}


def _degrade_label_worker(idx_src, labels_out_dir, noise_configs, seed):
    """Parallel worker: degrade labelsTr/<file> with seed + index. Returns the voxel change."""
    idx, src = idx_src
    dst = os.path.join(labels_out_dir, os.path.basename(src))
    stats = _degrade_label(src, dst, noise_configs, seed + idx)
    return stats["before"] - stats["after"]


def main():
    parser = argparse.ArgumentParser(
        description="Build a dataset with frozen degraded training labels"
    )
    parser.add_argument(
        "--dataset_id", type=int, required=True,
        help="target dataset ID, as defined in fixed_datasets of the configuration (e.g. 103)",
    )
    parser.add_argument(
        "--source_dir", required=True,
        help="Dataset100_PARSE folder in nnUNet_raw",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="output folder (e.g. $nnUNet_raw/Dataset103_PARSE_DriftMuMinus)",
    )
    parser.add_argument(
        "--config", default="configs/experiment_config.yaml",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="base seed; case i gets seed + i",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
        help="parallel processes (default: half of the cores); 1 = sequential",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        print(f"ERROR: {args.source_dir} does not exist.")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    fixed_datasets = config.get("fixed_datasets", {})
    if args.dataset_id not in fixed_datasets:
        available = sorted(fixed_datasets.keys())
        print(f"ERROR: dataset_id={args.dataset_id} not in fixed_datasets. Available: {available}")
        sys.exit(1)

    dataset_cfg = fixed_datasets[args.dataset_id]
    noise_configs = dataset_cfg["degradations"]
    fixed_id = args.dataset_id

    print(f"Source  : {args.source_dir}")
    print(f"Output  : {args.output_dir}")
    print(f"Dataset : {dataset_cfg.get('name', f'Dataset{fixed_id:03d}')}")
    print(f"Noise pipeline ({len(noise_configs)} step(s)):")
    for i, d in enumerate(noise_configs, 1):
        family = d.get("family", d.get("type", "?"))
        params = {k: v for k, v in d.items() if k not in ("family", "type")}
        print(f"  {i}. {family} - {params}")
    print(f"Seed    : {args.seed} (+ case index)")

    for subdir in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # imagesTr: symbolic links
    images_tr = sorted(glob.glob(os.path.join(args.source_dir, "imagesTr", "*.nii.gz")))
    for src in tqdm(images_tr, desc="imagesTr (symlinks)", unit="case"):
        _symlink(src, os.path.join(args.output_dir, "imagesTr", os.path.basename(src)))
    print(f"  imagesTr: {len(images_tr)} links")

    # labelsTr: frozen per-case noise (parallel)
    labels_tr = sorted(glob.glob(os.path.join(args.source_dir, "labelsTr", "*.nii.gz")))
    labels_out = os.path.join(args.output_dir, "labelsTr")
    work = partial(_degrade_label_worker, labels_out_dir=labels_out,
                   noise_configs=noise_configs, seed=args.seed)
    items = list(enumerate(labels_tr))
    delta_voxels = []
    print(f"  labelsTr: degrading {len(items)} cases ({args.workers} worker(s))...")
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for delta in tqdm(pool.imap_unordered(work, items), total=len(items),
                              desc="labelsTr (noise)", unit="case"):
                delta_voxels.append(delta)
    else:
        for item in tqdm(items, desc="labelsTr (noise)", unit="case"):
            delta_voxels.append(work(item))
    n_degraded = sum(1 for d in delta_voxels if d != 0)

    if delta_voxels:
        print(f"  labelsTr: {len(labels_tr)} labels | {n_degraded} with a voxel change | "
              f"mean change = {np.mean(delta_voxels):+.0f} voxels/case")

    # imagesTs: symbolic links
    images_ts = sorted(glob.glob(os.path.join(args.source_dir, "imagesTs", "*.nii.gz")))
    for src in tqdm(images_ts, desc="imagesTs (symlinks)", unit="case"):
        _symlink(src, os.path.join(args.output_dir, "imagesTs", os.path.basename(src)))
    print(f"  imagesTs: {len(images_ts)} links")

    # labelsTs: symbolic links (evaluation unchanged)
    labels_ts = sorted(glob.glob(os.path.join(args.source_dir, "labelsTs", "*.nii.gz")))
    for src in tqdm(labels_ts, desc="labelsTs (symlinks)", unit="case"):
        _symlink(src, os.path.join(args.output_dir, "labelsTs", os.path.basename(src)))
    print(f"  labelsTs: {len(labels_ts)} links")

    # dataset.json
    with open(os.path.join(args.source_dir, "dataset.json")) as f:
        dataset_json = json.load(f)

    dataset_json["name"] = dataset_cfg.get("name", f"Dataset{fixed_id:03d}_PARSE_Fixed")
    pipeline_summary = " | ".join(
        f"{d.get('family', d.get('type', '?'))} " + " ".join(
            f"{k}={v}" for k, v in d.items() if k not in ("family", "type")
        )
        for d in noise_configs
    )
    dataset_json["description"] = (
        f"PARSE with fixed degraded training labels — {pipeline_summary} — seed={args.seed}"
    )
    with open(os.path.join(args.output_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDataset written to: {args.output_dir}")
    print("Next step: reuse the plans of Dataset100 (cluster/prepare_q3.sh); do not run")
    print(f"nnUNetv2_plan_and_preprocess on Dataset{fixed_id:03d}.")


if __name__ == "__main__":
    main()
