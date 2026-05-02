#!/usr/bin/env python3
"""Crée Dataset10{n}_PARSE_Fixed : mêmes images, labels d'entraînement dégradés fixes.

Ce script génère un dataset nnU-Net avec :
- imagesTr/imagesTs : liens symboliques vers Dataset100_PARSE (pas de duplication)
- labelsTr         : dégradations déterministes (seed par cas), générées une seule fois
- labelsTs         : liens symboliques vers Dataset100_PARSE/labelsTs (évaluation inchangée)

Le pipeline de dégradations est lu depuis ``fixed_datasets[dataset_id]`` dans la config.
Les dégradations sont stackées dans l'ordre de la liste — ajouter un type ne requiert
aucune modification du script.

Usage
-----
    # Dataset101 (morpho seule)
    python scripts/create_fixed_degraded_dataset.py \\
        --dataset_id 101 \\
        --source_dir $nnUNet_raw/Dataset100_PARSE \\
        --output_dir $nnUNet_raw/Dataset101_PARSE_Fixed \\
        --config configs/experiment_config.yaml

    # Dataset102 (morpho + omission)
    python scripts/create_fixed_degraded_dataset.py \\
        --dataset_id 102 \\
        --source_dir $nnUNet_raw/Dataset100_PARSE \\
        --output_dir $nnUNet_raw/Dataset102_PARSE_MorphoOmission

Après exécution :
    nnUNetv2_plan_and_preprocess -d <dataset_id> --verify_dataset_integrity
"""

import argparse
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from degradations import apply_degradation_pipeline


def _symlink(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def _degrade_label(
    src_path: str,
    dst_path: str,
    degradation_configs: list,
    seed: int,
) -> dict:
    """Dégrade un label NIfTI avec un seed fixe, retourne des stats."""
    np.random.seed(seed)
    nii = nib.load(src_path)
    data = nii.get_fdata().astype(np.float32)

    n_voxels_before = int((data > 0).sum())
    degraded = apply_degradation_pipeline(data[np.newaxis, ...], degradation_configs)[0]
    n_voxels_after = int((degraded > 0).sum())

    nib.save(nib.Nifti1Image(degraded, nii.affine, nii.header), dst_path)
    return {"before": n_voxels_before, "after": n_voxels_after}


def main():
    parser = argparse.ArgumentParser(
        description="Crée un DatasetXXX_PARSE_Fixed avec labels d'entraînement dégradés fixes"
    )
    parser.add_argument(
        "--dataset_id", type=int, required=True,
        help="ID du dataset cible tel que défini dans fixed_datasets de la config (ex: 101, 102)",
    )
    parser.add_argument(
        "--source_dir", required=True,
        help="Chemin vers Dataset100_PARSE dans nnUNet_raw",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Chemin de sortie (ex: $nnUNet_raw/Dataset101_PARSE_Fixed)",
    )
    parser.add_argument(
        "--config", default="configs/experiment_config.yaml",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed de base — chaque cas reçoit seed + index ordinal",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        print(f"ERREUR: {args.source_dir} n'existe pas.")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    fixed_datasets = config.get("fixed_datasets", {})
    if args.dataset_id not in fixed_datasets:
        available = sorted(fixed_datasets.keys())
        print(f"ERREUR: dataset_id={args.dataset_id} absent de fixed_datasets. Disponibles: {available}")
        sys.exit(1)

    dataset_cfg = fixed_datasets[args.dataset_id]
    degradation_configs = dataset_cfg["degradations"]
    fixed_id = args.dataset_id

    print(f"Source  : {args.source_dir}")
    print(f"Sortie  : {args.output_dir}")
    print(f"Dataset : {dataset_cfg.get('name', f'Dataset{fixed_id:03d}')}")
    print(f"Pipeline de dégradations ({len(degradation_configs)} étape(s)) :")
    for i, d in enumerate(degradation_configs, 1):
        params = {k: v for k, v in d.items() if k != "type"}
        print(f"  {i}. {d['type']} — {params}")
    print(f"Seed    : {args.seed} (+ index par cas)")

    for subdir in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # ── imagesTr : liens symboliques ──
    images_tr = sorted(glob.glob(os.path.join(args.source_dir, "imagesTr", "*.nii.gz")))
    for src in tqdm(images_tr, desc="imagesTr (symlinks)", unit="cas"):
        _symlink(src, os.path.join(args.output_dir, "imagesTr", os.path.basename(src)))
    print(f"  imagesTr : {len(images_tr)} liens créés")

    # ── labelsTr : dégradations fixes par cas ──
    labels_tr = sorted(glob.glob(os.path.join(args.source_dir, "labelsTr", "*.nii.gz")))
    delta_voxels = []
    n_degraded = 0
    with tqdm(enumerate(labels_tr), total=len(labels_tr), desc="labelsTr (dégradation)", unit="cas") as pbar:
        for idx, src in pbar:
            fname = os.path.basename(src)
            dst = os.path.join(args.output_dir, "labelsTr", fname)
            stats = _degrade_label(
                src, dst,
                degradation_configs=degradation_configs,
                seed=args.seed + idx,
            )
            delta = stats["before"] - stats["after"]
            delta_voxels.append(delta)
            if delta != 0:
                n_degraded += 1
            pbar.set_postfix(case=fname, delta=f"{-delta:+d}", degraded=f"{n_degraded}/{idx+1}")

    if delta_voxels:
        print(f"  labelsTr : {len(labels_tr)} labels | {n_degraded} effectivement dégradés | "
              f"Delta moyen = {np.mean(delta_voxels):+.0f} voxels/cas")

    # ── imagesTs : liens symboliques ──
    images_ts = sorted(glob.glob(os.path.join(args.source_dir, "imagesTs", "*.nii.gz")))
    for src in tqdm(images_ts, desc="imagesTs (symlinks)", unit="cas"):
        _symlink(src, os.path.join(args.output_dir, "imagesTs", os.path.basename(src)))
    print(f"  imagesTs : {len(images_ts)} liens créés")

    # ── labelsTs : liens symboliques (évaluation inchangée) ──
    labels_ts = sorted(glob.glob(os.path.join(args.source_dir, "labelsTs", "*.nii.gz")))
    for src in tqdm(labels_ts, desc="labelsTs (symlinks)", unit="cas"):
        _symlink(src, os.path.join(args.output_dir, "labelsTs", os.path.basename(src)))
    print(f"  labelsTs : {len(labels_ts)} liens créés")

    # ── dataset.json ──
    with open(os.path.join(args.source_dir, "dataset.json")) as f:
        dataset_json = json.load(f)

    dataset_json["name"] = dataset_cfg.get("name", f"Dataset{fixed_id:03d}_PARSE_Fixed")
    pipeline_summary = " | ".join(
        f"{d['type']} " + " ".join(f"{k}={v}" for k, v in d.items() if k != "type")
        for d in degradation_configs
    )
    dataset_json["description"] = (
        f"PARSE with fixed degraded training labels — {pipeline_summary} — seed={args.seed}"
    )
    with open(os.path.join(args.output_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDataset créé dans : {args.output_dir}")
    print(f"Prochaine étape  :")
    print(f"  nnUNetv2_plan_and_preprocess -d {fixed_id} --verify_dataset_integrity")


if __name__ == "__main__":
    main()
