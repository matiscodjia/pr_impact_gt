#!/usr/bin/env python3
"""Génère des copies dégradées du dataset pour l'évaluation offline.

Ce script crée des versions dégradées des labels de test (GT-) qui
seront utilisées pour la cross-évaluation. Contrairement aux
dégradations on-the-fly du custom trainer (qui s'appliquent pendant
l'entraînement), celles-ci sont **déterministes et sauvegardées sur
disque** pour garantir la reproductibilité de l'évaluation.

Pour chaque scénario de dégradation, un dossier de labels dégradés
est créé dans ``labelsTs_<scenario_name>/``.

Usage
-----
    python generate_degraded_dataset.py \\
        --dataset_dir /path/to/nnUNet_raw/Dataset100_PARSE \\
        --config configs/experiment_config.yaml \\
        --seed 42
"""

import argparse
import glob
import os

import nibabel as nib
import numpy as np
import yaml
from tqdm import tqdm

from degradations import apply_degradation_pipeline


def degrade_labels(
    labels_dir: str,
    output_dir: str,
    degradation_configs: list,
    seed: int = 42,
) -> None:
    """Génère des labels dégradés à partir d'un dossier de labels propres.

    Parameters
    ----------
    labels_dir : str
        Chemin vers le dossier contenant les labels propres (.nii.gz).
    output_dir : str
        Chemin de sortie pour les labels dégradés.
    degradation_configs : list[dict]
        Pipeline de dégradations formelles — chaque dict contient ``"family"``, ``"r"``, ``"p"``.
    seed : int
        Graine de base (chaque cas reçoit seed + index).
    """
    os.makedirs(output_dir, exist_ok=True)

    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
    print(f"Dégradation de {len(label_files)} labels...")
    for i, d in enumerate(degradation_configs, 1):
        params = {k: v for k, v in d.items() if k != "family"}
        print(f"  {i}. {d['family']} — {params}")
    print(f"  seed: {seed}")

    n_degraded = 0
    with tqdm(enumerate(label_files), total=len(label_files), unit="cas") as pbar:
        for idx, lbl_path in pbar:
            fname = os.path.basename(lbl_path)
            nii = nib.load(lbl_path)
            data = nii.get_fdata().astype(np.float32)
            # Espacement voxel (mm) propagé au générateur pour les familles mm-aware
            spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])

            before = int((data > 0).sum())
            degraded = apply_degradation_pipeline(
                data[np.newaxis, ...], degradation_configs,
                seed_base=seed + idx, spacing=spacing,
            )
            after = int((degraded[0] > 0).sum())
            delta = before - after

            if delta != 0:
                n_degraded += 1
            pbar.set_postfix(case=fname, delta=f"{-delta:+d}", degraded=f"{n_degraded}/{idx+1}")

            out_nii = nib.Nifti1Image(degraded[0], nii.affine, nii.header)
            nib.save(out_nii, os.path.join(output_dir, fname))

    print(f"  -> {len(label_files)} labels sauvegardés ({n_degraded} effectivement dégradés) dans {output_dir}")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Génère des labels dégradés pour l'évaluation"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Chemin vers le dataset nnU-Net (ex: nnUNet_raw/Dataset100_PARSE)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Fichier de configuration YAML",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Graine aléatoire"
    )
    args = parser.parse_args()

    # Charger la config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Évaluation out-of-fold : on dégrade AUSSI les labels d'entraînement (les cas
    # prédits en validation par les folds), pas seulement le test set. Un dossier
    # labels{Tr,Ts}_<scenario> est produit pour chaque split présent.
    splits = [s for s in ("labelsTr", "labelsTs")
              if os.path.isdir(os.path.join(args.dataset_dir, s))]
    if not splits:
        print(f"ERREUR: ni labelsTr ni labelsTs dans {args.dataset_dir}.")
        print("Assurez-vous d'avoir copié les labels lors de la conversion.")
        return

    # Générer un dossier de labels dégradés pour chaque scénario × split
    for scenario in config["evaluation"]["scenarios"]:
        name = scenario["name"]
        deg = scenario.get("degradation")

        if deg is None:
            print(f"\n[{name}] Pas de dégradation (GT*) — skip")
            continue

        for split in splits:
            output_dir = os.path.join(args.dataset_dir, f"{split}_{name}")
            print(f"\n[{name}/{split}] Génération des labels dégradés...")
            degrade_labels(
                labels_dir=os.path.join(args.dataset_dir, split),
                output_dir=output_dir,
                degradation_configs=deg,
                seed=args.seed,
            )

    print("\nGénération terminée.")


if __name__ == "__main__":
    main()
