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

from degradations import (
    apply_combined_degradation,
    apply_morpho_degradation,
    apply_omission_degradation,
)


def degrade_labels(
    labels_dir: str,
    output_dir: str,
    morpho_prob: float = 1.0,
    morpho_max_radius: int = 3,
    omission_prob: float = 0.0,
    omission_min_size: int = 150,
    seed: int = 42,
) -> None:
    """Génère des labels dégradés à partir d'un dossier de labels propres.

    Parameters
    ----------
    labels_dir : str
        Chemin vers le dossier contenant les labels propres (.nii.gz).
    output_dir : str
        Chemin de sortie pour les labels dégradés.
    morpho_prob : float
        Probabilité de dégradation morphologique (1.0 = toujours).
    morpho_max_radius : int
        Rayon max pour l'érosion/dilatation.
    omission_prob : float
        Probabilité d'omission de composantes.
    omission_min_size : int
        Taille seuil pour les composantes à omettre.
    seed : int
        Graine aléatoire pour la reproductibilité.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
    print(f"Dégradation de {len(label_files)} labels...")
    print(f"  morpho: prob={morpho_prob}, max_radius={morpho_max_radius}")
    print(f"  omission: prob={omission_prob}, min_size={omission_min_size}")
    print(f"  seed: {seed}")

    for lbl_path in label_files:
        fname = os.path.basename(lbl_path)
        nii = nib.load(lbl_path)
        data = nii.get_fdata().astype(np.float32)

        # Ajouter une dimension canal pour la compatibilité
        data_ch = data[np.newaxis, ...]

        # Appliquer les dégradations
        degraded = apply_combined_degradation(
            data_ch,
            morpho_prob=morpho_prob,
            morpho_max_radius=morpho_max_radius,
            omission_prob=omission_prob,
            omission_min_size=omission_min_size,
        )

        # Retirer la dimension canal
        degraded_3d = degraded[0]

        # Sauvegarder avec le même header/affine
        out_nii = nib.Nifti1Image(degraded_3d, nii.affine, nii.header)
        out_path = os.path.join(output_dir, fname)
        nib.save(out_nii, out_path)

    print(f"  -> {len(label_files)} labels dégradés sauvegardés dans {output_dir}")


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

    # Dossier des labels de test
    labels_ts = os.path.join(args.dataset_dir, "labelsTs")
    if not os.path.isdir(labels_ts):
        print(f"ERREUR: {labels_ts} n'existe pas.")
        print("Assurez-vous d'avoir copié les labels de test lors de la conversion.")
        return

    # Générer un dossier de labels dégradés pour chaque scénario
    for scenario in config["evaluation"]["scenarios"]:
        name = scenario["name"]
        deg = scenario.get("degradation")

        if deg is None:
            print(f"\n[{name}] Pas de dégradation (GT*) — skip")
            continue

        morpho = deg.get("morpho", {})
        omission = deg.get("omission", {})

        output_dir = os.path.join(
            args.dataset_dir, f"labelsTs_{name}"
        )

        print(f"\n[{name}] Génération des labels dégradés...")
        degrade_labels(
            labels_dir=labels_ts,
            output_dir=output_dir,
            morpho_prob=morpho.get("prob", 0.0),
            morpho_max_radius=morpho.get("max_radius", 1),
            omission_prob=omission.get("prob", 0.0),
            omission_min_size=omission.get("min_size", 100),
            seed=args.seed,
        )

    print("\nGénération terminée.")


if __name__ == "__main__":
    main()
