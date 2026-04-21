#!/usr/bin/env python3
"""Convertit le dataset PARSE au format nnU-Net v2.

Ce script prend les données PARSE organisées en dossiers train/val/test
et les restructure au format attendu par nnU-Net v2 :

    Dataset100_PARSE/
    ├── dataset.json
    ├── imagesTr/
    │   ├── PARSE_0001_0000.nii.gz
    │   └── ...
    ├── labelsTr/
    │   ├── PARSE_0001.nii.gz
    │   └── ...
    └── imagesTs/
        ├── PARSE_0051_0000.nii.gz
        └── ...

Le suffixe ``_0000`` indique le canal (modalité unique : CT).

Usage
-----
    python convert_parse_to_nnunet.py \\
        --data_dir /chemin/vers/PARSE \\
        --dataset_id 100

Notes
-----
- Les images PARSE doivent être au format NIfTI (.nii.gz).
- Le script gère deux layouts sources :
    1. ``data_dir/{train,val,test}/{images,labels}/`` (votre structure actuelle)
    2. ``data_dir/{images,labels}/`` (flat, avec split automatique)
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np


def _find_nifti(directory: str) -> list[str]:
    """Trouve tous les fichiers NIfTI dans un répertoire.

    Parameters
    ----------
    directory : str
        Chemin du répertoire à scanner.

    Returns
    -------
    list[str]
        Liste triée des chemins absolus vers les fichiers ``.nii.gz``.
    """
    pattern = os.path.join(directory, "*.nii.gz")
    return sorted(glob.glob(pattern))


def _copy_case(
    img_src: str,
    lbl_src: str,
    images_dst: str,
    labels_dst: str,
    case_id: str,
) -> None:
    """Copie une paire image/label au format nnU-Net.

    Parameters
    ----------
    img_src : str
        Chemin source de l'image.
    lbl_src : str
        Chemin source du label.
    images_dst : str
        Répertoire de destination pour les images.
    labels_dst : str
        Répertoire de destination pour les labels.
    case_id : str
        Identifiant du cas (ex: ``PARSE_0001``).
    """
    # Image : ajout du suffixe _0000 (canal unique CT)
    img_dst = os.path.join(images_dst, f"{case_id}_0000.nii.gz")
    lbl_dst = os.path.join(labels_dst, f"{case_id}.nii.gz")

    shutil.copy2(img_src, img_dst)
    shutil.copy2(lbl_src, lbl_dst)


def convert_parse_to_nnunet(
    data_dir: str,
    dataset_id: int = 100,
    nnunet_raw: str | None = None,
) -> str:
    """Convertit le dataset PARSE au format nnU-Net v2.

    Parameters
    ----------
    data_dir : str
        Chemin vers le dataset PARSE source.
    dataset_id : int
        Identifiant numérique du dataset nnU-Net (par défaut 100).
    nnunet_raw : str or None
        Chemin vers le dossier nnUNet_raw. Si None, utilise la variable
        d'environnement ``nnUNet_raw``.

    Returns
    -------
    str
        Chemin vers le dataset nnU-Net créé.

    Raises
    ------
    ValueError
        Si aucune image n'est trouvée ou si le nombre d'images et de
        labels ne correspond pas.
    EnvironmentError
        Si ``nnUNet_raw`` n'est pas défini.
    """
    if nnunet_raw is None:
        nnunet_raw = os.environ.get("nnUNet_raw")
        if nnunet_raw is None:
            raise EnvironmentError(
                "La variable d'environnement nnUNet_raw n'est pas définie. "
                "Lancez setup_env.sh ou exportez-la manuellement."
            )

    dataset_name = f"Dataset{dataset_id:03d}_PARSE"
    dataset_dir = os.path.join(nnunet_raw, dataset_name)

    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")
    images_ts = os.path.join(dataset_dir, "imagesTs")

    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)
    os.makedirs(images_ts, exist_ok=True)

    # ── Détection du layout source ──
    has_splits = os.path.isdir(os.path.join(data_dir, "train", "images"))

    case_counter = 1

    if has_splits:
        print("Layout détecté : train/val/test subdirectories")

        # Train + Val → imagesTr / labelsTr (nnU-Net fait sa propre CV)
        for split in ["train", "val"]:
            split_img_dir = os.path.join(data_dir, split, "images")
            split_lbl_dir = os.path.join(data_dir, split, "labels")

            if not os.path.isdir(split_img_dir):
                print(f"  [SKIP] {split_img_dir} n'existe pas")
                continue

            imgs = _find_nifti(split_img_dir)
            lbls = _find_nifti(split_lbl_dir)

            if len(imgs) != len(lbls):
                raise ValueError(
                    f"Mismatch dans {split}: {len(imgs)} images vs "
                    f"{len(lbls)} labels"
                )

            for img, lbl in zip(imgs, lbls):
                case_id = f"PARSE_{case_counter:04d}"
                _copy_case(img, lbl, images_tr, labels_tr, case_id)
                case_counter += 1

            print(f"  {split}: {len(imgs)} cas copiés")

        num_training = case_counter - 1

        # Test → imagesTs (pas de labels pour le test nnU-Net)
        test_img_dir = os.path.join(data_dir, "test", "images")
        if os.path.isdir(test_img_dir):
            test_imgs = _find_nifti(test_img_dir)
            for img in test_imgs:
                case_id = f"PARSE_{case_counter:04d}"
                dst = os.path.join(images_ts, f"{case_id}_0000.nii.gz")
                shutil.copy2(img, dst)
                case_counter += 1
            print(f"  test: {len(test_imgs)} cas copiés dans imagesTs")

            # On copie aussi les labels de test pour l'évaluation manuelle
            test_lbl_dir = os.path.join(data_dir, "test", "labels")
            if os.path.isdir(test_lbl_dir):
                labels_ts = os.path.join(dataset_dir, "labelsTs")
                os.makedirs(labels_ts, exist_ok=True)
                test_lbls = _find_nifti(test_lbl_dir)
                ts_counter = num_training + 1
                for lbl in test_lbls:
                    case_id = f"PARSE_{ts_counter:04d}"
                    dst = os.path.join(labels_ts, f"{case_id}.nii.gz")
                    shutil.copy2(lbl, dst)
                    ts_counter += 1
                print(f"  test labels: {len(test_lbls)} cas copiés dans labelsTs")
    else:
        print("Layout détecté : flat (images/ + labels/)")
        imgs = _find_nifti(os.path.join(data_dir, "images"))
        lbls = _find_nifti(os.path.join(data_dir, "labels"))

        if len(imgs) == 0:
            raise ValueError(f"Aucune image trouvée dans {data_dir}/images")
        if len(imgs) != len(lbls):
            raise ValueError(
                f"Mismatch: {len(imgs)} images vs {len(lbls)} labels"
            )

        for img, lbl in zip(imgs, lbls):
            case_id = f"PARSE_{case_counter:04d}"
            _copy_case(img, lbl, images_tr, labels_tr, case_id)
            case_counter += 1

        num_training = case_counter - 1
        print(f"  {num_training} cas copiés dans imagesTr/labelsTr")

    # ── Génération du dataset.json ──
    dataset_json = {
        "channel_names": {
            "0": "CT",
        },
        "labels": {
            "background": 0,
            "pulmonary_artery": 1,
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }

    json_path = os.path.join(dataset_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"\n{'='*60}")
    print(f"Dataset nnU-Net créé : {dataset_dir}")
    print(f"  Cas d'entraînement : {num_training}")
    print(f"  dataset.json       : {json_path}")
    print(f"{'='*60}")

    return dataset_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convertit le dataset PARSE au format nnU-Net v2"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Chemin vers le dataset PARSE source",
    )
    parser.add_argument(
        "--dataset_id",
        type=int,
        default=100,
        help="ID du dataset nnU-Net (défaut: 100)",
    )
    parser.add_argument(
        "--nnunet_raw",
        type=str,
        default=None,
        help="Chemin nnUNet_raw (défaut: variable d'env)",
    )
    args = parser.parse_args()

    convert_parse_to_nnunet(
        data_dir=args.data_dir,
        dataset_id=args.dataset_id,
        nnunet_raw=args.nnunet_raw,
    )
