#!/usr/bin/env python3
"""Orchestrateur principal des expériences nnU-Net PARSE.

Ce script coordonne l'intégralité du pipeline expérimental :
conversion des données, prétraitement, entraînement des modèles
(Star et Minus), prédiction et évaluation. Il peut être lancé
étape par étape ou en mode complet.

Usage
-----
    # Pipeline complet
    python run_experiment.py --data_dir /chemin/PARSE --step all

    # Étapes individuelles
    python run_experiment.py --data_dir /chemin/PARSE --step convert
    python run_experiment.py --step preprocess
    python run_experiment.py --step train_star
    python run_experiment.py --step train_minus
    python run_experiment.py --step predict
    python run_experiment.py --step evaluate

    # Mode debug sur macOS (réduit les époques, fold unique)
    python run_experiment.py --data_dir /chemin/PARSE --step all --debug
"""

import argparse
import os
import platform
import subprocess
import sys

import torch
import yaml


def detect_device() -> str:
    """Détecte le device disponible.

    Returns
    -------
    str
        ``'cuda'``, ``'mps'`` ou ``'cpu'``.
    """
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        name = torch.cuda.get_device_name(0)
        print(f"Device: CUDA — {name} ({vram:.0f} Go)")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Device: MPS (Apple Silicon)")
        return "mps"
    else:
        print("Device: CPU (entraînement sera très lent)")
        return "cpu"


def run_cmd(cmd: list[str], env: dict | None = None) -> int:
    """Lance une commande subprocess avec logging.

    Parameters
    ----------
    cmd : list[str]
        Commande à exécuter.
    env : dict or None
        Variables d'environnement additionnelles.

    Returns
    -------
    int
        Code de retour du processus.
    """
    cmd_str = " ".join(cmd)
    print(f"\n{'─'*60}")
    print(f"CMD: {cmd_str}")
    print(f"{'─'*60}")

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    result = subprocess.run(cmd, env=full_env)

    if result.returncode != 0:
        print(f"ERREUR: commande échouée (code {result.returncode})")
    return result.returncode


def step_convert(config: dict, args: argparse.Namespace) -> None:
    """Convertit le dataset PARSE au format nnU-Net.

    Parameters
    ----------
    config : dict
        Configuration chargée depuis le YAML.
    args : argparse.Namespace
        Arguments de la ligne de commande.
    """
    data_dir = args.data_dir or config["dataset"]["raw_dir"]
    dataset_id = config["dataset"]["id"]

    run_cmd([
        sys.executable, "scripts/convert_parse_to_nnunet.py",
        "--data_dir", data_dir,
        "--dataset_id", str(dataset_id),
    ])


def step_preprocess(config: dict, args: argparse.Namespace) -> None:
    """Lance le prétraitement nnU-Net.

    Parameters
    ----------
    config : dict
        Configuration chargée depuis le YAML.
    args : argparse.Namespace
        Arguments de la ligne de commande.
    """
    dataset_id = config["dataset"]["id"]

    run_cmd([
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "--verify_dataset_integrity",
    ])


def step_train(
    config: dict,
    args: argparse.Namespace,
    trainer: str = "nnUNetTrainer",
    tag: str = "star",
) -> None:
    """Lance l'entraînement nnU-Net pour un trainer donné.

    Parameters
    ----------
    config : dict
        Configuration YAML.
    args : argparse.Namespace
        Arguments CLI.
    trainer : str
        Nom de la classe trainer nnU-Net.
    tag : str
        Tag pour le logging (``'star'`` ou ``'minus'``).
    """
    dataset_id = config["dataset"]["id"]
    device = detect_device()

    folds = config["training"]["folds"]
    configs = config["training"]["configurations"]

    if args.debug:
        folds = [0]
        print(f"[DEBUG] Fold unique: {folds}")

    for nnunet_config in configs:
        for fold in folds:
            cmd = [
                "nnUNetv2_train",
                str(dataset_id),
                nnunet_config,
                str(fold),
                "-tr", trainer,
                "--npz",
                "-device", device,
            ]

            print(f"\n{'='*60}")
            print(f"  Training Model_{tag} | config={nnunet_config} | "
                  f"fold={fold} | trainer={trainer}")
            print(f"{'='*60}")

            rc = run_cmd(cmd)
            if rc != 0:
                print(f"ERREUR: Training fold {fold} échoué. Arrêt.")
                sys.exit(1)


def step_predict(config: dict, args: argparse.Namespace) -> None:
    """Lance les prédictions pour les deux modèles.

    Parameters
    ----------
    config : dict
        Configuration YAML.
    args : argparse.Namespace
        Arguments CLI.
    """
    dataset_id = config["dataset"]["id"]
    dataset_name = f"Dataset{dataset_id:03d}_PARSE"
    nnunet_raw = os.environ["nnUNet_raw"]
    images_ts = os.path.join(nnunet_raw, dataset_name, "imagesTs")

    if not os.path.isdir(images_ts):
        print(f"ERREUR: {images_ts} n'existe pas. Pas d'images de test.")
        return

    nnunet_config = config["training"]["configurations"][0]
    device = detect_device()

    for trainer, tag in [
        ("nnUNetTrainer", "star"),
        ("nnUNetTrainerDegraded", "minus"),
    ]:
        output_dir = os.path.join("predictions", f"model_{tag}")
        os.makedirs(output_dir, exist_ok=True)

        fold_str = "0" if args.debug else "0 1 2 3 4"

        cmd = [
            "nnUNetv2_predict",
            "-i", images_ts,
            "-o", output_dir,
            "-d", str(dataset_id),
            "-c", nnunet_config,
            "-tr", trainer,
            "-f", *fold_str.split(),
            "-device", device,
        ]

        print(f"\n Prediction Model_{tag}...")
        run_cmd(cmd)


def step_evaluate(config: dict, args: argparse.Namespace) -> None:
    """Lance la cross-évaluation.

    Parameters
    ----------
    config : dict
        Configuration YAML.
    args : argparse.Namespace
        Arguments CLI.
    """
    run_cmd([
        sys.executable, "scripts/cross_evaluate.py",
        "--config", args.config,
    ])


def main():
    """Point d'entrée principal de l'orchestrateur."""
    parser = argparse.ArgumentParser(
        description="Orchestrateur des expériences nnU-Net PARSE"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Chemin vers le dataset PARSE source (requis pour 'convert')",
    )
    parser.add_argument(
        "--step", type=str, default="all",
        choices=[
            "all", "convert", "preprocess", "train_star",
            "train_minus", "predict", "evaluate",
        ],
        help="Étape à exécuter",
    )
    parser.add_argument(
        "--config", type=str, default="configs/experiment_config.yaml",
        help="Fichier de configuration YAML",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Mode debug : 1 fold, vérification rapide du pipeline",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    steps_all = [
        "convert", "preprocess", "train_star",
        "train_minus", "predict", "evaluate",
    ]
    steps_to_run = steps_all if args.step == "all" else [args.step]

    print(f"\n{'#'*60}")
    print(f"  nnU-Net PARSE Experiment Pipeline")
    print(f"  Steps: {', '.join(steps_to_run)}")
    print(f"  Debug: {args.debug}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"{'#'*60}")

    for step in steps_to_run:
        print(f"\n\n{'▶'*3} STEP: {step}")

        if step == "convert":
            step_convert(config, args)
        elif step == "preprocess":
            step_preprocess(config, args)
        elif step == "train_star":
            step_train(config, args, trainer="nnUNetTrainer", tag="star")
        elif step == "train_minus":
            step_train(
                config, args,
                trainer="nnUNetTrainerDegraded", tag="minus",
            )
        elif step == "predict":
            step_predict(config, args)
        elif step == "evaluate":
            step_evaluate(config, args)

    print(f"\n\n{'='*60}")
    print("  Pipeline terminé.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
