#!/usr/bin/env python3
"""Orchestrateur principal des expériences nnU-Net PARSE.

Ce script coordonne l'intégralité du pipeline expérimental :
conversion, prétraitement, entraînement (Star, Minus_Stoch, Minus_Fixed),
prédiction et évaluation.

Usage
-----
    # Pipeline complet
    python run_experiment.py --data_dir /chemin/PARSE --step all

    # Étapes individuelles
    python run_experiment.py --data_dir /chemin/PARSE --step convert
    python run_experiment.py --step preprocess
    python run_experiment.py --step train_star
    python run_experiment.py --step train_minus_stoch
    python run_experiment.py --step create_fixed_dataset
    python run_experiment.py --step preprocess_fixed
    python run_experiment.py --step train_minus_fixed
    python run_experiment.py --step predict
    python run_experiment.py --step evaluate

    # Mode debug (1 fold, pipeline rapide)
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
    data_dir = args.data_dir or config["dataset"]["raw_dir"]
    run_cmd([
        sys.executable, "scripts/convert_parse_to_nnunet.py",
        "--data_dir", data_dir,
        "--dataset_id", str(config["dataset"]["id"]),
    ])


def step_preprocess(config: dict, args: argparse.Namespace, dataset_id: int | None = None) -> None:
    did = dataset_id or config["dataset"]["id"]
    run_cmd([
        "nnUNetv2_plan_and_preprocess",
        "-d", str(did),
        "--verify_dataset_integrity",
    ])


def step_train(
    config: dict,
    args: argparse.Namespace,
    trainer: str = "nnUNetTrainer",
    tag: str = "star",
    dataset_id: int | None = None,
) -> None:
    did = dataset_id or config["dataset"]["id"]
    device = detect_device()
    folds = [0] if args.debug else config["training"]["folds"]

    for nnunet_config in config["training"]["configurations"]:
        for fold in folds:
            print(f"\n{'='*60}")
            print(f"  Training Model_{tag} | config={nnunet_config} | "
                  f"fold={fold} | trainer={trainer} | dataset={did}")
            print(f"{'='*60}")

            rc = run_cmd([
                "nnUNetv2_train",
                str(did), nnunet_config, str(fold),
                "-tr", trainer, "--npz", "-device", device,
            ])
            if rc != 0:
                print(f"ERREUR: fold {fold} échoué. Arrêt.")
                sys.exit(1)


def step_create_fixed_dataset(config: dict, args: argparse.Namespace) -> None:
    nnunet_raw = os.environ.get("nnUNet_raw", "nnUNet_data/nnUNet_raw")
    dataset_id = config["dataset"]["id"]
    fixed_id = config["dataset"]["fixed_id"]
    dataset_name = f"Dataset{dataset_id:03d}_PARSE"
    fixed_name = f"Dataset{fixed_id:03d}_PARSE_Fixed"

    source_dir = os.path.join(nnunet_raw, dataset_name)
    output_dir = os.path.join(nnunet_raw, fixed_name)

    run_cmd([
        sys.executable, "scripts/create_fixed_degraded_dataset.py",
        "--source_dir", source_dir,
        "--output_dir", output_dir,
        "--config", args.config,
    ])


def step_predict(config: dict, args: argparse.Namespace) -> None:
    dataset_id = config["dataset"]["id"]
    fixed_id = config["dataset"]["fixed_id"]
    dataset_name = f"Dataset{dataset_id:03d}_PARSE"
    nnunet_raw = os.environ.get("nnUNet_raw", "nnUNet_data/nnUNet_raw")
    images_ts = os.path.join(nnunet_raw, dataset_name, "imagesTs")

    if not os.path.isdir(images_ts):
        print(f"ERREUR: {images_ts} n'existe pas.")
        return

    nnunet_config = config["training"]["configurations"][0]
    device = detect_device()
    fold_args = ["0"] if args.debug else [str(f) for f in config["training"]["folds"]]

    models_to_predict = [
        ("nnUNetTrainer",        dataset_id,  "model_star"),
        ("nnUNetTrainerDegraded", dataset_id,  "model_minus_stoch"),
        ("nnUNetTrainer",        fixed_id,    "model_minus_fixed"),
    ]

    for trainer, did, tag in models_to_predict:
        output_dir = os.path.join("predictions", tag)
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "nnUNetv2_predict",
            "-i", images_ts,
            "-o", output_dir,
            "-d", str(did),
            "-c", nnunet_config,
            "-tr", trainer,
            "-f", *fold_args,
            "-device", device,
        ]
        print(f"\n Prédiction {tag} (dataset={did}, trainer={trainer})...")
        run_cmd(cmd)


def step_evaluate(config: dict, args: argparse.Namespace) -> None:
    run_cmd([
        sys.executable, "scripts/cross_evaluate.py",
        "--config", args.config,
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrateur des expériences nnU-Net PARSE"
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument(
        "--step", type=str, default="all",
        choices=[
            "all", "convert", "preprocess",
            "train_star", "train_minus_stoch",
            "create_fixed_dataset", "preprocess_fixed", "train_minus_fixed",
            "predict", "evaluate",
        ],
    )
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    steps_all = [
        "convert", "preprocess",
        "train_star", "train_minus_stoch",
        "create_fixed_dataset", "preprocess_fixed", "train_minus_fixed",
        "predict", "evaluate",
    ]
    steps_to_run = steps_all if args.step == "all" else [args.step]

    print(f"\n{'#'*60}")
    print(f"  nnU-Net PARSE Experiment Pipeline")
    print(f"  Steps : {', '.join(steps_to_run)}")
    print(f"  Debug : {args.debug}")
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
        elif step == "train_minus_stoch":
            step_train(config, args, trainer="nnUNetTrainerDegraded", tag="minus_stoch")
        elif step == "create_fixed_dataset":
            step_create_fixed_dataset(config, args)
        elif step == "preprocess_fixed":
            step_preprocess(config, args, dataset_id=config["dataset"]["fixed_id"])
        elif step == "train_minus_fixed":
            step_train(
                config, args,
                trainer="nnUNetTrainer", tag="minus_fixed",
                dataset_id=config["dataset"]["fixed_id"],
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
