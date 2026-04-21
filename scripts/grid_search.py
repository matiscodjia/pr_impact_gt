#!/usr/bin/env python3
"""Grid search des paramètres de dégradation avec nnU-Net.

Génère dynamiquement des custom trainers nnU-Net pour chaque
combinaison de paramètres de dégradation, les installe dans le
dossier nnUNetTrainer, puis lance l'entraînement et l'évaluation.

Phases
------
1. **Morpho seule** : exploration de ``prob × max_radius``
2. **Omission seule** : exploration de ``prob × min_size``
3. **Combinaisons** : top-N des phases 1 et 2 combinés

Usage
-----
    # Phase 1 uniquement
    python grid_search.py --config configs/experiment_config.yaml --phase 1

    # Toutes les phases
    python grid_search.py --config configs/experiment_config.yaml --phase all

    # Mode debug (5 époques, 1 fold)
    python grid_search.py --config configs/experiment_config.yaml --phase 1 --debug
"""

import argparse
import itertools
import os
import subprocess
import sys
import textwrap
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import yaml


def get_nnunet_trainer_dir() -> str:
    """Retourne le chemin du dossier nnUNetTrainer dans l'installation.

    Returns
    -------
    str
        Chemin absolu du dossier contenant les trainers nnU-Net.

    Raises
    ------
    ImportError
        Si nnunetv2 n'est pas installé.
    """
    import nnunetv2
    return os.path.join(
        nnunetv2.__path__[0], "training", "nnUNetTrainer"
    )


def generate_trainer_class(
    class_name: str,
    morpho_prob: float,
    morpho_max_radius: int,
    omission_prob: float,
    omission_min_size: int,
) -> str:
    """Génère le code Python d'un custom trainer paramétré.

    Parameters
    ----------
    class_name : str
        Nom de la classe trainer.
    morpho_prob : float
        Probabilité de dégradation morphologique.
    morpho_max_radius : int
        Rayon maximum pour la morphologie.
    omission_prob : float
        Probabilité d'omission.
    omission_min_size : int
        Taille seuil pour l'omission.

    Returns
    -------
    str
        Code source Python de la classe.
    """
    return textwrap.dedent(f'''\
        """Auto-generated trainer for grid search: {class_name}."""
        import torch
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerDegraded import (
            nnUNetTrainerDegraded,
        )

        class {class_name}(nnUNetTrainerDegraded):
            """Grid search variant: mP={morpho_prob} mR={morpho_max_radius} """
            """oP={omission_prob} oS={omission_min_size}."""

            def __init__(self, plans, configuration, fold, dataset_json,
                         unpack_dataset=True, device=torch.device("cuda")):
                super().__init__(
                    plans, configuration, fold, dataset_json,
                    unpack_dataset, device,
                )
                self.morpho_prob = {morpho_prob}
                self.morpho_max_radius = {morpho_max_radius}
                self.omission_prob = {omission_prob}
                self.omission_min_size = {omission_min_size}
    ''')


def make_class_name(
    morpho_prob: float,
    morpho_max_radius: int,
    omission_prob: float,
    omission_min_size: int,
) -> str:
    """Génère un nom de classe unique pour un jeu de paramètres.

    Parameters
    ----------
    morpho_prob : float
        Probabilité morpho.
    morpho_max_radius : int
        Rayon morpho.
    omission_prob : float
        Probabilité omission.
    omission_min_size : int
        Taille seuil omission.

    Returns
    -------
    str
        Nom de classe valide Python (ex: ``nnUNetTrainerDeg_mP03_mR3_oP02_oS150``).
    """
    mp = str(morpho_prob).replace(".", "")
    op = str(omission_prob).replace(".", "")
    return f"nnUNetTrainerDeg_mP{mp}_mR{morpho_max_radius}_oP{op}_oS{omission_min_size}"


def install_trainer(class_name: str, code: str) -> str:
    """Installe un trainer généré dans le dossier nnUNetTrainer.

    Parameters
    ----------
    class_name : str
        Nom de la classe.
    code : str
        Code source de la classe.

    Returns
    -------
    str
        Chemin du fichier installé.
    """
    trainer_dir = get_nnunet_trainer_dir()
    filepath = os.path.join(trainer_dir, f"{class_name}.py")

    with open(filepath, "w") as f:
        f.write(code)

    print(f"  Trainer installé: {filepath}")
    return filepath


def run_training(
    dataset_id: int,
    config_name: str,
    fold: int,
    trainer_name: str,
    device: str,
) -> int:
    """Lance l'entraînement nnU-Net pour un trainer.

    Parameters
    ----------
    dataset_id : int
        ID du dataset nnU-Net.
    config_name : str
        Configuration nnU-Net (ex: ``3d_fullres``).
    fold : int
        Fold de cross-validation.
    trainer_name : str
        Nom du trainer.
    device : str
        Device (``cuda``, ``mps``, ``cpu``).

    Returns
    -------
    int
        Code de retour du processus.
    """
    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        config_name,
        str(fold),
        "-tr", trainer_name,
        "--npz",
        "-device", device,
    ]
    print(f"  CMD: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def run_prediction(
    dataset_id: int,
    config_name: str,
    fold: int,
    trainer_name: str,
    input_dir: str,
    output_dir: str,
    device: str,
) -> int:
    """Lance la prédiction nnU-Net.

    Parameters
    ----------
    dataset_id : int
        ID du dataset.
    config_name : str
        Configuration nnU-Net.
    fold : int
        Fold à utiliser.
    trainer_name : str
        Nom du trainer.
    input_dir : str
        Dossier des images de test.
    output_dir : str
        Dossier de sortie des prédictions.
    device : str
        Device.

    Returns
    -------
    int
        Code de retour.
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", str(dataset_id),
        "-c", config_name,
        "-tr", trainer_name,
        "-f", str(fold),
        "-device", device,
    ]
    print(f"  CMD: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def evaluate_from_folders(pred_dir: str, gt_dir: str) -> dict:
    """Évalue les prédictions contre une GT.

    Parameters
    ----------
    pred_dir : str
        Dossier des prédictions.
    gt_dir : str
        Dossier des labels de référence.

    Returns
    -------
    dict
        Dictionnaire avec les métriques moyennes.
    """
    # Import local pour éviter les dépendances circulaires
    from cross_evaluate import evaluate_predictions

    results = evaluate_predictions(pred_dir, gt_dir, "temp", "temp")
    if not results:
        return {"dice": 0.0, "hd95": np.inf}

    dices = [r["dice"] for r in results]
    hd95s = [r["hd95"] for r in results if r["hd95"] < np.inf]

    return {
        "dice": np.mean(dices),
        "hd95": np.mean(hd95s) if hd95s else np.inf,
    }


def run_grid_search(config: dict, phase: int, debug: bool = False) -> pd.DataFrame:
    """Exécute une phase du grid search.

    Parameters
    ----------
    config : dict
        Configuration YAML complète.
    phase : int
        Numéro de la phase (1, 2 ou 3).
    debug : bool
        Si True, mode debug rapide.

    Returns
    -------
    pd.DataFrame
        Résultats de la phase.
    """
    dataset_id = config["dataset"]["id"]
    dataset_name = f"Dataset{dataset_id:03d}_PARSE"
    nnunet_raw = os.environ.get("nnUNet_raw", "nnUNet_data/nnUNet_raw")
    dataset_dir = os.path.join(nnunet_raw, dataset_name)

    gs_config = config["grid_search"]
    fold = gs_config["fold"]
    nnunet_config = config["training"]["configurations"][0]

    # Détection device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Chemins GT
    gt_star_dir = os.path.join(dataset_dir, "labelsTs")
    images_ts = os.path.join(dataset_dir, "imagesTs")

    results = []

    # ── Génération des configs ──
    if phase == 1:
        configs = []
        for mp in gs_config["morpho_grid"]["prob"]:
            for mr in gs_config["morpho_grid"]["max_radius"]:
                configs.append({
                    "morpho_prob": mp,
                    "morpho_max_radius": mr,
                    "omission_prob": 0.0,
                    "omission_min_size": 100,
                })
    elif phase == 2:
        configs = []
        for op in gs_config["omission_grid"]["prob"]:
            for os_ in gs_config["omission_grid"]["min_size"]:
                configs.append({
                    "morpho_prob": 0.0,
                    "morpho_max_radius": 1,
                    "omission_prob": op,
                    "omission_min_size": os_,
                })
    else:
        # Phase 3 : charger les top-N des phases 1 et 2
        print("Phase 3 : chargement des résultats précédents...")
        csv_path = "results/grid_search_results.csv"
        if not os.path.exists(csv_path):
            print("ERREUR: lancez d'abord les phases 1 et 2.")
            return pd.DataFrame()

        prev_df = pd.read_csv(csv_path)
        
        # S'assurer que les colonnes nécessaires sont là
        required_cols = ["phase", "dice_star", "morpho_prob", "morpho_max_radius", "omission_prob", "omission_min_size"]
        if not all(col in prev_df.columns for col in required_cols):
            print(f"ERREUR: Colonnes manquantes dans {csv_path}. Attendues: {required_cols}")
            return pd.DataFrame()

        top_n = gs_config.get("top_n_combos", 3)

        # Top morpho
        phase1 = prev_df[prev_df["phase"] == 1].nlargest(top_n, "dice_star")
        # Top omission
        phase2 = prev_df[prev_df["phase"] == 2].nlargest(top_n, "dice_star")

        if phase1.empty or phase2.empty:
            print("ERREUR: Phase 1 ou Phase 2 n'a pas de résultats valides.")
            return pd.DataFrame()

        configs = []
        for _, m_row in phase1.iterrows():
            for _, o_row in phase2.iterrows():
                configs.append({
                    "morpho_prob": m_row["morpho_prob"],
                    "morpho_max_radius": int(m_row["morpho_max_radius"]),
                    "omission_prob": o_row["omission_prob"],
                    "omission_min_size": int(o_row["omission_min_size"]),
                })

    total = len(configs)
    phase_names = {1: "MORPHO SEULE", 2: "OMISSION SEULE", 3: "COMBINAISONS"}

    print(f"\n{'#'*60}")
    print(f"  PHASE {phase} — {phase_names[phase]} — {total} configurations")
    print(f"{'#'*60}")

    for idx, cfg in enumerate(configs):
        class_name = make_class_name(**cfg)
        print(f"\n[{idx+1}/{total}] {class_name}")

        # Générer et installer le trainer
        code = generate_trainer_class(class_name, **cfg)
        install_trainer(class_name, code)

        start = time.time()

        # Entraînement
        rc = run_training(dataset_id, nnunet_config, fold, class_name, device)
        if rc != 0:
            print(f"  ERREUR: entraînement échoué, skip")
            continue

        # Prédiction
        pred_dir = os.path.join("predictions", "grid_search", class_name)
        rc = run_prediction(
            dataset_id, nnunet_config, fold, class_name,
            images_ts, pred_dir, device,
        )
        if rc != 0:
            print(f"  ERREUR: prédiction échouée, skip")
            continue

        # Évaluation sur GT*
        metrics_star = evaluate_from_folders(pred_dir, gt_star_dir)

        elapsed = time.time() - start

        result = {
            "phase": phase,
            "trainer": class_name,
            "morpho_prob": cfg["morpho_prob"],
            "morpho_max_radius": cfg["morpho_max_radius"],
            "omission_prob": cfg["omission_prob"],
            "omission_min_size": cfg["omission_min_size"],
            "dice_star": metrics_star["dice"],
            "hd95_star": metrics_star["hd95"],
            "time_s": int(elapsed),
        }
        results.append(result)

        print(
            f"  Dice(GT*)={metrics_star['dice']:.4f} | "
            f"Time={timedelta(seconds=int(elapsed))}"
        )

    return pd.DataFrame(results)


def main():
    """Point d'entrée principal du grid search."""
    parser = argparse.ArgumentParser(
        description="Grid search des paramètres de dégradation"
    )
    parser.add_argument(
        "--config", type=str, default="configs/experiment_config.yaml",
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["1", "2", "3", "all"],
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Mode debug (5 époques)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs("results", exist_ok=True)
    csv_path = "results/grid_search_results.csv"

    phases = [1, 2, 3] if args.phase == "all" else [int(args.phase)]

    for phase in phases:
        df = run_grid_search(config, phase, debug=args.debug)

        if not df.empty:
            # Append au CSV existant
            if os.path.exists(csv_path):
                existing = pd.read_csv(csv_path)
                df = pd.concat([existing, df], ignore_index=True)

            df.to_csv(csv_path, index=False)
            print(f"\nRésultats sauvegardés : {csv_path}")

    # Résumé final
    if os.path.exists(csv_path):
        final_df = pd.read_csv(csv_path)
        print(f"\n{'='*60}")
        print("RÉSUMÉ GRID SEARCH")
        print(f"{'='*60}")

        for p in sorted(final_df["phase"].unique()):
            phase_df = final_df[final_df["phase"] == p]
            best = phase_df.loc[phase_df["dice_star"].idxmax()]
            print(
                f"\n  Phase {p} — Meilleur Dice(GT*): {best['dice_star']:.4f}"
                f" — {best['trainer']}"
            )

        overall_best = final_df.loc[final_df["dice_star"].idxmax()]
        print(
            f"\n  ★ Meilleur global: {overall_best['dice_star']:.4f}"
            f" — {overall_best['trainer']}"
        )


if __name__ == "__main__":
    main()
