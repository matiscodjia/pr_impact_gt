"""Custom nnU-Net Trainer avec dégradations on-the-fly des labels.

Ce trainer hérite de ``nnUNetTrainer`` et injecte des dégradations
stochastiques (morphologiques et/ou omission) sur les masques de
segmentation à chaque itération d'entraînement, **avant** le calcul
de la loss.

L'objectif est d'entraîner un modèle sur des annotations
imparfaites (Model_Minus) qui peut être comparé à un modèle entraîné
sur des annotations parfaites (Model_Star, entraîné avec le trainer
standard ``nnUNetTrainer``).

Usage
-----
Le trainer doit être placé dans le dossier
``nnunetv2/training/nnUNetTrainer/`` pour que nnU-Net le trouve
automatiquement via ``-tr nnUNetTrainerDegraded``.

Commandes d'entraînement::

    # Model_Minus avec dégradations par défaut
    nnUNetv2_train 100 3d_fullres 0 -tr nnUNetTrainerDegraded

    # Model_Star (baseline sans dégradation)
    nnUNetv2_train 100 3d_fullres 0

Les variantes pré-configurées suivantes sont également disponibles :

- ``nnUNetTrainerDegradedMorphoOnly`` : morpho seule (prob=0.3, r=3)
- ``nnUNetTrainerDegradedOmissionOnly`` : omission seule (prob=0.2, s=150)
- ``nnUNetTrainerDegradedMild`` : dégradation légère
- ``nnUNetTrainerDegradedSevere`` : dégradation forte
- ``nnUNetTrainerDegraded_mP{}_mR{}_oP{}_oS{}`` : via grid search

Notes
-----
Les dégradations sont appliquées directement sur le tenseur ``target``
dans la méthode ``train_step``, qui est le point d'injection le plus
propre car il intervient après le data augmentation de nnU-Net mais
avant le forward pass. Les dégradations sont appliquées sur le label
*avant* le deep supervision downsampling, car ``train_step`` reçoit
les targets déjà downsamplés. On opère donc sur la résolution la plus
haute (index 0 dans la liste de deep supervision).
"""

import numpy as np
import torch
import yaml
import os
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    generate_binary_structure,
    label as label_cc,
)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerDebug(nnUNetTrainer):
    """Trainer pour debug rapide (2 époques)."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

class nnUNetTrainerDegraded(nnUNetTrainer):
    """Trainer nnU-Net avec dégradations morpho + omission on-the-fly."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, device
        )
        # Pipeline de dégradations par défaut (morpho + omission)
        self.degradation_pipeline = [
            {"type": "morpho",   "prob": 0.3, "max_radius": 3},
            {"type": "omission", "prob": 0.2, "min_size": 150},
        ]

        config_path = os.path.join(os.getcwd(), "configs", "experiment_config.yaml")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    if "stochastic_degradations" in config:
                        self.degradation_pipeline = config["stochastic_degradations"]
                    if os.environ.get("DEBUG_PIPELINE") == "1":
                        self.num_epochs = 2
                        self.print_to_log_file("DEBUG MODE: num_epochs set to 2")
            except Exception as e:
                self.print_to_log_file(f"WARNING: Could not load config: {e}")

        pipeline_summary = ", ".join(
            f"{d['type']}(" + " ".join(f"{k}={v}" for k, v in d.items() if k != "type") + ")"
            for d in self.degradation_pipeline
        )
        self.print_to_log_file(
            f"\n{'='*60}\n"
            f"nnUNetTrainerDegraded initialized\n"
            f"  pipeline: {pipeline_summary}\n"
            f"{'='*60}\n"
        )

    def _apply_morpho(self, result: np.ndarray, cfg: dict) -> np.ndarray:
        if np.random.rand() >= cfg["prob"]:
            return result
        binary_mask = result > 0
        if not binary_mask.any():
            return result
        radius = np.random.randint(1, cfg["max_radius"] + 1)
        mode = np.random.choice(["erode", "dilate"])
        struct = generate_binary_structure(binary_mask.ndim, 1)
        if mode == "erode":
            new_mask = binary_erosion(binary_mask, structure=struct, iterations=radius)
        else:
            new_mask = binary_dilation(binary_mask, structure=struct, iterations=radius)
        result = np.where(new_mask, result, 0).astype(result.dtype)
        if mode == "dilate":
            result[new_mask & (~binary_mask)] = 1
        return result

    def _apply_omission(self, result: np.ndarray, cfg: dict) -> np.ndarray:
        if np.random.rand() >= cfg["prob"]:
            return result
        binary_mask = result > 0
        if not binary_mask.any():
            return result
        labeled_array, num_features = label_cc(binary_mask)
        if num_features <= 1:
            return result
        comp_sizes = np.bincount(labeled_array.ravel())
        small_comps = np.where(
            (comp_sizes[1:] > 0) & (comp_sizes[1:] < cfg["min_size"])
        )[0] + 1
        if len(small_comps) == 0:
            return result
        result[labeled_array == np.random.choice(small_comps)] = 0
        return result

    def _degrade_segmentation(self, seg_np: np.ndarray) -> np.ndarray:
        result = seg_np.copy()
        for cfg in self.degradation_pipeline:
            dtype = cfg["type"]
            method = getattr(self, f"_apply_{dtype}", None)
            if method is None:
                raise ValueError(f"Type de dégradation inconnu dans le trainer: '{dtype}'")
            result = method(result, cfg)
        return result

    def train_step(self, batch: dict) -> dict:
        """Exécute un pas d'entraînement avec dégradation des labels.

        Override de ``nnUNetTrainer.train_step`` pour injecter les
        dégradations avant le forward pass. Le batch contient :

        - ``data`` : tensor de shape ``(B, C, *spatial)``
        - ``target`` : liste de tensors (un par résolution de deep
          supervision), chacun de shape ``(B, 1, *spatial_ds)``

        On dégrade uniquement le target à la résolution la plus haute
        (index 0), puis on re-downsample pour les autres résolutions.

        Parameters
        ----------
        batch : dict
            Dictionnaire avec les clés ``'data'`` et ``'target'``.

        Returns
        -------
        dict
            Dictionnaire de loss retourné par le parent.
        """
        # target est une liste de tenseurs (deep supervision)
        target = batch["target"]

        # On dégrade le target à pleine résolution (index 0)
        # Shape: (B, 1, D, H, W) pour de la 3D
        target_fullres = target[0]
        target_np = target_fullres.cpu().numpy()

        for b in range(target_np.shape[0]):
            # Squeeze le canal pour travailler en (D, H, W)
            seg = target_np[b, 0]
            seg_degraded = self._degrade_segmentation(seg)
            target_np[b, 0] = seg_degraded

        target_fullres_degraded = torch.from_numpy(target_np).to(
            target_fullres.device, dtype=target_fullres.dtype
        )

        # Reconstruire la liste de targets deep supervision
        # en re-downsamplant le target dégradé
        new_target = [target_fullres_degraded]
        for ds_target in target[1:]:
            # Downsample par interpolation nearest
            ds_shape = ds_target.shape[2:]
            ds_degraded = torch.nn.functional.interpolate(
                target_fullres_degraded.float(),
                size=ds_shape,
                mode="nearest",
            ).to(dtype=ds_target.dtype)
            new_target.append(ds_degraded)

        batch["target"] = new_target

        return super().train_step(batch)


# ============================================================================
# Variantes pré-configurées pour le grid search
# ============================================================================


class nnUNetTrainerDegradedMorphoOnly(nnUNetTrainerDegraded):
    """Dégradation morphologique seule, sans omission."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.degradation_pipeline = [
            {"type": "morpho", "prob": 0.3, "max_radius": 3},
        ]


class nnUNetTrainerDegradedOmissionOnly(nnUNetTrainerDegraded):
    """Omission seule, sans dégradation morphologique."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.degradation_pipeline = [
            {"type": "omission", "prob": 0.2, "min_size": 150},
        ]


class nnUNetTrainerDegradedMild(nnUNetTrainerDegraded):
    """Dégradation légère (prob faibles, rayon petit)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.degradation_pipeline = [
            {"type": "morpho",   "prob": 0.1, "max_radius": 1},
            {"type": "omission", "prob": 0.1, "min_size": 50},
        ]


class nnUNetTrainerDegradedSevere(nnUNetTrainerDegraded):
    """Dégradation sévère (prob élevées, rayon grand)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.degradation_pipeline = [
            {"type": "morpho",   "prob": 0.5, "max_radius": 5},
            {"type": "omission", "prob": 0.5, "min_size": 500},
        ]


# ── Variantes paramétriques pour grid search ──
# Générées dynamiquement par le script grid_search.py qui crée
# des classes à la volée et les installe dans le dossier nnUNetTrainer.
