"""Custom nnU-Net Trainer avec dégradations formelles on-the-fly des labels.

Les dégradations utilisent les 4 familles formelles θ = (family, r, p, seed) de
l'étude metric_reliability_study, intégrées dans scripts/degradations.py :

  distal_omission   : ouverture morpho + Bernoulli par composante fine (radius-gated)
  boundary_jitter   : déplacement surface ±r voxels, champ lissé spatialement corrélé
  distal_truncation : élagage tronçons terminaux squelette, Bernoulli par tronçon
  homogeneous_morpho: CONTRÔLE — érosion uniforme, sans gating radius (irréaliste)

Chaque famille est paramétrée par (r, p) et prend un seed isolé → résultats
reproductibles et indépendants de l'état global du RNG.

Lien avec les 3 questions de recherche
---------------------------------------
Q1 — Pénalisation artificielle par GT⁻ d'évaluation :
    Model_Star (entraîné sur GT*) vs GT⁻ lors de l'évaluation.
    Testé par la Famille B des tests Wilcoxon (inter-scénario, même modèle).

Q2 — Robustesse induite (GT⁻ comme data augmentation) :
    Model_Minus_Stoch (ce trainer) vs Model_Star sur GT⁻.
    Les dégradations stochastiques on-the-fly élargissent l'espace d'entraînement.
    Comparé au contrôle homogeneous_morpho (Model_Minus_HomoMorpho) pour tester
    si la robustesse est spécifique à la structure du bruit.

Q3 — Apprentissage du biais de segmentation :
    Même distribution train/test → pénalité d'évaluation réduite pour Model_Minus_Stoch.
    Testé par noise_learning_test() : pénalité = CLDice(GT*) − CLDice(GT⁻).

Usage
-----
    nnUNetv2_train 100 3d_fullres 0 -tr nnUNetTrainerDegraded

Trainers du plan d'expériences (cf. experiments_plan.md) :
    nnUNetTrainerStd                   — M0/M3/M4 : standard (GT propre ou dataset figé)
    nnUNetTrainerDegradedOmissionOnly  — M1 : distal_omission r2 p0.3, on-the-fly
    nnUNetTrainerDriftMu0              — M2 : boundary_drift r1 p0.5 μ=0, on-the-fly
"""

import os
import sys

import numpy as np
import torch
import yaml

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# Le générateur formel est dans scripts/degradations.py — on l'importe en ajoutant
# le dossier scripts/ au path si nécessaire.
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from degradations import generate  # noqa: E402  (import after sys.path manipulation)


def _num_epochs_from_config(default: int = 250) -> int:
    if os.environ.get("DEBUG_PIPELINE") == "1":
        return 2
    config_path = os.path.join(os.getcwd(), "configs", "experiment_config.yaml")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return int(config.get("training", {}).get("num_epochs", default))
    except Exception:
        return default


def _load_stochastic_pipeline(default: list) -> list:
    config_path = os.path.join(os.getcwd(), "configs", "experiment_config.yaml")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("stochastic_degradations", default)
    except Exception:
        return default


class nnUNetTrainerDebug(nnUNetTrainer):
    """Trainer pour debug rapide (2 époques)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2


class nnUNetTrainerStd(nnUNetTrainer):
    """nnU-Net standard, num_epochs lu depuis le config (500).

    Utilisé pour M0_Star (Dataset100) et les modèles sur datasets FIGÉS
    M3/M4_Drift (Dataset103/104).
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = _num_epochs_from_config()
        self.print_to_log_file(f"num_epochs set to {self.num_epochs} (config)")


class nnUNetTrainerDegraded(nnUNetTrainer):
    """Trainer nnU-Net avec dégradations formelles on-the-fly.

    Pipeline par défaut : distal_omission + boundary_jitter.
    Les seeds sont tirés aléatoirement à chaque step pour ne pas fixer le bruit
    (comportement data augmentation, pas reproductible par step mais reproductible
    au niveau du fold via le seed global PyTorch/NumPy).
    """

    # Paramètre par défaut (surchargé par la config ou les sous-classes).
    # p = sévérité, PAS probabilité d'application par image.
    # 100 % des batches sont dégradés — la garantie Bernoulli dans degradations.py
    # assure qu'au moins une structure fine est retirée si elle existe.
    # Point combiné MESURÉ sur PARSE : clDice(GT⁻,GT*) = 0.866 ∈ [0.85, 0.90].
    _DEFAULT_PIPELINE = [
        {"family": "distal_omission", "r": 1, "p": 0.3},
        {"family": "boundary_jitter", "r": 2, "p": 1.0},
    ]

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.degradation_pipeline = _load_stochastic_pipeline(self._DEFAULT_PIPELINE)
        self.num_epochs = _num_epochs_from_config()
        if os.environ.get("DEBUG_PIPELINE") == "1":
            self.num_epochs = 2

        # Espacement voxel (mm) — propagé au générateur pour les familles mm-aware.
        # Disponible après __init__ de nnUNetTrainer via configuration_manager.
        try:
            self._spacing = tuple(float(s) for s in self.configuration_manager.spacing)
        except Exception:
            self._spacing = (1.0, 1.0, 1.0)

        self._corr_len = 4.0  # longueur de corrélation (voxels), cohérente avec reliability study

        pipeline_summary = " | ".join(
            f"{c['family']}(r={c['r']}, p={c['p']})" for c in self.degradation_pipeline
        )
        self.print_to_log_file(
            f"\n{'='*60}\n"
            f"nnUNetTrainerDegraded — dégradations formelles on-the-fly\n"
            f"  pipeline : {pipeline_summary}\n"
            f"  spacing  : {self._spacing}\n"
            f"  epochs   : {self.num_epochs}\n"
            f"{'='*60}\n"
        )

    def _degrade_segmentation(self, seg_np: np.ndarray) -> np.ndarray:
        """Applique séquentiellement le pipeline de dégradations formelles.

        Politique : 100 % des appels dégradent l'image — pas de skip par batch.
        Un seed aléatoire est tiré par famille à chaque appel (variance entre
        epochs = diversité des réalisations de bruit → data augmentation).
        La garantie Bernoulli dans degradations.py assure qu'au moins une
        structure fine est toujours retirée si elle existe.
        """
        result = seg_np.copy()
        for cfg in self.degradation_pipeline:
            family = cfg["family"]
            r = float(cfg["r"])
            p = float(cfg["p"])
            mu = float(cfg.get("mu", 0.0))   # biais de dérive (boundary_drift) ; 0 = jitter pur
            seed = int(np.random.randint(0, 2**31 - 1))
            result = generate(result, family, r, p, seed, self._spacing, self._corr_len, mu)
        return result

    def train_step(self, batch: dict) -> dict:
        """Pas d'entraînement avec dégradation formelle des labels.

        Override de nnUNetTrainer.train_step : injecte les dégradations sur le
        target à pleine résolution (index 0 de la liste deep supervision) puis
        re-downsample pour les autres niveaux.
        """
        target = batch["target"]
        target_fullres = target[0]                    # (B, 1, D, H, W)
        target_np = target_fullres.cpu().numpy()

        for b in range(target_np.shape[0]):
            seg = target_np[b, 0]                     # (D, H, W)
            target_np[b, 0] = self._degrade_segmentation(seg)

        target_fullres_degraded = torch.from_numpy(target_np).to(
            target_fullres.device, dtype=target_fullres.dtype
        )

        new_target = [target_fullres_degraded]
        for ds_target in target[1:]:
            ds_shape = ds_target.shape[2:]
            ds_degraded = torch.nn.functional.interpolate(
                target_fullres_degraded.float(), size=ds_shape, mode="nearest"
            ).to(dtype=ds_target.dtype)
            new_target.append(ds_degraded)

        batch["target"] = new_target
        return super().train_step(batch)


# ============================================================================
# Variantes pré-configurées
# ============================================================================

class nnUNetTrainerDegradedOmissionOnly(nnUNetTrainerDegraded):
    """M1 — bruit topologique biaisé : distal_omission seule, on-the-fly.

    r2 p0.3 → clDice 0.878 (mesuré, dans la fenêtre inter-observateur).
    100 % des images dégradées ; nouvelle réalisation à chaque epoch.
    """
    _DEFAULT_PIPELINE = [
        {"family": "distal_omission", "r": 2, "p": 0.3},
    ]


class nnUNetTrainerDriftMu0(nnUNetTrainerDegraded):
    """M2 — bruit de bord NON biaisé : boundary_drift μ=0, on-the-fly.

    r1 p0.5, μ=0 → NSD@0.5 ≈ 0.88 (dans la fenêtre). Re-tiré à chaque epoch :
    les fluctuations de moyenne nulle se compensent → augmentation/robustesse (Q2).
    NB : pour un biais APPRENABLE (Q3, μ≠0), on utilise un dataset FIGÉ (Dataset103/104),
    pas ce trainer on-the-fly.
    """
    _DEFAULT_PIPELINE = [
        {"family": "boundary_drift", "r": 1, "p": 0.5, "mu": 0.0},
    ]


# ============================================================================
# Calibration du schedule
# ============================================================================
# Objectif : situer le plateau des métriques de TOPOLOGIE (clDice/Betti0).
# Le trainer tourne sur `calibration.budget` époques et sauvegarde un checkpoint
# à chaque jalon ; scripts/calibrate.py les évalue ensuite en out-of-fold.

def _calibration_cfg() -> tuple[int, set[int]]:
    if os.environ.get("DEBUG_PIPELINE") == "1":
        return 4, {2, 4}
    config_path = os.path.join(os.getcwd(), "configs", "experiment_config.yaml")
    try:
        cal = (yaml.safe_load(open(config_path)) or {}).get("calibration", {}) or {}
        return int(cal.get("budget", 1000)), {int(x) for x in cal.get("milestones", [])}
    except Exception:
        return 1000, set()


def _calib_init(trainer) -> None:
    budget, milestones = _calibration_cfg()
    trainer.num_epochs = budget
    trainer._calib_milestones = milestones
    trainer.print_to_log_file(
        f"[calib] budget={budget} epochs, jalons={sorted(milestones)}")


def _calib_snapshot(trainer) -> None:
    ep = int(getattr(trainer, "current_epoch", -1))
    if ep in getattr(trainer, "_calib_milestones", set()):
        path = os.path.join(trainer.output_folder, f"checkpoint_ep{ep}.pth")
        trainer.save_checkpoint(path)
        trainer.print_to_log_file(f"[calib] snapshot epoch {ep} → {path}")


class nnUNetTrainerCalib(nnUNetTrainer):
    """Calibration sur GT propres (pour Model_Star / Model_Minus_Fixed)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        _calib_init(self)

    def on_epoch_end(self):
        super().on_epoch_end()
        _calib_snapshot(self)


class nnUNetTrainerCalibDegraded(nnUNetTrainerDegraded):
    """Calibration avec dégradations formelles on-the-fly (pour Model_Minus_Stoch)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        _calib_init(self)  # écrase num_epochs hérité par le budget de calibration

    def on_epoch_end(self):
        super().on_epoch_end()
        _calib_snapshot(self)
