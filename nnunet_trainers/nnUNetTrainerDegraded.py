"""Custom nnU-Net trainers: on-the-fly label noise, seed replicates, schedule calibration.

nnU-Net finds trainers by class name inside its own package, so this file is COPIED to
``nnunetv2/training/nnUNetTrainer/`` (pipeline/orchestrator.py and cluster/launch_gpu.sh do
it). The noise operator comes from the ``prgt`` package, which must be installed in the
same environment (``pip install -e . --no-deps`` at the repository root). The configuration
is read from ``configs/experiment_config.yaml`` relative to the working directory, so
nnU-Net must be launched from the repository root.

Trainers of the study (configs/experiment_config.yaml):
    nnUNetTrainerStd                   M0, M3, M4: standard training (clean or frozen labels)
    nnUNetTrainerDegradedOmissionOnly  M1: distal_omission r2 p0.3, redrawn on the fly
    nnUNetTrainerDriftMu0              M2: boundary_drift r1 p0.5 mu=0, redrawn on the fly
    *_s1, *_s2                         seed replicates of the three above
    nnUNetTrainerCalib[Degraded]       one long run with checkpoint snapshots (num_epochs)

The noise families and their parameters theta = (family, r, p, mu) are described in
prgt/noise.py. Every training batch is degraded (p is a severity, not a probability of
degrading a batch), with a new random seed at each call, so each epoch sees a new noise
realisation.

    nnUNetv2_train 100 3d_fullres 0 -tr nnUNetTrainerDegradedOmissionOnly
"""

import os

import numpy as np
import torch
import yaml

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

try:
    from prgt.noise import apply_noise
except ImportError as e:   # the trainer runs inside nnU-Net, outside the repository
    raise ImportError(
        "nnUNetTrainerDegraded needs the prgt package: run `pip install -e . --no-deps` "
        "at the root of the repository, in the environment that runs nnU-Net."
    ) from e


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
    """Quick debug trainer (2 epochs)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2


class nnUNetTrainerStd(nnUNetTrainer):
    """Standard nnU-Net with num_epochs read from the configuration (500).

    Used for M0_Star (Dataset100) and for the models trained on FROZEN degraded labels,
    M3/M4 (Dataset103/104).
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = _num_epochs_from_config()
        self.print_to_log_file(f"num_epochs set to {self.num_epochs} (config)")


class nnUNetTrainerDegraded(nnUNetTrainer):
    """nnU-Net trainer that degrades the training labels on the fly.

    The noise pipeline comes from ``stochastic_degradations`` in the configuration, or
    from ``_DEFAULT_PIPELINE`` of the subclass. A new seed is drawn for each family at
    each call, so the noise is not frozen (data-augmentation behaviour).
    """

    # Default pipeline (overridden by the configuration or the subclasses).
    # p is a severity, NOT a probability of degrading an image: every batch is degraded.
    # Combined point measured on PARSE: clDice(GT-, GT*) = 0.866, inside [0.85, 0.90].
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

        # Voxel size (mm), passed to the noise operator (the families work in mm).
        # Available after nnUNetTrainer.__init__ through configuration_manager.
        try:
            self._spacing = tuple(float(s) for s in self.configuration_manager.spacing)
        except Exception:
            self._spacing = (1.0, 1.0, 1.0)

        self._corr_len = 4.0  # correlation length (voxels) of the smoothed noise fields

        pipeline_summary = " | ".join(
            f"{c['family']}(r={c['r']}, p={c['p']})" for c in self.degradation_pipeline
        )
        self.print_to_log_file(
            f"\n{'='*60}\n"
            f"nnUNetTrainerDegraded: on-the-fly label noise\n"
            f"  pipeline : {pipeline_summary}\n"
            f"  spacing  : {self._spacing}\n"
            f"  epochs   : {self.num_epochs}\n"
            f"{'='*60}\n"
        )

    def _degrade_segmentation(self, seg_np: np.ndarray) -> np.ndarray:
        """Apply the noise pipeline, family by family, to one segmentation.

        Every call degrades the image (no batch is skipped). A random seed is drawn per
        family at each call, so successive epochs see different noise realisations.
        """
        result = seg_np.copy()
        for cfg in self.degradation_pipeline:
            family = cfg["family"]
            r = float(cfg["r"])
            p = float(cfg["p"])
            mu = float(cfg.get("mu", 0.0))   # drift bias (boundary_drift); 0 = pure jitter
            seed = int(np.random.randint(0, 2**31 - 1))
            result = apply_noise(result, family, r, p, seed, self._spacing, self._corr_len, mu)
        return result

    def train_step(self, batch: dict) -> dict:
        """Training step on degraded labels.

        Overrides nnUNetTrainer.train_step: the noise is applied to the full-resolution
        target (index 0 of the deep-supervision list), which is then downsampled again
        for the other levels.
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
# Trainers of the study
# ============================================================================

class nnUNetTrainerDegradedOmissionOnly(nnUNetTrainerDegraded):
    """M1: biased topological noise, distal_omission only, on the fly.

    r2 p0.3 -> clDice 0.878 (measured, inside the inter-observer window). Every image is
    degraded, with a new realisation at each epoch.
    """
    _DEFAULT_PIPELINE = [
        {"family": "distal_omission", "r": 2, "p": 0.3},
    ]


class nnUNetTrainerDriftMu0(nnUNetTrainerDegraded):
    """M2: UNBIASED boundary noise, boundary_drift mu = 0, on the fly.

    r1 p0.5, mu = 0 -> NSD@0.5 ~ 0.88 (inside the window). Redrawn at each epoch: the
    zero-mean fluctuations average out, so this acts as augmentation (Q2). A LEARNABLE
    bias (Q3, mu != 0) uses frozen labels instead (Dataset103/104), not this trainer.
    """
    _DEFAULT_PIPELINE = [
        {"family": "boundary_drift", "r": 1, "p": 0.5, "mu": 0.0},
    ]


# ============================================================================
# Seed replicates (training-seed variance)
# ============================================================================
# nnU-Net names the output folder after the trainer, so two trainings of the SAME trainer
# would overwrite each other. A replicate is therefore a separate subclass (separate
# folder): same dataset, same folds (splits_final.json), different seed.
#
# The seed fixes the weight initialisation and the RNG of the main process. This is NOT
# bit-for-bit reproducibility (cuDNN and the augmentation workers are not deterministic):
# it is a documented independent replicate, which is what a variance estimate needs.

def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _SeedMixin:
    """Put it BEFORE the base class: ``class X_s1(_SeedMixin, nnUNetTrainerStd)``."""
    _SEED = 0

    def initialize(self):
        # fold-dependent seed: the 5 folds of a replicate differ, the set stays fixed
        _set_seed(self._SEED * 1000 + int(self.fold))
        self.print_to_log_file(f"[seed] replicate seed={self._SEED}, effective={self._SEED * 1000 + int(self.fold)}")
        super().initialize()


class nnUNetTrainerStd_s1(_SeedMixin, nnUNetTrainerStd):
    _SEED = 1


class nnUNetTrainerStd_s2(_SeedMixin, nnUNetTrainerStd):
    _SEED = 2


class nnUNetTrainerDegradedOmissionOnly_s1(_SeedMixin, nnUNetTrainerDegradedOmissionOnly):
    _SEED = 1


class nnUNetTrainerDegradedOmissionOnly_s2(_SeedMixin, nnUNetTrainerDegradedOmissionOnly):
    _SEED = 2


class nnUNetTrainerDriftMu0_s1(_SeedMixin, nnUNetTrainerDriftMu0):
    _SEED = 1


class nnUNetTrainerDriftMu0_s2(_SeedMixin, nnUNetTrainerDriftMu0):
    _SEED = 2


# ============================================================================
# Schedule calibration
# ============================================================================
# Goal: locate the plateau of the TOPOLOGY metrics (clDice, Betti-0). The trainer runs for
# `calibration.budget` epochs and saves a checkpoint at each milestone;
# pipeline/calibrate_epochs.py then scores them out of fold.

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
        f"[calib] budget={budget} epochs, milestones={sorted(milestones)}")


def _calib_snapshot(trainer) -> None:
    ep = int(getattr(trainer, "current_epoch", -1))
    if ep in getattr(trainer, "_calib_milestones", set()):
        path = os.path.join(trainer.output_folder, f"checkpoint_ep{ep}.pth")
        trainer.save_checkpoint(path)
        trainer.print_to_log_file(f"[calib] snapshot epoch {ep} -> {path}")


class nnUNetTrainerCalib(nnUNetTrainer):
    """Calibration run on clean labels."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        _calib_init(self)

    def on_epoch_end(self):
        super().on_epoch_end()
        _calib_snapshot(self)


class nnUNetTrainerCalibDegraded(nnUNetTrainerDegraded):
    """Calibration run with on-the-fly label noise."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        _calib_init(self)  # replaces the inherited num_epochs with the calibration budget

    def on_epoch_end(self):
        super().on_epoch_end()
        _calib_snapshot(self)
