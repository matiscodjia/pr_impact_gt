#!/usr/bin/env python3
"""Génère configs/generated_seeds_config.yaml : la config de l'étude avec, à la place de
la matrice de modèles, les réplicats de graine (M0/M1/M2 x graines 1 et 2, Dataset100).

Pourquoi un fichier à part : `experiment_config.yaml` est lu par report.py/collect_metrics
en local ; y ajouter des modèles changerait les tables de l'étude. Ici on ne touche à rien.
Tous les autres blocs (training, scénarios, calibration...) sont recopiés tels quels.

Tiers : S = graine 1, T = graine 2. Noms de modèle : M0_Star_s1, M1_Omission_s1, ...
"""
import argparse
import os

import yaml

REPLICATES = [  # (nom de base, trainer de base)
    ("M0_Star", "nnUNetTrainerStd"),
    ("M1_Omission", "nnUNetTrainerDegradedOmissionOnly"),
    ("M2_Drift_mu0", "nnUNetTrainerDriftMu0"),
]
TIER_OF_SEED = {1: "S", 2: "T"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/experiment_config.yaml")
    ap.add_argument("--out", default="configs/generated_seeds_config.yaml")
    args = ap.parse_args()
    with open(args.base, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    models = []
    for seed, tier in TIER_OF_SEED.items():
        for name, trainer in REPLICATES:
            models.append({"name": f"{name}_s{seed}", "trainer": f"{trainer}_s{seed}",
                           "dataset_id": 100, "tier": tier})
    cfg["experiment"]["models"] = models
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# GÉNÉRÉ par scripts/cluster/make_seed_config.py -- ne pas éditer\n")
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"{args.out} : {len(models)} modèles ({', '.join(m['name'] for m in models)})")


if __name__ == "__main__":
    main()
