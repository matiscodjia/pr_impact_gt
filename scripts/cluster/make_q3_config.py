#!/usr/bin/env python3
"""Génère configs/generated_q3_config.yaml : la config de l'étude réduite aux modèles Q3.

Q3 = « un réseau apprend-il un biais systématique d'annotation ? ». M3 est entraîné sur
Dataset103 (labels = GT⁻ drift μ=−0,5, une réalisation figée par cas), M4 sur Dataset104
(μ=+0,5). Tier B dans experiment_config.yaml.

Pourquoi un fichier à part (comme make_seed_config.py) : collect_metrics collecte TOUS les
modèles de la config. Avec experiment_config.yaml, chaque dossier results_seeds/M3_... ou
M4_... re-scorerait aussi M0-M2 (1020 lignes, 1-2 h de CPU sur le worker, pendant que les
workers d'augmentation en ont besoin). M0 est déjà scoré en local (results/metrics.csv) contre
les mêmes références : l'analyse Q3 les combine.
"""
import argparse
import os

import yaml

Q3_MODELS = ("M3_Drift_muMinus", "M4_Drift_muPlus")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/experiment_config.yaml")
    ap.add_argument("--out", default="configs/generated_q3_config.yaml")
    args = ap.parse_args()
    with open(args.base, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    models = [m for m in cfg["experiment"]["models"] if m["name"] in Q3_MODELS]
    if len(models) != len(Q3_MODELS):
        raise SystemExit(f"{args.base} : modèles Q3 attendus {Q3_MODELS}, trouvés {[m['name'] for m in models]}")
    cfg["experiment"]["models"] = models
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# GÉNÉRÉ par scripts/cluster/make_q3_config.py -- ne pas éditer\n")
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    desc = ", ".join("{} (d{}, tier {})".format(m["name"], m["dataset_id"], m["tier"]) for m in models)
    print(f"{args.out} : {desc}")


if __name__ == "__main__":
    main()
