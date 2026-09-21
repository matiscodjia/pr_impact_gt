#!/usr/bin/env python3
"""Imprime les noms de modèles d'une config dont le tier figure dans la chaîne donnée.
Usage : list_models.py CONFIG TIERS   (ex. list_models.py configs/x.yaml ST)"""
import sys

import yaml

cfg, tiers = sys.argv[1], sys.argv[2]
for m in yaml.safe_load(open(cfg, encoding="utf-8"))["experiment"]["models"]:
    if m["tier"] in tiers:
        print(m["name"])
