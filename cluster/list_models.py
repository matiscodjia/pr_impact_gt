#!/usr/bin/env python3
"""Print the names of the models of a configuration whose tier is in TIERS.

    python cluster/list_models.py CONFIG TIERS     (e.g. cluster/list_models.py configs/x.yaml ST)
"""
import sys

import yaml

cfg, tiers = sys.argv[1], sys.argv[2]
for m in yaml.safe_load(open(cfg, encoding="utf-8"))["experiment"]["models"]:
    if m["tier"] in tiers:
        print(m["name"])
