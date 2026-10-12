#!/usr/bin/env python3
"""Version parallèle de collect_metrics.py — mêmes calculs, boucle distribuée.

Réutilise à l'identique les fonctions de métriques (``PredFeatures``,
``evaluate_pair_cached``) et la résolution de chemins de ``collect_metrics`` ; seule
la boucle OOF (une tâche = un couple (modèle, fold, cas)) est distribuée sur un pool de
processus. Le résultat (table maîtresse ``results/metrics.csv``) est **identique** à la
version série : mêmes clés, même upsert idempotent.

Mémoire : chaque ``PredFeatures`` (~1,5 Go sur PARSE) + GT est local à une tâche et
libéré à sa fin ; ``max_tasks_per_child`` recycle les workers pour rendre la
fragmentation à l'OS. Dimensionné pour 24 Go de RAM (4 workers ≈ 16 Go de pic).

Usage
-----
    python scripts/collect_metrics_parallel.py --config configs/experiment_config.yaml \
        --output results --workers 4
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

# BLAS/threads à 1 AVANT tout import numpy : le parallélisme vient du Pool, pas des libs
# (évite l'oversubscription 14 workers × N threads et les deadlocks fork+threads).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nibabel as nib
import numpy as np
import yaml

import results_store as rs
from collect_metrics import (resolve_dataset_dir, resolve_results_dir,
                             scenario_gt_dir, _checkpoint_info)
from cross_evaluate import PredFeatures, evaluate_pair_cached


def _union_bbox(masks, margin=2):
    """Boîte englobante (avec marge) de l'union des masques booléens.

    Recadrer à cette boîte est EXACT pour toutes nos métriques : squelette, composantes
    connexes et distances de surface ne dépendent pas du fond vide au-delà du foreground
    (une marge ≥1 suffit à préserver les surfaces de bord). Le foreground des artères
    n'occupe que ~0,3 % du volume 512³ → gain ~100× sur la squelettisation.
    """
    any_fg = None
    for m in masks:
        any_fg = m if any_fg is None else (any_fg | m)
    if any_fg is None or not any_fg.any():
        return tuple(slice(0, s) for s in masks[0].shape)
    sl = []
    for ax in range(any_fg.ndim):
        idx = np.any(any_fg, axis=tuple(i for i in range(any_fg.ndim) if i != ax))
        nz = np.where(idx)[0]
        lo = max(0, int(nz[0]) - margin)
        hi = min(any_fg.shape[ax], int(nz[-1]) + 1 + margin)
        sl.append(slice(lo, hi))
    return tuple(sl)


def _process_case(task):
    """Worker : évalue une prédiction OOF contre tous ses scénarios GT à calculer.

    ``task`` est entièrement picklable (dicts/str/float). Retourne une liste de lignes
    ``results_store`` prêtes à upserter. Pred et GT sont recadrés à leur boîte englobante
    commune (exact, cf. ``_union_bbox``) avant tout calcul — c'est le gros gain de vitesse.
    """
    model, fold, case, pred_path, mtime, clean_dir, scenarios, num_epochs = task
    pred_nii = nib.load(pred_path)
    spacing = pred_nii.header.get_zooms()[:3]
    pred = np.asarray(pred_nii.get_fdata()) > 0

    gts = []
    for sc in scenarios:
        gt_path = os.path.join(scenario_gt_dir(clean_dir, "labelsTr", sc), case)
        if os.path.exists(gt_path):
            gts.append((sc, np.asarray(nib.load(gt_path).get_fdata()) > 0))
    if not gts:
        return []

    bbox = _union_bbox([pred] + [g for _, g in gts], margin=2)
    pf = PredFeatures(pred[bbox].astype(np.float32), spacing=spacing)
    rows = []
    for sc, g in gts:
        metrics = evaluate_pair_cached(pf, g[bbox].astype(np.float32))
        rows.append(rs.make_row(
            model=model["name"], trainer=model["trainer"],
            dataset=model["dataset_id"], fold=fold, scenario=sc["name"],
            case=case, eval_kind="oof", metrics=metrics,
            epochs=num_epochs, checkpoint_mtime=mtime))
    return rows


def build_tasks(config, clean_dir, nnunet_results, cfg, scenarios, done, num_epochs):
    """Liste des tâches (modèle, fold, cas) à calculer, en respectant ``done``."""
    tasks = []
    for model in config["experiment"]["models"]:
        rdir = resolve_results_dir(nnunet_results, model["dataset_id"], model["trainer"], cfg)
        if rdir is None:
            continue
        for fold_dir in sorted(glob.glob(os.path.join(rdir, "fold_*"))):
            fold = int(os.path.basename(fold_dir).split("_")[1])
            val_dir = os.path.join(fold_dir, "validation")
            _, mtime, finished = _checkpoint_info(fold_dir)
            if not finished or not os.path.isdir(val_dir):
                continue
            for pred_path in sorted(glob.glob(os.path.join(val_dir, "*.nii.gz"))):
                case = os.path.basename(pred_path)
                todo = [sc for sc in scenarios
                        if (model["name"], model["trainer"], int(model["dataset_id"]),
                            fold, sc["name"], case, "oof") not in done]
                if todo:
                    tasks.append((model, fold, case, pred_path, mtime, clean_dir,
                                  todo, num_epochs))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Collecte OOF parallèle")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tasks-per-child", type=int, default=4,
                        help="recycle chaque worker après N cas (borne mémoire)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    nnunet_raw = os.environ.get("nnUNet_raw", "nnUNet_data/nnUNet_raw")
    nnunet_results = os.environ.get("nnUNet_results", "nnUNet_data/nnUNet_results")
    cfg = config["training"]["configurations"][0]
    num_epochs = config["training"].get("num_epochs")
    scenarios = config["evaluation"]["scenarios"]

    clean_dir = resolve_dataset_dir(nnunet_raw, config["dataset"]["id"])
    if clean_dir is None:
        print(f"ERREUR: dataset source introuvable dans {nnunet_raw}")
        return

    df = rs.load(args.output)
    done = rs.existing_keys(df)
    print(f"Table maîtresse : {len(df)} lignes déjà présentes")

    tasks = build_tasks(config, clean_dir, nnunet_results, cfg, scenarios, done, num_epochs)
    if not tasks:
        print("Rien de nouveau à collecter.")
        return
    print(f"{len(tasks)} cas à évaluer sur {args.workers} workers "
          f"(recyclage tous les {args.max_tasks_per_child} cas)…")

    all_rows, t0, n_done = [], time.time(), 0
    # fork : démarrage instantané des workers, pas de re-import, pages partagées (COW).
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers,
                  maxtasksperchild=args.max_tasks_per_child) as pool:
        for rows in pool.imap_unordered(_process_case, tasks, chunksize=1):
            all_rows.extend(rows)
            n_done += 1
            if n_done % 10 == 0 or n_done == len(tasks):
                el = time.time() - t0
                rate = n_done / el if el else 0
                eta = (len(tasks) - n_done) / rate if rate else 0
                print(f"  {n_done}/{len(tasks)} cas · {el:5.0f}s écoulé · ETA {eta:4.0f}s",
                      flush=True)

    n_new = rs.upsert(args.output, all_rows)
    print(f"\n{n_new} lignes upsertées → {os.path.join(args.output, rs.CSV_NAME)} "
          f"({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
