#!/usr/bin/env python3
"""P1.2 (avancé en urgence) -- Le retournement HD95 survit-il au post-traitement LCC ?

Confirmé en Phase 0 : aucun post-traitement "largest connected component" (LCC) n'est
appliqué aux prédictions utilisées pour les métriques actuelles (44-81 composantes par
cas). HD95 est la métrique la plus sensible à un fragment parasite isolé (un seul
faux-positif éloigné suffit à déplacer le 95e percentile). Ce script recalcule TOUTES
les métriques, pour les 1020 paires (model,scenario,case) du tier A, EN PLUS avec LCC
(garder uniquement la plus grande composante 26-connexe de la prédiction, la référence
n'est jamais post-traitée), et compare directement les deux versions du test de
retournement.

Convention : `postprocessing` in {"none", "lcc"} ajouté comme colonne. LCC est appliqué
à la PRÉDICTION seule (les GT ne sont jamais touchées). Le bbox de crop est calculé sur
la version "none" (pred brute + GT) -- comme pred_lcc est un sous-ensemble strict de
pred, ce bbox reste exact a fortiori pour la version LCC (mêmes arguments qu'en triage
pré-P0, vérifiés empiriquement sur le pipeline principal).

Sortie : results/metrics_lcc.csv (2040 lignes : 1020 x {none, lcc}), et
results/rank_reversal_lcc_summary.csv (comparaison directe des 8 lignes retournées de
`rank_reversal_tests.csv` en "none" vs. leur équivalent recalculé en "lcc").

Usage
-----
    # estimation de temps sur 3 prédictions
    python analysis/lcc_postprocessing.py --limit 3

    # run complet
    python analysis/lcc_postprocessing.py --workers 4
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cc3d
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import results_store as rs  # noqa: E402
from collect_metrics_parallel import _union_bbox  # noqa: E402
from cross_evaluate import PredFeatures, evaluate_pair_cached  # noqa: E402

TRAINER_DIR = {
    "M0_Star": "nnUNetTrainerStd",
    "M1_Omission": "nnUNetTrainerDegradedOmissionOnly",
    "M2_Drift_mu0": "nnUNetTrainerDriftMu0",
}
SCENARIO_DIR = {
    "GT_star": "labelsTr",
    "GT_minus_omission": "labelsTr_GT_minus_omission",
    "GT_minus_drift_neg": "labelsTr_GT_minus_drift_neg",
    "GT_minus_drift_pos": "labelsTr_GT_minus_drift_pos",
}
SCENARIOS = list(SCENARIO_DIR)
OUT_CSV = "metrics_lcc.csv"
KEY = ["model", "fold", "case", "scenario", "postprocessing"]


def _dataset_root(nnunet_data_root: str) -> str:
    return os.path.join(nnunet_data_root, "nnUNet_raw", "Dataset100_PARSE")


def _pred_path(nnunet_data_root: str, model: str, fold: int, case: str) -> str:
    trainer = TRAINER_DIR[model]
    return os.path.join(
        nnunet_data_root, "nnUNet_results", "Dataset100_PARSE",
        f"{trainer}__nnUNetPlans__3d_fullres", f"fold_{fold}", "validation", case,
    )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Garde uniquement la plus grande composante 26-connexe. Masque vide -> inchangé."""
    if not mask.any():
        return mask
    labels, n = cc3d.connected_components(mask, connectivity=26, return_N=True)
    if n <= 1:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # fond
    biggest = int(np.argmax(counts))
    return labels == biggest


def unique_predictions(results_dir: str, limit: int | None) -> list[tuple]:
    df = rs.load(results_dir)
    df = df[(df["eval_kind"] == "oof") & (df["model"].isin(TRAINER_DIR))]
    uniq = df.drop_duplicates(subset=["model", "fold", "case"])[["model", "fold", "case"]]
    rows = [tuple(r) for r in uniq.itertuples(index=False, name=None)]
    rows.sort()
    if limit:
        rows = rows[:limit]
    return rows


def existing_keys(results_dir: str) -> set[tuple]:
    path = os.path.join(results_dir, OUT_CSV)
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(zip(df["model"], df["fold"], df["case"], df["scenario"], df["postprocessing"]))


def _process_one(task: tuple) -> list[dict]:
    model, fold, case, nnunet_data_root = task
    dataset_root = _dataset_root(nnunet_data_root)
    pred_path = _pred_path(nnunet_data_root, model, fold, case)
    if not os.path.exists(pred_path):
        return []

    pred_nii = nib.load(pred_path)
    spacing = pred_nii.header.get_zooms()[:3]
    pred = np.asarray(pred_nii.get_fdata()) > 0
    pred_lcc = _largest_component(pred)

    rows = []
    for scenario in SCENARIOS:
        gt_path = os.path.join(dataset_root, SCENARIO_DIR[scenario], case)
        if not os.path.exists(gt_path):
            continue
        gt = np.asarray(nib.load(gt_path).get_fdata()) > 0

        # bbox calculé sur pred brute (none) -- exact a fortiori pour pred_lcc, sous-ensemble.
        bbox = _union_bbox([pred, gt], margin=2)
        gt_c = gt[bbox].astype(np.float32)

        for post, mask in (("none", pred), ("lcc", pred_lcc)):
            pf = PredFeatures(mask[bbox].astype(np.float32), spacing=spacing)
            m = evaluate_pair_cached(pf, gt_c)
            rows.append({
                "model": model, "fold": fold, "case": case, "scenario": scenario,
                "postprocessing": post,
                "cldice": m.get("cldice"), "hd95": m.get("hd95"), "nsd": m.get("nsd"),
                "nsd05": m.get("nsd05"), "betti0": m.get("betti0"),
                "volume_delta": m.get("volume_delta"),
            })
    return rows


def upsert(results_dir: str, rows: list[dict], seed: int) -> None:
    path = os.path.join(results_dir, OUT_CSV)
    current = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=KEY)
    incoming = pd.DataFrame(rows)
    incoming["git_commit"] = rs.git_commit()
    incoming["seed"] = seed
    incoming["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    combined = pd.concat([current, incoming], ignore_index=True)
    combined = combined.drop_duplicates(subset=KEY, keep="last").sort_values(KEY)
    os.makedirs(results_dir, exist_ok=True)
    combined.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description="P1.2 -- Métriques avec/sans LCC")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--nnunet_data_root", default="nnUNet_data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limite le nombre de PRÉDICTIONS uniques (model,fold,case) -- estimation.")
    args = ap.parse_args()

    preds = unique_predictions(args.results_dir, args.limit)
    done_keys = existing_keys(args.results_dir)
    done_preds = {(m, f, c) for (m, f, c, s, p) in done_keys
                  if all((m, f, c, s2, p2) in done_keys for s2 in SCENARIOS for p2 in ("none", "lcc"))}
    todo = [(m, f, c, args.nnunet_data_root) for (m, f, c) in preds if (m, f, c) not in done_preds]

    if not todo:
        print("[SKIP] rien à faire (déjà calculé, ou --limit trop restrictif).")
        return

    print(f"[START] {len(todo)} prédictions uniques à traiter "
          f"({len(todo) * len(SCENARIOS) * 2} lignes à produire), workers={args.workers}")
    t0 = time.time()
    all_rows = []
    if args.workers > 1:
        with mp.Pool(args.workers, maxtasksperchild=10) as pool:
            for i, rows in enumerate(pool.imap_unordered(_process_one, todo), 1):
                all_rows.extend(rows)
                if i % 10 == 0 or i == len(todo):
                    dt = time.time() - t0
                    print(f"  [{i}/{len(todo)}] {dt:.1f}s ({dt / i:.2f}s/pred, "
                          f"ETA {dt / i * (len(todo) - i):.0f}s)")
    else:
        for i, task in enumerate(todo, 1):
            all_rows.extend(_process_one(task))
            if i % 5 == 0 or i == len(todo):
                dt = time.time() - t0
                print(f"  [{i}/{len(todo)}] {dt:.1f}s ({dt / i:.2f}s/pred, "
                      f"ETA {dt / i * (len(todo) - i):.0f}s)")

    if all_rows:
        upsert(args.results_dir, all_rows, args.seed)
    dt = time.time() - t0
    print(f"[OK] {len(all_rows)} lignes écrites dans "
          f"{os.path.join(args.results_dir, OUT_CSV)} en {dt:.1f}s")


if __name__ == "__main__":
    main()
