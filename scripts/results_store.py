#!/usr/bin/env python3
"""Table maîtresse des résultats — stockage long-format, idempotent, traçable.

Toutes les métriques produites par le pipeline (out-of-fold ET test set) vivent
dans **une seule** table long-format, une ligne par
``(model, trainer, dataset, fold, scenario, case, eval_kind)``. Chaque ligne
porte sa provenance (epochs, checkpoint, commit git, version nnU-Net, horodatage)
de sorte que la maturité de chaque chiffre est lisible sans contexte externe.

L'écriture est un **upsert idempotent** : recollecter après une interruption ne
recalcule jamais une clé déjà présente (elle est simplement remplacée à
l'identique), ce qui rend le pipeline rejouable sans perte ni doublon.

Le CSV est la source de vérité (petit volume, diffable, auditable) ; un miroir
Parquet est écrit en plus si ``pyarrow``/``fastparquet`` est disponible.

Usage (programmatique)
----------------------
    from results_store import load, upsert, existing_keys, KEY
    df = load("results")
    done = existing_keys(df)            # set de tuples clés déjà calculés
    upsert("results", [row1, row2, ...])  # chaque row = dict couvrant SCHEMA
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone

import pandas as pd

# ── Schéma canonique ──
METRICS = ["cldice", "hd95", "nsd", "betti0"]
KEY = ["model", "trainer", "dataset", "fold", "scenario", "case", "eval_kind"]
PROVENANCE = ["epochs", "checkpoint_mtime", "git_commit", "nnunet_version", "eval_ts"]
SCHEMA = KEY + METRICS + PROVENANCE

CSV_NAME = "metrics.csv"
PARQUET_NAME = "metrics.parquet"


# ─────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────

def git_commit() -> str:
    """Hash court du commit courant (``unknown`` hors dépôt git)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def nnunet_version() -> str:
    """Version de nnU-Net installée (``unknown`` si absent)."""
    try:
        import nnunetv2
        return getattr(nnunetv2, "__version__", "unknown")
    except Exception:
        return "unknown"


def now_iso() -> str:
    """Horodatage ISO-8601 UTC."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────

def _csv_path(results_dir: str) -> str:
    return os.path.join(results_dir, CSV_NAME)


def _parquet_path(results_dir: str) -> str:
    return os.path.join(results_dir, PARQUET_NAME)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMA)


def load(results_dir: str) -> pd.DataFrame:
    """Charge la table maîtresse (DataFrame vide au schéma si absente)."""
    path = _csv_path(results_dir)
    if not os.path.exists(path):
        return _empty()
    df = pd.read_csv(path)
    # Garantit la présence de toutes les colonnes du schéma (compat ascendante).
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = pd.NA
    return df[SCHEMA]


def key_of(row: dict) -> tuple:
    """Tuple clé d'une ligne (pour test d'appartenance rapide)."""
    return tuple(row[k] for k in KEY)


def existing_keys(df: pd.DataFrame) -> set[tuple]:
    """Ensemble des clés déjà présentes — pour sauter les calculs faits."""
    if df.empty:
        return set()
    return set(map(tuple, df[KEY].astype(object).values.tolist()))


def upsert(results_dir: str, rows: list[dict]) -> int:
    """Insère/remplace des lignes par clé ; écrit CSV (+ Parquet si possible).

    Idempotent : une clé déjà présente est remplacée par la nouvelle valeur
    (dédup ``keep='last'``), jamais dupliquée. Retourne le nombre de lignes
    nouvelles (clés absentes auparavant).
    """
    os.makedirs(results_dir, exist_ok=True)
    current = load(results_dir)
    before_keys = existing_keys(current)

    incoming = pd.DataFrame(rows)
    for col in SCHEMA:
        if col not in incoming.columns:
            incoming[col] = pd.NA
    incoming = incoming[SCHEMA]

    combined = pd.concat([current, incoming], ignore_index=True)
    combined = combined.drop_duplicates(subset=KEY, keep="last")
    combined = combined.sort_values(KEY).reset_index(drop=True)

    combined.to_csv(_csv_path(results_dir), index=False)
    try:
        combined.to_parquet(_parquet_path(results_dir), index=False)
    except Exception:
        pass  # pyarrow/fastparquet absent → CSV seul fait foi

    n_new = len(existing_keys(combined) - before_keys)
    return n_new


def make_row(
    *, model: str, trainer: str, dataset: int, fold: int, scenario: str,
    case: str, eval_kind: str, metrics: dict,
    epochs: int | None = None, checkpoint_mtime: float | None = None,
) -> dict:
    """Construit une ligne conforme au schéma, provenance auto-remplie."""
    row = {
        "model": model, "trainer": trainer, "dataset": int(dataset),
        "fold": int(fold), "scenario": scenario, "case": case,
        "eval_kind": eval_kind,
        "epochs": epochs, "checkpoint_mtime": checkpoint_mtime,
        "git_commit": git_commit(), "nnunet_version": nnunet_version(),
        "eval_ts": now_iso(),
    }
    for m in METRICS:
        row[m] = metrics.get(m)
    return row
