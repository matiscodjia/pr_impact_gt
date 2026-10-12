"""Master table of per-case scores: long format, idempotent writes, provenance on every row.

Every score produced by the pipeline (out-of-fold and test set) lives in one long-format
table, ``<results_dir>/metrics.csv``, with one row per
``(model, trainer, dataset, fold, scenario, case, eval_kind)``. Each row carries its
provenance (epochs, checkpoint time, git commit, nnU-Net version, timestamp), so the
maturity of every number can be read without outside context.

Writes are an idempotent upsert: collecting again after an interruption never duplicates
a key (an existing key is replaced), so the pipeline can be replayed safely.

Usage
-----
    from prgt import results_store as rs
    df = rs.load("results")
    done = rs.existing_keys(df)            # set of key tuples already computed
    rs.upsert("results", [row1, row2])     # each row is a dict covering rs.SCHEMA
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone

import pandas as pd

# Canonical schema.
#   nsd          = NSD at 2 mm (clinical tolerance); nsd05 = NSD at 0.5 mm (sub-voxel,
#                  sensitive to boundary drift)
#   volume_delta = signed relative volume difference (direction of a segmentation bias)
METRICS = ["cldice", "hd95", "nsd", "nsd05", "betti0", "volume_delta"]
KEY = ["model", "trainer", "dataset", "fold", "scenario", "case", "eval_kind"]
PROVENANCE = ["epochs", "checkpoint_mtime", "git_commit", "nnunet_version", "eval_ts"]
SCHEMA = KEY + METRICS + PROVENANCE

CSV_NAME = "metrics.csv"


# ---------------------------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------------------------

def git_commit() -> str:
    """Short hash of the current commit ("unknown" outside a git repository)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def nnunet_version() -> str:
    """Installed nnU-Net version ("unknown" if absent)."""
    try:
        import nnunetv2
        return getattr(nnunetv2, "__version__", "unknown")
    except Exception:
        return "unknown"


def now_iso() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------------------------

def _csv_path(results_dir: str) -> str:
    return os.path.join(results_dir, CSV_NAME)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMA)


def load(results_dir: str) -> pd.DataFrame:
    """Load the master table (an empty frame with the schema if it does not exist)."""
    path = _csv_path(results_dir)
    if not os.path.exists(path):
        return _empty()
    df = pd.read_csv(path)
    for col in SCHEMA:   # tables written by older versions may lack columns
        if col not in df.columns:
            df[col] = pd.NA
    return df[SCHEMA]


def key_of(row: dict) -> tuple:
    """Key tuple of one row."""
    return tuple(row[k] for k in KEY)


def existing_keys(df: pd.DataFrame) -> set[tuple]:
    """Set of keys already present, to skip scores that are already computed."""
    if df.empty:
        return set()
    return set(map(tuple, df[KEY].astype(object).values.tolist()))


def upsert(results_dir: str, rows: list[dict]) -> int:
    """Insert or replace rows by key and write the CSV.

    A key that is already present is replaced by the new row (``keep="last"``), never
    duplicated. Returns the number of new keys.
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

    return len(existing_keys(combined) - before_keys)


def make_row(
    *, model: str, trainer: str, dataset: int, fold: int, scenario: str,
    case: str, eval_kind: str, metrics: dict,
    epochs: int | None = None, checkpoint_mtime: float | None = None,
) -> dict:
    """Build one row of the schema; provenance columns are filled automatically."""
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


def pooled(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """One row per (model, scenario, case) for one ``eval_kind``, with the metric columns.

    Out of fold, each case is predicted by exactly one fold network, so this is one row
    per case; duplicates are dropped as a safeguard (keep the last).
    """
    sub = df[df["eval_kind"] == kind].copy()
    if sub.empty:
        return sub
    sub = sub.drop_duplicates(subset=["model", "scenario", "case"], keep="last")
    return sub[["model", "scenario", "case"] + METRICS]
