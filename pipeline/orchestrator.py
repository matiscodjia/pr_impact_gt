#!/usr/bin/env python3
"""Resumable, fault-tolerant driver of the nnU-Net trainings.

Built for a single shared GPU that can be interrupted: each unit of work
``(dataset, trainer, fold)`` is idempotent and resumes where it stopped (nnU-Net
checkpoint + ``--c``). A persistent ledger (``<results_dir>/ledger.json``) records the
state of every unit; rerunning the script never redoes finished work.

Results arrive as early as possible: within a tier the folds are interleaved (fold 0 of
every model, then fold 1, ...), and after EACH finished fold the scores are collected and
the report refreshed, so a complete comparison exists early and its precision then grows.

Per unit:
  - ``checkpoint_final.pth`` and ``validation/`` present -> ``done`` (skipped);
  - ``checkpoint_latest.pth`` present -> resumed with ``nnUNetv2_train ... --c``;
  - otherwise -> trained from scratch.
Crashes (out of memory on a shared GPU, kill, reboot) are absorbed by a retry loop with
back-off that always restarts from the last checkpoint.

    # Tier A (the 3 core models), all folds, on GPU
    python pipeline/orchestrator.py --config configs/experiment_config.yaml

    # Quick dry run (DEBUG_PIPELINE=1 -> 2 epochs) to check the mechanics
    python pipeline/orchestrator.py --debug --tiers A

    # State only
    python pipeline/orchestrator.py --status
"""

import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml
from tqdm import tqdm

PLANS = "nnUNetPlans"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Patterns of the nnU-Net output that drive the progress bar.
_EPOCH_RE = re.compile(r"\bepoch\s+(\d+)", re.IGNORECASE)
_DICE_RE = re.compile(r"Pseudo dice\s*\[([^\]]*)\]")
_FLOAT_RE = re.compile(r"[-+]?\d*\.\d+")
_ALERT_RE = re.compile(r"error|exception|traceback|out of memory|cuda error|killed|runtimeerror",
                       re.IGNORECASE)

_INTERRUPTED = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _handle_signal(signum, frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    print(f"\n[orchestrator] signal {signum} received: clean stop after the current unit...")


# ---------------------------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------------------------

def ledger_path(results_dir: str) -> str:
    return os.path.join(results_dir, "ledger.json")


def load_ledger(results_dir: str) -> dict:
    path = ledger_path(results_dir)
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return {"updated": _now(), "units": {}}


def save_ledger(results_dir: str, ledger: dict) -> None:
    os.makedirs(results_dir, exist_ok=True)
    ledger["updated"] = _now()
    tmp = ledger_path(results_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ledger, f, indent=2)
    os.replace(tmp, ledger_path(results_dir))  # atomic write


def unit_id(dataset: int, trainer: str, fold: int) -> str:
    return f"d{dataset}_{trainer}_f{fold}"


# ---------------------------------------------------------------------------------------------
# nnU-Net paths and fold state
# ---------------------------------------------------------------------------------------------

def results_fold_dir(nnunet_results: str, dataset: int, trainer: str, cfg: str, fold: int) -> str | None:
    base = sorted(glob.glob(os.path.join(nnunet_results, f"Dataset{dataset:03d}_*")))
    if not base:
        return None
    return os.path.join(base[0], f"{trainer}__{PLANS}__{cfg}", f"fold_{fold}")


def _progress_epoch(fold_dir: str) -> int:
    """Last epoch reached according to the training logs (-1 if none)."""
    last = -1
    for log in glob.glob(os.path.join(fold_dir, "training_log_*.txt")):
        try:
            with open(log, errors="ignore") as f:
                for line in f:
                    m = _EPOCH_RE.search(line)
                    if m:
                        last = max(last, int(m.group(1)))
        except OSError:
            pass
    return last


def fold_state(fold_dir: str | None, expected_epochs: int | None = None) -> str:
    """``done`` | ``resumable`` | ``fresh``, from the checkpoints AND the epoch budget.

    A run with ``checkpoint_final`` is only ``done`` if it actually reached
    ``expected_epochs``: a short run (e.g. ``--debug``) or a raised budget is not taken as
    finished; it is resumed and extended with ``--c`` (epochs already done are kept).
    """
    if fold_dir is None:
        return "fresh"
    has_final = os.path.exists(os.path.join(fold_dir, "checkpoint_final.pth"))
    has_latest = os.path.exists(os.path.join(fold_dir, "checkpoint_latest.pth"))
    has_val = os.path.isdir(os.path.join(fold_dir, "validation"))
    reached = expected_epochs is None or (_progress_epoch(fold_dir) + 1) >= expected_epochs
    if has_final and has_val and reached:
        return "done"
    if has_final or has_latest:
        return "resumable"
    return "fresh"


def detect_device(override: str | None) -> str:
    if override:
        return override
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------------------------
# Prerequisites: trainer deployment and shared splits
# ---------------------------------------------------------------------------------------------

def deploy_trainers() -> None:
    """Copy the custom trainers module into the nnU-Net package, where nnU-Net finds them.

    The trainers import the noise operator from ``prgt``, which must be installed in the
    same environment (``pip install -e . --no-deps``).
    """
    try:
        import nnunetv2
        dst_dir = os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer")
        src = os.path.join(SCRIPT_DIR, "..", "nnunet_trainers", "nnUNetTrainerDegraded.py")
        shutil.copy(os.path.abspath(src), dst_dir)
        print(f"[orchestrator] trainers deployed -> {dst_dir}")
    except Exception as e:
        print(f"[orchestrator] WARN could not deploy the trainers: {e}")


def ensure_shared_splits(nnunet_preprocessed: str, src_id: int, dst_id: int) -> None:
    """Copy splits_final.json from Dataset{src} to Dataset{dst} (identical folds).

    Paired out-of-fold comparisons between models trained on different datasets then use
    exactly the same cases in each fold. No-op if dst == src or if the source is missing.
    """
    if dst_id == src_id:
        return
    src = sorted(glob.glob(os.path.join(nnunet_preprocessed, f"Dataset{src_id:03d}_*")))
    dst = sorted(glob.glob(os.path.join(nnunet_preprocessed, f"Dataset{dst_id:03d}_*")))
    if not src or not dst:
        print(f"[orchestrator] WARN cannot share the splits (src={bool(src)}, dst={bool(dst)})")
        return
    src_splits = os.path.join(src[0], "splits_final.json")
    dst_splits = os.path.join(dst[0], "splits_final.json")
    if not os.path.exists(src_splits):
        print(f"[orchestrator] WARN {src_splits} missing: splits not shared")
        return
    shutil.copy(src_splits, dst_splits)
    print(f"[orchestrator] splits shared Dataset{src_id} -> Dataset{dst_id}")


# ---------------------------------------------------------------------------------------------
# Training one unit (resume + retry)
# ---------------------------------------------------------------------------------------------

def expected_epochs(config: dict, trainer: str, debug: bool) -> int | None:
    """Expected number of epochs of a trainer (progress bar and "done" check)."""
    if debug:
        return 4 if "Calib" in trainer else 2
    if "Calib" in trainer:
        return int(config.get("calibration", {}).get("budget", 1000))
    return int(config.get("training", {}).get("num_epochs", 1000))


def run_streamed(cmd, log_path, total_epochs, label, show_progress) -> int:
    """Run nnU-Net, copy its output to a log file and drive an epoch progress bar.

    The full output goes to ``log_path`` (nothing is lost); only a progress bar (epochs,
    ETA, pseudo-Dice) and alerts (out of memory, errors) are shown on screen.
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    # PYTHONUNBUFFERED=1: nnU-Net flushes stdout at every line instead of buffering blocks
    # into the pipe, so the bar moves in real time.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)
    bar = tqdm(total=total_epochs, desc=label, unit="ep", leave=False,
               dynamic_ncols=True, disable=not show_progress)
    last_dice = None
    with open(log_path, "a") as log:
        # readline (not `for line in proc.stdout`) avoids a read-ahead that would hold
        # lines back on the parent side.
        for line in iter(proc.stdout.readline, ""):
            log.write(line)
            log.flush()
            me = _EPOCH_RE.search(line)
            if me:
                ep = int(me.group(1))
                bar.n = min(ep, total_epochs) if total_epochs else ep
                if last_dice is not None:
                    bar.set_postfix(dice=last_dice)
                bar.refresh()
                continue
            md = _DICE_RE.search(line)
            if md:
                vals = _FLOAT_RE.findall(md.group(1))
                if vals:
                    last_dice = vals[0]
                continue
            if show_progress and _ALERT_RE.search(line):
                tqdm.write(f"  ! {line.rstrip()}")
    proc.stdout.close()
    proc.wait()
    if total_epochs and proc.returncode == 0:
        bar.n = total_epochs
        bar.refresh()
    bar.close()
    return proc.returncode


def train_unit(dataset, trainer, fold, cfg, device, fold_dir, max_attempts, backoff,
               total_epochs, label, log_path, show_progress, save_npz=True):
    """Train one unit with automatic resume and retry/back-off. Returns (state, rc, error)."""
    rc = -1
    for attempt in range(1, max_attempts + 1):
        state = fold_state(fold_dir, total_epochs)
        if state == "done":
            return "done", 0, None
        resume = state == "resumable"
        # --npz writes the validation softmax probabilities (~2/3 of the size of a fold); the
        # analyses only use the .nii.gz (argmax), which --no-npz does not change.
        cmd = ["nnUNetv2_train", str(dataset), cfg, str(fold), "-tr", trainer]
        if save_npz:
            cmd.append("--npz")
        cmd += ["-device", device]
        if resume:
            cmd.append("--c")
        tag = "resume --c" if resume else "fresh"
        tqdm.write(f"  [attempt {attempt}/{max_attempts}, {tag}] log -> {log_path}")
        try:
            rc = run_streamed(cmd, log_path, total_epochs,
                              f"{label} [{tag}]", show_progress)
        except FileNotFoundError as e:
            return "failed", -1, f"nnUNetv2_train not found: {e}"
        if rc == 0 and fold_state(fold_dir) == "done":
            return "done", 0, None
        if _INTERRUPTED:
            return "interrupted", rc, "signal received"
        tqdm.write(f"  failed (rc={rc}): back-off {backoff}s, then resume from the checkpoint")
        time.sleep(backoff)
    return "failed", rc, f"gave up after {max_attempts} attempts"


# ---------------------------------------------------------------------------------------------
# Work queue (folds interleaved within a tier)
# ---------------------------------------------------------------------------------------------

def build_queue(config, tiers, folds_override, models_override):
    models = config["experiment"]["models"]
    if models_override:
        models = [m for m in models if m["name"] in models_override]
    models = [m for m in models if m["tier"] in tiers]
    folds = folds_override or config["training"]["folds"]
    # Interleaving: fold 0 of every model, then fold 1, and so on.
    queue = []
    for fold in folds:
        for m in models:
            queue.append((m, fold))
    return queue, models


def refresh_outputs(config_path, results_dir):
    """Incremental scoring, then the report (refreshed after every finished fold)."""
    for script in ("collect_metrics.py", "report.py"):
        arg = (["--config", config_path] if script == "collect_metrics.py"
               else [])
        out = (["--output", results_dir] if script == "collect_metrics.py"
               else ["--results_dir", results_dir])
        subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, script), *arg, *out])


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Resumable driver of the nnU-Net trainings")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--tiers", default="A", help="tiers to run, e.g. A or ABC")
    parser.add_argument("--folds", type=int, nargs="*", help="override the folds")
    parser.add_argument("--models", nargs="*", help="override the models (by name)")
    parser.add_argument("--device", default=None, help="cuda|mps|cpu (auto if absent)")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--backoff", type=int, default=30, help="seconds between attempts")
    parser.add_argument("--debug", action="store_true", help="DEBUG_PIPELINE=1 -> tiny run")
    parser.add_argument("--status", action="store_true", help="print the state and exit")
    parser.add_argument("--dry-run", action="store_true", help="list the queue without training")
    parser.add_argument("--calibrate", action="store_true",
                        help="calibration run (one long fold + snapshots), then calibrate_epochs.py")
    parser.add_argument("--no-npz", action="store_true",
                        help="do not write the validation softmax probabilities (saves space and time)")
    parser.add_argument("--no-progress", action="store_true",
                        help="no progress bar (raw logs)")
    args = parser.parse_args()
    show_progress = not args.no_progress

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cfg = config["training"]["configurations"][0]
    src_id = config["dataset"]["id"]
    nnunet_results = os.environ.get("nnUNet_results", "nnUNet_data/nnUNet_results")
    nnunet_preprocessed = os.environ.get("nnUNet_preprocessed", "nnUNet_data/nnUNet_preprocessed")
    tiers = list(args.tiers)

    ledger = load_ledger(args.results_dir)

    # status mode
    if args.status:
        queue, _ = build_queue(config, list("ABC"), args.folds, args.models)
        print(f"State ({len(ledger['units'])} units in the ledger):")
        for m, fold in queue:
            uid = unit_id(m["dataset_id"], m["trainer"], fold)
            fd = results_fold_dir(nnunet_results, m["dataset_id"], m["trainer"], cfg, fold)
            print(f"  {uid:<42} disk={fold_state(fd):<10} "
                  f"ledger={ledger['units'].get(uid, {}).get('state', '-')}")
        return

    # calibration mode: one long resumable fold, then the snapshots are scored
    if args.calibrate:
        cal = config["calibration"]
        if args.debug:
            os.environ["DEBUG_PIPELINE"] = "1"
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        device = detect_device(args.device)
        deploy_trainers()
        ensure_shared_splits(nnunet_preprocessed, src_id, cal["dataset_id"])
        fd = results_fold_dir(nnunet_results, cal["dataset_id"], cal["trainer"], cfg, cal["fold"])
        uid = unit_id(cal["dataset_id"], cal["trainer"], cal["fold"])
        print(f"[orchestrator] CALIBRATION {uid} (budget={cal['budget']} epochs, "
              f"milestones={cal['milestones']}) device={device}")
        total = expected_epochs(config, cal["trainer"], args.debug)
        log_path = os.path.join(args.results_dir, "logs", f"{uid}.log")
        state, rc, err = train_unit(cal["dataset_id"], cal["trainer"], cal["fold"], cfg,
                                    device, fd, args.max_attempts, args.backoff,
                                    total, uid, log_path, show_progress)
        print(f"  -> calibration {uid}: {state}" + (f" ({err})" if err else ""))
        if state == "done":
            subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "calibrate_epochs.py"),
                            "--config", args.config, "--results_dir", args.results_dir])
        return

    queue, models = build_queue(config, tiers, args.folds, args.models)

    if args.dry_run:
        print(f"Queue ({len(queue)} units, tiers={tiers}, interleaved by fold):")
        for m, fold in queue:
            print(f"  {unit_id(m['dataset_id'], m['trainer'], fold)}  ({m['name']})")
        return

    if args.debug:
        os.environ["DEBUG_PIPELINE"] = "1"
        print("[orchestrator] DEBUG_PIPELINE=1 -> 2 epochs per fold")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    device = detect_device(args.device)
    print(f"[orchestrator] device={device} | tiers={tiers} | {len(queue)} units")

    deploy_trainers()
    # Shared splits for every dataset other than the source (before its first training).
    for dst in sorted({m["dataset_id"] for m in models}):
        ensure_shared_splits(nnunet_preprocessed, src_id, dst)

    for i, (m, fold) in enumerate(queue, 1):
        if _INTERRUPTED:
            break
        uid = unit_id(m["dataset_id"], m["trainer"], fold)
        fd = results_fold_dir(nnunet_results, m["dataset_id"], m["trainer"], cfg, fold)
        total = expected_epochs(config, m["trainer"], args.debug)

        if fold_state(fd, total) == "done":
            ledger["units"][uid] = {**ledger["units"].get(uid, {}),
                "dataset": m["dataset_id"], "trainer": m["trainer"], "fold": fold,
                "model": m["name"], "state": "done", "updated_at": _now()}
            save_ledger(args.results_dir, ledger)
            tqdm.write(f"[{i}/{len(queue)}] {uid} already done: skipped")
            continue

        tqdm.write(f"\n[{i}/{len(queue)}] > {uid}  ({m['name']}, fold {fold})")
        u = ledger["units"].get(uid, {"attempts": 0})
        u.update({"dataset": m["dataset_id"], "trainer": m["trainer"], "fold": fold,
                  "model": m["name"], "state": "running",
                  "started_at": u.get("started_at", _now()), "updated_at": _now()})
        ledger["units"][uid] = u
        save_ledger(args.results_dir, ledger)

        log_path = os.path.join(args.results_dir, "logs", f"{uid}.log")
        state, rc, err = train_unit(
            m["dataset_id"], m["trainer"], fold, cfg, device, fd,
            args.max_attempts, args.backoff,
            total, f"[{i}/{len(queue)}] {m['name']} f{fold}", log_path, show_progress,
            save_npz=not args.no_npz)

        u["attempts"] = u.get("attempts", 0) + 1
        u["state"] = state
        u["last_exit"] = rc
        u["last_error"] = err
        u["updated_at"] = _now()
        ledger["units"][uid] = u
        save_ledger(args.results_dir, ledger)
        print(f"  -> {uid}: {state}" + (f" ({err})" if err else ""))

        # Refresh the scores and the report after every finished fold.
        if state == "done":
            refresh_outputs(args.config, args.results_dir)

    # final refresh
    refresh_outputs(args.config, args.results_dir)
    done = sum(1 for u in ledger["units"].values() if u.get("state") == "done")
    print(f"\n[orchestrator] finished: {done}/{len(ledger['units'])} units done. "
          f"See {os.path.join(args.results_dir, 'STATUS.md')}")


if __name__ == "__main__":
    main()
