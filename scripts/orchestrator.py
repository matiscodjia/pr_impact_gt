#!/usr/bin/env python3
"""Orchestrateur reprenable et résilient des entraînements nnU-Net.

Conçu pour un **GPU unique partagé** sujet aux coupures : chaque unité de travail
``(dataset, trainer, fold)`` est idempotente et **reprend là où elle s'est
arrêtée** (checkpoint nnU-Net + ``--c``). Un ledger persistant (``results/
ledger.json``) garde l'état de chaque unité ; relancer le script ne refait jamais
le travail terminé.

Propriété *anytime* : les folds sont **entrelacés** au sein d'un tier
(fold 0 des 3 modèles → fold 1 des 3 modèles → …), et après **chaque** fold
terminé la collecte + le rapport sont rafraîchis → un résultat comparatif complet
arrive au plus tôt, puis sa précision statistique croît.

Reprise par unité :
  - ``checkpoint_final.pth`` + ``validation/`` présents → ``done`` (skip).
  - ``checkpoint_latest.pth`` présent → reprise ``nnUNetv2_train … --c``.
  - sinon → entraînement neuf.
Les crashs (OOM GPU partagé, kill, reboot) sont absorbés par une boucle
retry/back-off qui repart systématiquement du dernier checkpoint.

Usage
-----
    # Tier A (3 modèles cœur), tous les folds, sur GPU
    python scripts/orchestrator.py --config configs/experiment_config.yaml

    # Dry-run rapide (DEBUG_PIPELINE=1 → 2 époques) pour valider la mécanique
    python scripts/orchestrator.py --debug --tiers A

    # Reprendre / état seulement
    python scripts/orchestrator.py --status
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

# Parsing du flux nnUNet pour piloter la barre de progression.
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
    print(f"\n[orchestrator] signal {signum} reçu — arrêt propre après l'unité courante…")


# ─────────────────────────────────────────────────────────────────
# Ledger
# ─────────────────────────────────────────────────────────────────

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
    os.replace(tmp, ledger_path(results_dir))  # écriture atomique


def unit_id(dataset: int, trainer: str, fold: int) -> str:
    return f"d{dataset}_{trainer}_f{fold}"


# ─────────────────────────────────────────────────────────────────
# Résolution & état nnU-Net
# ─────────────────────────────────────────────────────────────────

def results_fold_dir(nnunet_results: str, dataset: int, trainer: str, cfg: str, fold: int) -> str | None:
    base = sorted(glob.glob(os.path.join(nnunet_results, f"Dataset{dataset:03d}_*")))
    if not base:
        return None
    return os.path.join(base[0], f"{trainer}__{PLANS}__{cfg}", f"fold_{fold}")


def _progress_epoch(fold_dir: str) -> int:
    """Dernière epoch atteinte d'après les training_log (−1 si aucun)."""
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
    """``done`` | ``resumable`` | ``fresh`` selon les checkpoints ET le budget.

    Un run avec ``checkpoint_final`` n'est ``done`` que s'il a réellement atteint
    ``expected_epochs`` : un run court (ex. ``--debug``) ou un budget revu à la
    hausse n'est pas pris pour terminé — il est repris et prolongé via ``--c``
    (les époques déjà faites sont conservées).
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


# ─────────────────────────────────────────────────────────────────
# Prérequis : déploiement des trainers + splits partagés
# ─────────────────────────────────────────────────────────────────

def deploy_trainers() -> None:
    """Copie le module custom (tous les trainers + variantes) dans le package nnU-Net."""
    try:
        import nnunetv2
        dst_dir = os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer")
        src = os.path.join(SCRIPT_DIR, "..", "custom_trainers", "nnUNetTrainerDegraded.py")
        shutil.copy(os.path.abspath(src), dst_dir)
        print(f"[orchestrator] trainers déployés → {dst_dir}")
    except Exception as e:
        print(f"[orchestrator] WARN déploiement trainers impossible: {e}")


def ensure_shared_splits(nnunet_preprocessed: str, src_id: int, dst_id: int) -> None:
    """Copie splits_final.json de Dataset{src} → Dataset{dst} (partitions identiques).

    Garantit que les comparaisons appariées OOF (Star vs Minus_Fixed) portent sur
    exactement les mêmes cas par fold. No-op si dst == src ou si la source manque.
    """
    if dst_id == src_id:
        return
    src = sorted(glob.glob(os.path.join(nnunet_preprocessed, f"Dataset{src_id:03d}_*")))
    dst = sorted(glob.glob(os.path.join(nnunet_preprocessed, f"Dataset{dst_id:03d}_*")))
    if not src or not dst:
        print(f"[orchestrator] WARN splits partagés impossibles (src={bool(src)}, dst={bool(dst)})")
        return
    src_splits = os.path.join(src[0], "splits_final.json")
    dst_splits = os.path.join(dst[0], "splits_final.json")
    if not os.path.exists(src_splits):
        print(f"[orchestrator] WARN {src_splits} absent — splits non partagés")
        return
    shutil.copy(src_splits, dst_splits)
    print(f"[orchestrator] splits partagés Dataset{src_id} → Dataset{dst_id}")


# ─────────────────────────────────────────────────────────────────
# Entraînement d'une unité (avec reprise + retry)
# ─────────────────────────────────────────────────────────────────

def expected_epochs(config: dict, trainer: str, debug: bool) -> int | None:
    """Nombre d'epochs attendu (pour la barre) selon le contexte du trainer."""
    if debug:
        return 4 if "Calib" in trainer else 2
    if "Calib" in trainer:
        return int(config.get("calibration", {}).get("budget", 1000))
    if trainer.startswith("nnUNetTrainerDeg_"):  # variantes grid search
        return int(config.get("grid_search", {}).get("num_epochs", 1000))
    return int(config.get("training", {}).get("num_epochs", 1000))


def run_streamed(cmd, log_path, total_epochs, label, show_progress) -> int:
    """Lance nnUNet, tee la sortie dans un log, pilote une barre d'epochs.

    La sortie complète va dans ``log_path`` (rien n'est perdu) ; seules une barre
    de progression (epochs + ETA + pseudo-Dice) et les alertes (OOM/erreurs)
    apparaissent à l'écran.
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    bar = tqdm(total=total_epochs, desc=label, unit="ep", leave=False,
               dynamic_ncols=True, disable=not show_progress)
    last_dice = None
    with open(log_path, "a") as log:
        for line in proc.stdout:
            log.write(line)
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
                tqdm.write(f"  ⚠ {line.rstrip()}")
    proc.wait()
    if total_epochs and proc.returncode == 0:
        bar.n = total_epochs
        bar.refresh()
    bar.close()
    return proc.returncode


def train_unit(dataset, trainer, fold, cfg, device, fold_dir, max_attempts, backoff,
               total_epochs, label, log_path, show_progress):
    """Entraîne une unité, reprise auto, retry/back-off. Retourne l'état final."""
    rc = -1
    for attempt in range(1, max_attempts + 1):
        state = fold_state(fold_dir, total_epochs)
        if state == "done":
            return "done", 0, None
        resume = state == "resumable"
        cmd = ["nnUNetv2_train", str(dataset), cfg, str(fold),
               "-tr", trainer, "--npz", "-device", device]
        if resume:
            cmd.append("--c")
        tag = "reprise --c" if resume else "neuf"
        tqdm.write(f"  [tentative {attempt}/{max_attempts}, {tag}] log → {log_path}")
        try:
            rc = run_streamed(cmd, log_path, total_epochs,
                              f"{label} [{tag}]", show_progress)
        except FileNotFoundError as e:
            return "failed", -1, f"nnUNetv2_train introuvable: {e}"
        if rc == 0 and fold_state(fold_dir) == "done":
            return "done", 0, None
        if _INTERRUPTED:
            return "interrupted", rc, "signal reçu"
        tqdm.write(f"  échec (rc={rc}) — back-off {backoff}s puis reprise depuis le checkpoint")
        time.sleep(backoff)
    return "failed", rc, f"épuisé après {max_attempts} tentatives"


# ─────────────────────────────────────────────────────────────────
# File de travail (entrelacée par fold au sein d'un tier)
# ─────────────────────────────────────────────────────────────────

def build_queue(config, tiers, folds_override, models_override):
    models = config["experiment"]["models"]
    if models_override:
        models = [m for m in models if m["name"] in models_override]
    models = [m for m in models if m["tier"] in tiers]
    folds = folds_override or config["training"]["folds"]
    # Entrelacement : fold 0 de tous les modèles, puis fold 1, etc.
    queue = []
    for fold in folds:
        for m in models:
            queue.append((m, fold))
    return queue, models


def refresh_outputs(config_path, results_dir):
    """Collecte incrémentale + rapport (rafraîchissement anytime)."""
    for script in ("collect_metrics.py", "report.py"):
        arg = (["--config", config_path] if script == "collect_metrics.py"
               else [])
        out = (["--output", results_dir] if script == "collect_metrics.py"
               else ["--results_dir", results_dir])
        subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, script), *arg, *out])


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Orchestrateur reprenable nnU-Net")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--tiers", default="A", help="tiers à exécuter, ex: A ou ABC")
    parser.add_argument("--folds", type=int, nargs="*", help="override des folds")
    parser.add_argument("--models", nargs="*", help="override des modèles (par nom)")
    parser.add_argument("--device", default=None, help="cuda|mps|cpu (auto si absent)")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--backoff", type=int, default=30, help="secondes entre tentatives")
    parser.add_argument("--debug", action="store_true", help="DEBUG_PIPELINE=1 → run minuscule")
    parser.add_argument("--status", action="store_true", help="affiche l'état et quitte")
    parser.add_argument("--dry-run", action="store_true", help="liste la file sans entraîner")
    parser.add_argument("--calibrate", action="store_true",
                        help="run de calibration (1 fold long + snapshots) puis calibrate.py")
    parser.add_argument("--no-progress", action="store_true",
                        help="désactive la barre de progression (logs bruts)")
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

    # ── Mode statut ──
    if args.status:
        queue, _ = build_queue(config, list("ABC"), args.folds, args.models)
        print(f"État ({len(ledger['units'])} unités au ledger) :")
        for m, fold in queue:
            uid = unit_id(m["dataset_id"], m["trainer"], fold)
            fd = results_fold_dir(nnunet_results, m["dataset_id"], m["trainer"], cfg, fold)
            print(f"  {uid:<42} disque={fold_state(fd):<10} "
                  f"ledger={ledger['units'].get(uid, {}).get('state', '—')}")
        return

    # ── Mode calibration : 1 fold long reprenable + évaluation des snapshots ──
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
              f"jalons={cal['milestones']}) device={device}")
        total = expected_epochs(config, cal["trainer"], args.debug)
        log_path = os.path.join(args.results_dir, "logs", f"{uid}.log")
        state, rc, err = train_unit(cal["dataset_id"], cal["trainer"], cal["fold"], cfg,
                                    device, fd, args.max_attempts, args.backoff,
                                    total, uid, log_path, show_progress)
        print(f"  → calibration {uid}: {state}" + (f" ({err})" if err else ""))
        if state == "done":
            subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "calibrate.py"),
                            "--config", args.config, "--results_dir", args.results_dir])
        return

    queue, models = build_queue(config, tiers, args.folds, args.models)

    if args.dry_run:
        print(f"File ({len(queue)} unités, tiers={tiers}, entrelacée par fold) :")
        for m, fold in queue:
            print(f"  {unit_id(m['dataset_id'], m['trainer'], fold)}  ({m['name']})")
        return

    if args.debug:
        os.environ["DEBUG_PIPELINE"] = "1"
        print("[orchestrator] DEBUG_PIPELINE=1 → 2 époques par fold")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    device = detect_device(args.device)
    print(f"[orchestrator] device={device} | tiers={tiers} | {len(queue)} unités")

    deploy_trainers()
    # Splits partagés pour tout dataset != source (avant son 1er entraînement).
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
            tqdm.write(f"[{i}/{len(queue)}] ✓ {uid} déjà terminé — skip")
            continue

        tqdm.write(f"\n[{i}/{len(queue)}] ▶ {uid}  ({m['name']}, fold {fold})")
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
            total, f"[{i}/{len(queue)}] {m['name']} f{fold}", log_path, show_progress)

        u["attempts"] = u.get("attempts", 0) + 1
        u["state"] = state
        u["last_exit"] = rc
        u["last_error"] = err
        u["updated_at"] = _now()
        ledger["units"][uid] = u
        save_ledger(args.results_dir, ledger)
        print(f"  → {uid}: {state}" + (f" ({err})" if err else ""))

        # Rafraîchissement anytime après chaque fold terminé.
        if state == "done":
            refresh_outputs(args.config, args.results_dir)

    # Rafraîchissement final
    refresh_outputs(args.config, args.results_dir)
    done = sum(1 for u in ledger["units"].values() if u.get("state") == "done")
    print(f"\n[orchestrator] terminé — {done}/{len(ledger['units'])} unités done. "
          f"Voir {os.path.join(args.results_dir, 'STATUS.md')}")


if __name__ == "__main__":
    main()
