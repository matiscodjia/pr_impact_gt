#!/usr/bin/env python3
"""Courbe de convergence — justification du schedule court (num_epochs).

Parse les journaux d'entraînement nnU-Net (``training_log_*.txt`` de chaque fold)
pour extraire le **pseudo-Dice de validation par epoch**, et trace la courbe par
fold + moyenne. Une ligne verticale marque ``training.num_epochs`` : si la courbe
a atteint son plateau avant, le schedule court est justifié (à mettre en annexe du
papier).

Usage
-----
    python scripts/plot_convergence.py --config configs/experiment_config.yaml \
        --trainer nnUNetTrainer250 --dataset 100 --out results/figures/convergence.png
"""

import argparse
import glob
import os
import re

import numpy as np
import yaml

PLANS = "nnUNetPlans"
_EPOCH_RE = re.compile(r"\bepoch\s+(\d+)", re.IGNORECASE)
_DICE_RE = re.compile(r"Pseudo dice\s*\[([^\]]*)\]")
# Flottants à virgule uniquement : évite de capter le « 32 » de « np.float32(...) ».
_FLOAT_RE = re.compile(r"[-+]?\d*\.\d+")


def parse_log(path: str) -> dict[int, float]:
    """Mappe epoch → pseudo-Dice moyen depuis un training_log nnU-Net."""
    out: dict[int, float] = {}
    cur = None
    with open(path, errors="ignore") as f:
        for line in f:
            me = _EPOCH_RE.search(line)
            if me:
                cur = int(me.group(1))
                continue
            md = _DICE_RE.search(line)
            if md and cur is not None:
                vals = [float(x) for x in _FLOAT_RE.findall(md.group(1))]
                if vals:
                    out[cur] = float(np.mean(vals))
    return out


def fold_series(results_dir: str, dataset: int, trainer: str, cfg: str) -> dict[int, dict[int, float]]:
    """Pour chaque fold : la série epoch→dice (concat des logs, dernier gagne)."""
    base = sorted(glob.glob(os.path.join(results_dir, f"Dataset{dataset:03d}_*")))
    if not base:
        return {}
    tdir = os.path.join(base[0], f"{trainer}__{PLANS}__{cfg}")
    series: dict[int, dict[int, float]] = {}
    for fold_dir in sorted(glob.glob(os.path.join(tdir, "fold_*"))):
        fold = int(os.path.basename(fold_dir).split("_")[1])
        merged: dict[int, float] = {}
        for log in sorted(glob.glob(os.path.join(fold_dir, "training_log_*.txt"))):
            merged.update(parse_log(log))
        if merged:
            series[fold] = merged
    return series


def main():
    parser = argparse.ArgumentParser(description="Courbe de convergence nnU-Net")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--trainer", default="nnUNetTrainer250")
    parser.add_argument("--dataset", type=int, default=100)
    parser.add_argument("--out", default="results/figures/convergence.png")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    cfg = config["training"]["configurations"][0]
    num_epochs = config["training"].get("num_epochs")
    nnunet_results = os.environ.get("nnUNet_results", "nnUNet_data/nnUNet_results")

    series = fold_series(nnunet_results, args.dataset, args.trainer, cfg)
    if not series:
        print(f"[SKIP] aucun training_log trouvé pour {args.trainer} (dataset {args.dataset}).")
        return

    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    all_epochs = sorted({e for s in series.values() for e in s})
    for fold, s in sorted(series.items()):
        xs = sorted(s)
        ax.plot(xs, [s[e] for e in xs], alpha=0.45, lw=1, label=f"fold {fold}")

    # Moyenne inter-folds par epoch
    mean = [np.mean([s[e] for s in series.values() if e in s]) for e in all_epochs]
    ax.plot(all_epochs, mean, color="black", lw=2.2, label="moyenne")

    if num_epochs:
        ax.axvline(num_epochs, color="red", ls="--", lw=1.2,
                   label=f"num_epochs={num_epochs}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pseudo-Dice de validation (EMA nnU-Net)")
    ax.set_title(f"Convergence — {args.trainer} (Dataset{args.dataset:03d})\n"
                 "justification du schedule court")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Courbe de convergence → {args.out}")


if __name__ == "__main__":
    main()
