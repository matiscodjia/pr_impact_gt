#!/usr/bin/env python3
"""Avancement précis des entraînements (réplicats de graine) : ce que --status ne montre pas.

Lit les training_log_*.txt de nnU-Net (source de vérité, y compris après reprise --c) et
les logs de l'orchestrateur (taille de la file de chaque processus). Par pli : époque
atteinte / budget, s/époque (médiane des 20 dernières), pseudo-Dice, EMA, âge de la
dernière ligne, fin estimée ; par modèle : fin estimée de toute sa file.

À lancer SUR le worker, racine du repo :
    python scripts/cluster/progress.py                  # réplicats *_s1 / *_s2
    python scripts/cluster/progress.py --pattern '*'    # tous les trainers
    watch -n 60 python scripts/cluster/progress.py      # rafraîchi chaque minute
    python scripts/cluster/progress.py --datasets 'Dataset10[34]_*' --pattern nnUNetTrainerStd  # M3/M4
"""
import argparse
import glob
import os
import re
import statistics
import subprocess
from datetime import datetime, timedelta

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+: (.*)$")
EPOCH = re.compile(r"^Epoch (\d+)\s*$")
EP_TIME = re.compile(r"^Epoch time: ([\d.]+) s")
DICE = re.compile(r"^Pseudo dice \[(.*)\]")
EMA = re.compile(r"New best EMA pseudo Dice: ([\d.]+)")
BUDGET = re.compile(r"(?:epochs\s*:|num_epochs set to)\s*(\d+)")
NVAL = re.compile(r"This split has \d+ training and (\d+) validation cases")
QUEUE = re.compile(r"\[(\d+)/(\d+)\] ▶ (\S+)")
VAL_S_PER_CASE = 70.0   # repère mesuré sur les plis M0-M2 d'origine (~17 cas en ~20 min)


def parse_fold(fold_dir, default_budget):
    st = {"budget": default_budget, "n_val": None, "epoch": -1, "times": [], "dice": None,
          "ema": None, "predicted": 0, "val_done": False, "last": None, "val_start": None,
          "cur_done": False}
    logs = sorted(glob.glob(os.path.join(fold_dir, "training_log_*.txt")), key=os.path.getmtime)
    for log in logs:
        with open(log, errors="ignore") as f:
            for line in f:
                m = TS.match(line.rstrip())
                if not m:
                    continue
                t, msg = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), m.group(2).strip()
                st["last"] = t
                if (g := EPOCH.match(msg)):
                    st["epoch"], st["predicted"], st["val_start"] = int(g.group(1)), 0, None
                    st["cur_done"] = False
                elif (g := EP_TIME.match(msg)):
                    st["times"].append(float(g.group(1)))
                    st["cur_done"] = True
                elif (g := DICE.match(msg)):
                    st["dice"] = ",".join(re.findall(r"\d\.\d+", g.group(1))) or None
                elif (g := EMA.search(msg)):
                    st["ema"] = float(g.group(1))
                elif (g := BUDGET.search(msg)):
                    st["budget"] = int(g.group(1))
                elif (g := NVAL.search(msg)):
                    st["n_val"] = int(g.group(1))
                elif msg.startswith("predicting "):
                    st["predicted"] += 1
                    st["val_start"] = st["val_start"] or t
                elif msg.startswith("Validation complete"):
                    st["val_done"] = True
    return st


def remaining_s(st, n_val_default=17):
    """Secondes restantes pour finir ce pli (fin de l'entraînement + validation)."""
    if st["val_done"]:
        return 0.0
    n_val = st["n_val"] or n_val_default
    val_rate = VAL_S_PER_CASE
    if st["predicted"] > 1:
        val_rate = (st["last"] - st["val_start"]).total_seconds() / (st["predicted"] - 1)
    if st["predicted"]:
        return max(n_val - st["predicted"], 0) * val_rate
    if not st["times"]:
        return None
    sp = statistics.median(st["times"][-20:])
    return max(st["budget"] - completed(st), 0) * sp + n_val * val_rate


def completed(st):
    """Époques terminées : l'époque courante compte si son 'Epoch time' est déjà écrit."""
    return max(st["epoch"], 0) + (1 if st["cur_done"] else 0)


def fmt_dur(s):
    if s is None:
        return "?"
    s = int(s)
    if s < 3600:
        return f"{s // 60}min"
    return f"{s // 86400}j{s % 86400 // 3600:02d}h" if s >= 86400 else f"{s // 3600}h{s % 3600 // 60:02d}"


def fmt_eta(now, s):
    return "?" if s is None else (now + timedelta(seconds=s)).strftime("%a %d %H:%M")


def queue_of(model_log):
    """(rang courant, taille de la file, uid courant) d'après le log de l'orchestrateur."""
    last = None
    if os.path.isfile(model_log):
        with open(model_log, errors="ignore") as f:
            for line in f:
                if (m := QUEUE.search(line)):
                    last = (int(m.group(1)), int(m.group(2)), m.group(3))
    return last


def alive(pid_file):
    try:
        os.kill(int(open(pid_file).read().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="*_s[12]", help="glob sur le nom du trainer")
    ap.add_argument("--datasets", default="Dataset100_*",
                    help="glob sur les datasets, ex. 'Dataset10[34]_*' pour M3/M4")
    ap.add_argument("--seeds-dir", default="results_seeds")
    ap.add_argument("--budget", type=int, default=500, help="époques si absent du log")
    args = ap.parse_args()

    res = os.environ.get("nnUNet_results", "nnUNet_data/nnUNet_results")
    ds = sorted(glob.glob(os.path.join(res, args.datasets)))
    if not ds:
        raise SystemExit(f"{args.datasets} introuvable dans {res} (source .env_nnunet ?)")
    now = datetime.now()
    try:
        print(subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    except FileNotFoundError:
        pass
    print(f"{now:%a %d %H:%M}  ({res})\n")
    hdr = f"{'trainer':<38}{'pli':>3}  {'état':<11}{'époque':>9}{'%':>6}{'s/ép':>7}" \
          f"{'dice':>7}{'EMA':>7}{'vu il y a':>10}  {'reste':>6}  fin du pli"
    print(hdr + "\n" + "-" * len(hdr))

    per_trainer = {}   # clé (dataset, trainer) : M3 et M4 ont le même trainer
    tdirs = [t for d in ds for t in sorted(glob.glob(os.path.join(d, f"{args.pattern}__*")))]
    for tdir in tdirs:
        trainer = os.path.basename(tdir).split("__")[0]
        dsid = int(re.search(r"Dataset(\d+)_", tdir).group(1))
        label = trainer if len(ds) == 1 else f"d{dsid} {trainer}"
        for fd in sorted(glob.glob(os.path.join(tdir, "fold_*"))):
            fold = int(fd.rsplit("_", 1)[1])
            st = parse_fold(fd, args.budget)
            done_ep = completed(st)
            if st["val_done"]:
                state = "terminé"
            elif st["predicted"]:
                state = f"valid {st['predicted']}/{st['n_val'] or '?'}"
            elif st["epoch"] >= 0:
                state = "entraîne"
            else:
                state = "démarrage"
            rem = remaining_s(st)
            age = (now - st["last"]).total_seconds() if st["last"] else None
            sp = statistics.median(st["times"][-20:]) if st["times"] else None
            if age is not None and not st["val_done"] and age > max(900, 5 * (sp or 0)):
                state = "BLOQUÉ?"
            per_trainer.setdefault((dsid, trainer), []).append((fold, st, rem, sp))
            print(f"{label:<38}{fold:>3}  {state:<11}{done_ep:>5}/{st['budget']:<3}"
                  f"{100 * min(done_ep / st['budget'], 1):>6.1f}{sp or 0:>7.1f}"
                  f"{st['dice'] or '-':>7}{st['ema'] if st['ema'] is not None else 0:>7.3f}"
                  f"{fmt_dur(age):>10}  {fmt_dur(rem):>6}  "
                  f"{'-' if st['val_done'] else fmt_eta(now, rem)}")

    # File de chaque processus : plis pas encore commencés (absents du disque) inclus.
    print(f"\n{'modèle (processus)':<24}{'actif':<7}{'file':>8}  {'reste':>7}  fin de la file")
    for log in sorted(glob.glob(os.path.join(args.seeds_dir, "logs", "*.log"))):
        model = os.path.basename(log)[:-4]
        q = queue_of(log)
        pidf = log[:-4] + ".pid"
        up = ("oui" if alive(pidf) else "NON") if os.path.isfile(pidf) else "?"
        if not q:
            print(f"{model:<24}{up:<7}{'?':>8}")
            continue
        rank, total, uid = q
        dsid, trainer, fold = re.match(r"d(\d+)_(.+)_f(\d+)$", uid).groups()
        folds = per_trainer.get((int(dsid), trainer), [])
        cur = [f for f in folds if f[0] == int(fold)]
        rem_cur = cur[0][2] if cur else None
        sp = next((f[3] for f in reversed(folds) if f[3]), None)
        per_fold = sp * (folds[0][1]["budget"] if folds else args.budget) + 17 * VAL_S_PER_CASE if sp else None
        rem = None if rem_cur is None or per_fold is None else rem_cur + (total - rank) * per_fold
        print(f"{model:<24}{up:<7}{rank:>4}/{total:<3}  {fmt_dur(rem):>7}  {fmt_eta(now, rem)}")


if __name__ == "__main__":
    main()
