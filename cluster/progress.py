#!/usr/bin/env python3
"""Detailed progress of the trainings on the GPU node (what launch_gpu.sh --status cannot show).

Reads the nnU-Net training logs (training_log_*.txt, the source of truth, also after a
resume with --c) and the orchestrator ledgers. Per fold: epoch reached / budget, seconds
per epoch (median of the last 20), pseudo-Dice, EMA, age of the last log line, estimated
end; per model process: estimated end of its whole queue.

Run ON the node, from the repository root:
    python cluster/progress.py                  # seed replicates *_s1/_s2 + Q3 (M3/M4)
    python cluster/progress.py --pattern '*'    # every trainer
    watch -n 60 python cluster/progress.py      # refreshed every minute
    python cluster/progress.py --scan 'Dataset10[34]_*:nnUNetTrainerStd'   # Q3 only
"""
import argparse
import glob
import json
import os
import re
import socket
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
QUEUE = re.compile(r"\[(\d+)/(\d+)\] (?:▶|>) (\S+)")   # orchestrator line (old logs used ▶)
# Campaigns shown by default: seed replicates (Dataset100, *_s1/_s2) and Q3 (M3/M4 =
# nnUNetTrainerStd on Dataset103/104). The original M0-M2, finished, are not listed.
DEFAULT_SCANS = ["Dataset100_*:*_s[12]", "Dataset10[34]_*:nnUNetTrainerStd"]
VAL_S_PER_CASE = 70.0   # measured on the original M0-M2 folds (~17 cases in ~20 min)


def parse_log(log, default_budget):
    st = {"budget": default_budget, "n_val": None, "epoch": -1, "times": [], "dice": None,
          "ema": None, "predicted": 0, "val_done": False, "last": None, "val_start": None,
          "cur_done": False}
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


def parse_fold(fold_dir, default_budget):
    """One log per nnU-Net launch (resume with --c, --debug, or a duplicate process).

    The fold is finished if ONE log reached the budget and validated -- a --debug run
    validated after 2 epochs does not count, and a duplicate killed midway does not undo
    it. Otherwise the most recent log is shown (epoch numbers are absolute, so they stay
    right after --c).
    """
    logs = sorted(glob.glob(os.path.join(fold_dir, "training_log_*.txt")), key=os.path.getmtime)
    states = [parse_log(log, default_budget) for log in logs] or [parse_log(os.devnull, default_budget)]
    finished = [s for s in states if s["val_done"] and completed(s) >= default_budget]
    st = dict(finished[-1] if finished else states[-1])
    st["log_last"] = [s["last"] for s in states]
    return st


def remaining_s(st, n_val_default=17):
    """Seconds left to finish this fold (rest of the training + validation)."""
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
    """Finished epochs: the current epoch counts once its 'Epoch time' line is written."""
    return max(st["epoch"], 0) + (1 if st["cur_done"] else 0)


def fmt_dur(s):
    if s is None:
        return "?"
    s = int(s)
    if s < 3600:
        return f"{s // 60}min"
    return f"{s // 86400}d{s % 86400 // 3600:02d}h" if s >= 86400 else f"{s // 3600}h{s % 3600 // 60:02d}"


def fmt_eta(now, s):
    return "?" if s is None else (now + timedelta(seconds=s)).strftime("%a %d %H:%M")


def queue_of(model_log):
    """(current rank, queue size, current unit) from the orchestrator log."""
    last = None
    if os.path.isfile(model_log):
        with open(model_log, errors="ignore") as f:
            for line in f:
                if (m := QUEUE.search(line)):
                    last = (int(m.group(1)), int(m.group(2)), m.group(3))
    return last


def alive(pid_file):
    """'yes' / 'NO' / node name. The .pid file holds "pid@node" (old format "pid" = this node)."""
    try:
        pid, _, host = open(pid_file).read().strip().partition("@")
        if host and host != socket.gethostname():
            return host.split(".")[0]          # another node: the PID means nothing here
        os.kill(int(pid), 0)
        return "yes"
    except (OSError, ValueError):
        return "NO"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="append", metavar="DATASETS:TRAINERS",
                    help="dataset:trainer glob pair, repeatable (default: replicates + Q3)")
    ap.add_argument("--pattern", default=None, help="glob on the trainer name (single pair)")
    ap.add_argument("--datasets", default=None,
                    help="glob on the datasets, e.g. 'Dataset10[34]_*' for M3/M4")
    ap.add_argument("--seeds-dir", default="results_seeds")
    ap.add_argument("--budget", type=int, default=500, help="epochs if the log does not say")
    args = ap.parse_args()

    res = os.environ.get("nnUNet_results", "nnUNet_data/nnUNet_results")
    if args.datasets or args.pattern:
        scans = [(args.datasets or "Dataset100_*", args.pattern or "*_s[12]")]
    else:
        scans = [tuple(x.split(":", 1)) for x in (args.scan or DEFAULT_SCANS)]
    if not glob.glob(os.path.join(res, "Dataset*")):
        raise SystemExit(f"no dataset in {res} (source .env_nnunet?)")
    now = datetime.now()
    try:
        print(subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    except FileNotFoundError:
        pass
    print(f"{now:%a %d %H:%M}  ({res})\n")
    hdr = f"{'trainer':<38}{'fold':>4}  {'state':<13}{'epoch':>9}{'%':>6}{'s/ep':>7}" \
          f"{'dice':>7}{'EMA':>7}{'last seen':>10}  {'left':>6}  fold ends"
    print(hdr + "\n" + "-" * len(hdr))

    rows, per_trainer = [], {}   # key (dataset, trainer): M3 and M4 share the same trainer
    tdirs = [t for dglob, tglob in scans for d in sorted(glob.glob(os.path.join(res, dglob)))
             for t in sorted(glob.glob(os.path.join(d, f"{tglob}__*")))]
    for tdir in tdirs:
        trainer = os.path.basename(tdir).split("__")[0]
        dsid = int(re.search(r"Dataset(\d+)_", tdir).group(1))
        label = trainer if dsid == 100 else f"d{dsid} {trainer}"
        for fd in sorted(glob.glob(os.path.join(tdir, "fold_*"))):
            fold = int(fd.rsplit("_", 1)[1])
            st = parse_fold(fd, args.budget)
            sp = statistics.median(st["times"][-20:]) if st["times"] else None
            stale = max(900, 5 * (sp or 0))
            age = (now - st["last"]).total_seconds() if st["last"] else None
            fresh_logs = sum(1 for t in st["log_last"] if t and (now - t).total_seconds() < stale)
            if st["val_done"]:
                state = "done"
            elif age is not None and age > stale:
                state = "STALLED?"
            elif st["predicted"]:
                state = f"valid {st['predicted']}/{st['n_val'] or '?'}"
            elif st["epoch"] >= 0:
                state = "training"
            else:
                state = "starting"
            if fresh_logs > 1:
                state = "2 PROCESSES?"   # two logs of the same fold written recently
            rem = remaining_s(st)
            active = state not in ("done", "STALLED?")
            per_trainer.setdefault((dsid, trainer), []).append(
                {"fold": fold, "st": st, "rem": rem, "sp": sp, "active": active})
            rows.append([label, fold, state, st, sp, age, rem, (dsid, trainer)])

    for r in rows:   # one process trains one fold at a time
        if r[2] in ("training", "starting") or r[2].startswith("valid"):
            if sum(f["active"] for f in per_trainer[r[7]]) > 1:
                r[2] = "DUPLICATE?"
    for label, fold, state, st, sp, age, rem, _ in rows:
        done_ep = completed(st)
        print(f"{label:<38}{fold:>4}  {state:<13}{done_ep:>5}/{st['budget']:<3}"
              f"{100 * min(done_ep / st['budget'], 1):>6.1f}{sp or 0:>7.1f}"
              f"{st['dice'] or '-':>7}{st['ema'] if st['ema'] is not None else 0:>7.3f}"
              f"{fmt_dur(age):>10}  {fmt_dur(rem):>6}  "
              f"{'-' if state == 'done' else fmt_eta(now, rem)}")
    if any(r[2] in ("DUPLICATE?", "2 PROCESSES?") for r in rows):
        print("\n!! Several active folds for one model: one process too many? Check:\n"
              "   ps -eo pid,pgid,etime,args | grep -E '[n]nUNetv2_train|[o]rchestrator.py'\n"
              "   nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")

    # Per process: the ledger (written at the start of every fold) rather than the
    # orchestrator log, which Python does not flush while it writes to a file.
    print(f"\n{'model (process)':<24}{'alive':<14}{'folds done':>11}  {'running':<22}estimated end")
    for log in sorted(glob.glob(os.path.join(args.seeds_dir, "logs", "*.log"))):
        model = os.path.basename(log)[:-4]
        pidf = log[:-4] + ".pid"
        up = alive(pidf) if os.path.isfile(pidf) else "?"
        try:
            units = json.load(open(os.path.join(args.seeds_dir, model, "ledger.json")))["units"]
        except (OSError, ValueError, KeyError):
            units = {}
        done = [u for u in units.values() if u.get("state") == "done"]
        running = [f"f{u['fold']}" for u in units.values() if u.get("state") == "running"]
        keys = {(int(u["dataset"]), u["trainer"]) for u in units.values() if "trainer" in u}
        rems = [f["rem"] for k in keys for f in per_trainer.get(k, []) if f["active"]]
        eta = fmt_eta(now, max(rems)) if rems and None not in rems else "?"
        q = queue_of(log)   # if the log was flushed after all: queue size
        todo = f" (queue {q[0]}/{q[1]})" if q else ""
        print(f"{model:<24}{up:<14}{len(done):>11}  {(','.join(running) or '-') + todo:<22}{eta}")


if __name__ == "__main__":
    main()
