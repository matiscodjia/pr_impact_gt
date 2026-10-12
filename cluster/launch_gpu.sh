#!/usr/bin/env bash
# ============================================================================
# launch_gpu.sh -- start the nnU-Net trainings (seed replicates, Q3, or tiers of the study)
# on the GPU node, as detached processes (they survive the ssh logout).
#
# Run ON the node, from the repository root:
#     bash cluster/launch_gpu.sh --debug                 # 2 epochs, checks the mechanics
#     bash cluster/launch_gpu.sh --tiers S               # seed 1, 3 regimes, 5 folds
#     bash cluster/launch_gpu.sh --tiers S --folds 0 1   # folds 0 and 1 only
#     bash cluster/launch_gpu.sh --tiers T               # seed 2
#     bash cluster/launch_gpu.sh --q3 --folds 0 1        # Q3: M3/M4 (after prepare_q3.sh)
#     bash cluster/launch_gpu.sh --tiers B --study       # M3/M4 with the full configuration
#                                                        #  (not advised: rescores M0-M2 after each fold)
#     bash cluster/launch_gpu.sh --status | --stop
#
# One process per model, each with ITS OWN results folder (separate ledger.json and
# metrics.csv): no write race between processes, and nothing here touches results/.
# Outputs: results_seeds/<model>/{ledger.json,metrics.csv,...} and results_seeds/logs/<model>.log
#
# Requires prgt installed in the nnU-Net environment (pip install -e . --no-deps).
# Variables: GPUS="0"   (GPU list, assigned round-robin to the processes)
#            DA_PROCS=6 (nnUNet_n_proc_DA per process; lower it on a node with few cores)
# ============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"

TIERS="S"; FOLDS=(); DEBUG=0; STUDY=0; Q3=0; ACTION="launch"; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tiers)  TIERS="$2"; shift 2 ;;
    --folds)  shift; FOLDS=(); while [[ $# -gt 0 && "$1" != --* ]]; do FOLDS+=("$1"); shift; done ;;
    --debug)  DEBUG=1; shift ;;
    --study)  STUDY=1; shift ;;      # use experiment_config.yaml (M0-M4) instead of the replicates
    --q3)     Q3=1; TIERS="B"; shift ;;   # M3/M4 only (configs/generated_q3_config.yaml)
    --status) ACTION="status"; shift ;;
    --stop)   ACTION="stop"; shift ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done

OUT="results_seeds"; mkdir -p "$OUT/logs"
[[ -f .env_nnunet ]] && source .env_nnunet          # nnUNet_raw/preprocessed/results + venv
PY="${PYTHON:-python}"

# ---- stop / status -----------------------------------------------------------
# An orchestrator = a python process RUNNING pipeline/orchestrator.py (or scripts/orchestrator.py,
# its location before the 2026-10 restructuring). The pattern is anchored, so a shell whose
# command line merely mentions the name, or a grep, is never taken for one.
ORCH_RE='^[^ ]*python[0-9.]* +([^ ]*/)?(pipeline|scripts)/orchestrator\.py '
running_orchestrators() {   # PIDs of every LOCAL orchestrator writing under $OUT/ (= its group, via setsid)
  pgrep -f -- "${ORCH_RE}.*--results_dir $OUT/${1:-}" || true
}
HOST="$(hostname)"
# .pid file = "pid@node": with a /home shared between nodes, results_seeds/logs/ is common
# and a PID only means something on the node that created it (old format "pid" = this node).
pid_of()  { cut -d@ -f1 < "$1"; }
host_of() { local h; h=$(cut -s -d@ -f2 < "$1"); echo "${h:-$HOST}"; }
is_orch() { ps -o args= -p "$1" 2>/dev/null | grep -Eq "$ORCH_RE"; }   # reused PID? stale .pid?
if [[ "$ACTION" == "stop" ]]; then
  # .pid files (of THIS node) AND a search by command line: a double launch overwrites the
  # .pid files, and the orchestrators of the first launch would survive a --stop based on
  # the .pid files alone.
  pids=$(running_orchestrators)
  for f in "$OUT"/logs/*.pid; do
    [[ -f "$f" ]] || continue
    if [[ "$(host_of "$f")" != "$HOST" ]]; then
      echo "$(basename "$f" .pid) runs on $(host_of "$f"): run --stop there"; continue
    fi
    pids="$pids $(pid_of "$f")"; rm -f "$f"
  done
  for pid in $(echo $pids | tr ' ' '\n' | sort -u); do
    is_orch "$pid" || { echo "$pid is not an orchestrator on $HOST: ignored"; continue; }
    { kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null; } && echo "stopped $pid" || true
  done
  exit 0
fi
if [[ "$ACTION" == "status" ]]; then
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  for f in "$OUT"/logs/*.log; do [[ -f "$f" ]] || continue
    echo "== $(basename "$f" .log)"; tail -n 2 "$f" | cut -c1-160
    pf="${f%.log}.pid"; [[ -f "$pf" ]] || continue
    if [[ "$(host_of "$pf")" != "$HOST" ]]; then echo "   [on $(host_of "$pf")]"
    else is_orch "$(pid_of "$pf")" && echo "   [running]" || echo "   [finished/stopped]"; fi
  done
  exit 0
fi

# ---- pre-flight -----------------------------------------------------------------
: "${nnUNet_raw:?nnUNet_raw not set - run: source .env_nnunet}"
[[ -d "$nnUNet_raw/Dataset100_PARSE" ]] || { echo "Dataset100_PARSE not found in $nnUNet_raw"; exit 1; }
[[ "$DRY" == 1 ]] || command -v nvidia-smi >/dev/null || { echo "no nvidia-smi: not a GPU node"; exit 1; }

# The trainers are COPIED into the nnU-Net package (nnU-Net finds them there by class name);
# copy again after every change of nnunet_trainers/. They import the noise operator from prgt.
if [[ "$DRY" == 0 ]]; then
  $PY -c "import prgt.noise" 2>/dev/null || {
    echo "prgt is not importable: run  pip install -e . --no-deps  in this environment"; exit 1; }
  TDIR="$($PY -c "import nnunetv2,os;print(os.path.join(nnunetv2.__path__[0],'training','nnUNetTrainer'))")"
  cp nnunet_trainers/nnUNetTrainerDegraded.py "$TDIR/"
  echo "trainers installed in $TDIR"
  $PY -c "from nnunetv2.training.nnUNetTrainer.nnUNetTrainerDegraded import nnUNetTrainerStd_s1, nnUNetTrainerDriftMu0_s2; print('replicate trainers import: OK')"
fi

if [[ "$Q3" == 1 ]]; then
  # The plans and data of 103-104 must come from prepare_q3.sh (Dataset100 plans, labels = GT-).
  for n in Dataset103_PARSE_DriftMuMinus Dataset104_PARSE_DriftMuPlus; do
    [[ -f "${nnUNet_preprocessed:-nnUNet_data/nnUNet_preprocessed}/$n/.q3_ready" ]] && continue
    [[ "$DRY" == 1 ]] && { echo "(dry-run) $n not prepared yet"; continue; }
    echo "$n not prepared: run first  bash cluster/prepare_q3.sh"; exit 1
  done
  $PY cluster/make_q3_config.py; CFG="configs/generated_q3_config.yaml"
elif [[ "$STUDY" == 1 ]]; then CFG="configs/experiment_config.yaml"
else $PY cluster/make_seed_config.py; CFG="configs/generated_seeds_config.yaml"; fi
[[ "$DEBUG" == 1 ]] && export DEBUG_PIPELINE=1
if [[ "$DEBUG" == 1 ]]; then
  cat <<'MSG'
!! --debug trains 2 epochs and WRITES checkpoint_final.pth + validation/ to
!! nnUNet_results/Dataset100_PARSE/<trainer>__nnUNetPlans__3d_fullres/fold_<k>/.
!! The orchestrator would then see a "done" unit and SKIP the real training.
!! After the test, delete those folders AND the test ledgers before the real launch:
!!   rm -rf nnUNet_data/nnUNet_results/Dataset100_PARSE/*_s1__nnUNetPlans__3d_fullres/fold_0 results_seeds
MSG
fi
export nnUNet_n_proc_DA="${DA_PROCS:-6}"

# models of the configuration that belong to the requested tiers
MODELS=(); while IFS= read -r line; do MODELS+=("$line"); done < <($PY cluster/list_models.py "$CFG" "$TIERS")
[[ ${#MODELS[@]} -gt 0 ]] || { echo "no model of tiers '$TIERS' in $CFG"; exit 1; }
IFS=' ' read -r -a GPU_LIST <<< "${GPUS:-0}"

i=0; launched=0
for m in "${MODELS[@]}"; do
  # Safeguard: launching again while it runs would train each fold TWICE in the same folder
  # (checkpoints and validation/ overwritten, GPU shared in two). It happened on 2026-09-22.
  if [[ -n "$(running_orchestrators "$m( |$)")" ]]; then
    echo "[$m] already running (pid $(running_orchestrators "$m( |$)" | tr '\n' ' ')) -- not relaunched"
    continue
  fi
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"; i=$((i+1))
  cmd=( "$PY" pipeline/orchestrator.py --config "$CFG" --results_dir "$OUT/$m" --tiers "$TIERS"
        --models "$m" --no-progress )
  # replicates: no .npz (softmax), ~2/3 of the size of a fold and unused by the analyses.
  # KEEP_NPZ=1 keeps them; the --study mode (M0-M4) is unchanged.
  [[ "$STUDY" == 0 && "${KEEP_NPZ:-0}" != 1 ]] && cmd+=( --no-npz )
  [[ ${#FOLDS[@]} -gt 0 ]] && cmd+=( --folds "${FOLDS[@]}" )
  [[ "$DEBUG" == 1 ]] && cmd+=( --debug )
  echo "[$m] GPU $gpu : ${cmd[*]}"
  [[ "$DRY" == 1 ]] && continue
  mkdir -p "$OUT/$m"
  # setsid: new process group => survives the ssh logout; --stop kills the whole group.
  # PYTHONUNBUFFERED: otherwise the model log (a file) stays empty for hours (Python buffer).
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 setsid nohup "${cmd[@]}" >> "$OUT/logs/$m.log" 2>&1 < /dev/null &
  echo "$!@$HOST" > "$OUT/logs/$m.pid"; launched=$((launched+1))
done
[[ "$DRY" == 1 ]] && { echo; echo "dry-run: nothing launched."; exit 0; }
echo; echo "launched $launched process(es). Follow: bash cluster/launch_gpu.sh --status"
