#!/usr/bin/env bash
# ============================================================================
# launch_cpu_draws.sh -- K independent draws of GT- (omission p=0.3): variance of the noise
# REALISATION. CPU only (no GPU, no inference: reads the out-of-fold predictions already on
# the node).
#
# Run ON the node, from the repository root:
#     bash cluster/launch_cpu_draws.sh --draws 10 --detach
#     bash cluster/launch_cpu_draws.sh --draws 10 --workers 24
# Resumable: a draw whose results/dose_response_draw<seed>.csv exists is skipped.
# Seeds: 1042, 2042, ... (step of 1000 > number of cases: no overlap with the study's seed 42).
# Outputs: results/dose_response_draw<seed>*.csv, then results/reference_draws_summary.csv
#
# Memory: ~2 GB per worker observed. Default: min(cores, available RAM / 3 GB).
# (A pool that is too wide HANGS silently when a worker is killed for lack of memory.)
# ============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DRAWS=10; WORKERS=0; DETACH=0; P="0.3"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --draws) DRAWS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --ps) P="$2"; shift 2 ;;
    --detach) DETACH=1; shift ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done
[[ -f .env_nnunet ]] && source .env_nnunet
PY="${PYTHON:-python}"

if [[ "$WORKERS" -eq 0 ]]; then
  cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
  mem_gb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 12)
  by_mem=$(( mem_gb / 3 )); [[ $by_mem -lt 1 ]] && by_mem=1
  WORKERS=$(( cores < by_mem ? cores : by_mem ))
fi
echo "draws: $DRAWS | p=$P | workers: $WORKERS"

if [[ "$DETACH" == 1 ]]; then
  mkdir -p results/logs
  setsid nohup bash "$0" --draws "$DRAWS" --workers "$WORKERS" --ps "$P" > results/logs/draws.log 2>&1 < /dev/null &
  echo "detached (pid $!). Follow: tail -f results/logs/draws.log"; exit 0
fi

for k in $(seq 1 "$DRAWS"); do
  seed=$((42 + 1000 * k))
  if [[ -f "results/dose_response_draw${seed}.csv" ]]; then echo "[draw $seed] already done"; continue; fi
  echo "[draw $seed] $(date +%H:%M:%S)"
  $PY analysis/dose_response.py --ps "$P" --seed "$seed" --tag "_draw${seed}" --workers "$WORKERS"
done
$PY analysis/reference_draws.py
