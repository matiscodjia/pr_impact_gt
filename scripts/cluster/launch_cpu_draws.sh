#!/usr/bin/env bash
# ============================================================================
# launch_cpu_draws.sh -- K tirages de GT- (omission p=0.3) : variance de la RÉALISATION du bruit.
# CPU seulement (aucun GPU, aucune ré-inférence : lit les prédictions OOF déjà sur le worker).
#
# À exécuter SUR le worker, racine du repo, après `git pull` :
#     bash scripts/cluster/launch_cpu_draws.sh --draws 10 --detach
#     bash scripts/cluster/launch_cpu_draws.sh --draws 10 --workers 24
# Reprenable : un tirage dont results/dose_response_draw<graine>.csv existe est sauté.
# Graines : 1042, 2042, ... (pas de 1000 > nb de cas : aucun recouvrement avec la graine 42 de l'étude).
# Sorties : results/dose_response_draw<graine>*.csv puis results/reference_draws_summary.csv
#
# Mémoire : ~2 Go par worker observés en local. Par défaut : min(coeurs, RAM_dispo / 3 Go).
# (Un pool trop large se BLOQUE silencieusement quand un worker est tué par manque de RAM.)
# ============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DRAWS=10; WORKERS=0; DETACH=0; P="0.3"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --draws) DRAWS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --ps) P="$2"; shift 2 ;;
    --detach) DETACH=1; shift ;;
    *) echo "argument inconnu: $1"; exit 1 ;;
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
echo "tirages: $DRAWS | p=$P | workers: $WORKERS"

if [[ "$DETACH" == 1 ]]; then
  mkdir -p results/logs
  setsid nohup bash "$0" --draws "$DRAWS" --workers "$WORKERS" --ps "$P" > results/logs/draws.log 2>&1 < /dev/null &
  echo "détaché (pid $!). Suivi : tail -f results/logs/draws.log"; exit 0
fi

for k in $(seq 1 "$DRAWS"); do
  seed=$((42 + 1000 * k))
  if [[ -f "results/dose_response_draw${seed}.csv" ]]; then echo "[draw $seed] déjà fait"; continue; fi
  echo "[draw $seed] $(date +%H:%M:%S)"
  $PY analysis/dose_response.py --ps "$P" --seed "$seed" --tag "_draw${seed}" --workers "$WORKERS"
done
$PY analysis/aggregate_draws.py
