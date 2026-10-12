#!/usr/bin/env bash
#
# sync_and_report.sh -- pull the predictions from the GPU node, then score everything locally.
#
# Chain: rsync (node -> local)  ->  pipeline/collect_metrics.py  ->  pipeline/report.py
#
#   1. rsync: copies ONLY the fold_*/validation/ folders (the out-of-fold predictions) to
#             ./nnUNet_data/nnUNet_results/, a path IGNORED by git.
#   2. collect_metrics: CPU-heavy scoring (Betti-0, clDice, HD95, NSD) into the master table
#             results/metrics.csv (idempotent upsert).
#   3. report: paired statistics (Wilcoxon + FDR), outcomes (bootstrap CI), figures, STATUS.md.
#
# What lands in results/ is SMALL and diffable (commit it); what goes to nnUNet_data/ is
# LARGE (rsync only, never git).
#
# Usage
#   REMOTE=user@node REMOTE_DIR=pr_impact_gt bash cluster/sync_and_report.sh
#
# Options
#   --dry-run      show what rsync would copy, write nothing
#   --no-sync      skip rsync (rescore from the volumes already here)
#   --no-report    stop after collect_metrics (no figures or statistics)
#   --kind KIND    oof (default) | test -- set evaluated by report.py
#   --with-test    also rsync predictions/ (test-set ensemble), if present
#
# Environment
#   REMOTE       (required unless --no-sync)  e.g. mcodjia@cluster.univ.fr
#   REMOTE_DIR   (required unless --no-sync)  e.g. pr_impact_gt (relative to the remote home;
#                no ~, which would expand locally)
#   SSH_OPTS     ssh options (jump hosts), e.g. "-J bastion" (not needed with a ProxyJump
#                in ~/.ssh/config)
#   PYTHON       interpreter (default: .venv/bin/python if present, else python3)
#   CONFIG       configuration (default: configs/experiment_config.yaml)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

REMOTE="${REMOTE:-}"
REMOTE_DIR="${REMOTE_DIR:-}"
SSH_OPTS="${SSH_OPTS:-}"
CONFIG="${CONFIG:-configs/experiment_config.yaml}"
RESULTS_DIR="results"
KIND="oof"
DO_SYNC=1
DO_REPORT=1
WITH_TEST=0
DRY_RUN=""

# nnU-Net volumes (local defaults), set explicitly.
export nnUNet_results="${nnUNet_results:-$REPO_ROOT/nnUNet_data/nnUNet_results}"
export nnUNet_raw="${nnUNet_raw:-$REPO_ROOT/nnUNet_data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$REPO_ROOT/nnUNet_data/nnUNet_preprocessed}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then PYTHON="$REPO_ROOT/.venv/bin/python"; else PYTHON="python3"; fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)   DRY_RUN="--dry-run"; shift ;;
    --no-sync)   DO_SYNC=0; shift ;;
    --no-report) DO_REPORT=0; shift ;;
    --with-test) WITH_TEST=1; shift ;;
    --kind)      KIND="${2:?--kind needs a value}"; shift 2 ;;
    -h|--help)   sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

log() { printf '\n\033[1;36m> %s\033[0m\n' "$*"; }

# 1. rsync: node -> local (validation folders only)
if [[ "$DO_SYNC" -eq 1 ]]; then
  if [[ -z "$REMOTE" || -z "$REMOTE_DIR" ]]; then
    echo "ERROR: REMOTE and REMOTE_DIR must be set for rsync." >&2
    echo "  e.g. REMOTE=user@node REMOTE_DIR=pr_impact_gt $0" >&2
    echo "  (or use --no-sync to rescore without pulling)" >&2
    exit 2
  fi
  command -v rsync >/dev/null || { echo "ERROR: rsync not found." >&2; exit 3; }

  RSH_ARGS=()
  [[ -n "$SSH_OPTS" ]] && RSH_ARGS=(-e "ssh $SSH_OPTS")

  DEST="$nnUNet_results/"
  mkdir -p "$DEST"

  # Safeguard: the destination MUST be ignored by git.
  if ! git -C "$REPO_ROOT" check-ignore -q nnUNet_data; then
    echo "WARNING: nnUNet_data/ is not ignored by git: risk of mixing git and rsync." >&2
    echo "  Add 'nnUNet_data' to .gitignore first." >&2
    exit 4
  fi

  log "rsync of the validation predictions from $REMOTE:$REMOTE_DIR"
  # --prune-empty-dirs + include/exclude: only descends into fold_*/validation/.
  rsync -avz --partial --progress $DRY_RUN --prune-empty-dirs "${RSH_ARGS[@]+${RSH_ARGS[@]}}" \
    --include='*/' \
    --include='validation/**' \
    --exclude='*' \
    "$REMOTE:$REMOTE_DIR/nnUNet_data/nnUNet_results/" "$DEST"

  if [[ "$WITH_TEST" -eq 1 ]]; then
    log "rsync of the test-set predictions (predictions/)"
    rsync -avz --partial --progress $DRY_RUN "${RSH_ARGS[@]+${RSH_ARGS[@]}}" \
      "$REMOTE:$REMOTE_DIR/predictions/" "$REPO_ROOT/predictions/" || \
      echo "  (no predictions/ on the node: skipped)"
  fi

  [[ -n "$DRY_RUN" ]] && { log "--dry-run: stopping before scoring."; exit 0; }
fi

# 2. collect_metrics: volumes -> master table (heavy, local)
log "collect_metrics -> $RESULTS_DIR/metrics.csv"
"$PYTHON" pipeline/collect_metrics.py --config "$CONFIG" --output "$RESULTS_DIR"

# 3. report: statistics, outcomes, figures, STATUS.md
if [[ "$DO_REPORT" -eq 1 ]]; then
  log "report ($KIND) -> statistics, figures, STATUS.md"
  "$PYTHON" pipeline/report.py --results_dir "$RESULTS_DIR" --kind "$KIND"
fi

log "Done. Files to commit:"
for f in \
  "$RESULTS_DIR/metrics.csv" \
  "$RESULTS_DIR/metrics_stats_${KIND}.csv" \
  "$RESULTS_DIR/outcomes_${KIND}.csv" \
  "$RESULTS_DIR/STATUS.md"; do
  [[ -e "$f" ]] && printf '  %-40s %s\n' "$f" "$(du -h "$f" | cut -f1)"
done
echo
echo "  Figures: $RESULTS_DIR/figures/  (the volumes in nnUNet_data/ stay out of git)"
