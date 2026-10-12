#!/usr/bin/env bash
#
# sync_and_report.sh — rapatrie les prédictions du cluster puis (re)calcule tout en local.
#
# Chaîne : rsync (cluster → PC)  →  collect_metrics.py  →  report.py
#
#   1. rsync : ne récupère QUE les dossiers `fold_*/validation/` (les prédictions OOF)
#              vers ./nnUNet_data/nnUNet_results/  — chemin IGNORÉ par git (aucun conflit
#              possible avec ton cycle commit/pull ; cf. .gitignore).
#   2. collect_metrics : calcul CPU-lourd des métriques (Betti0, clDice, HD95, NSD) sur ton
#              PC puissant → table maîtresse results/metrics.csv (upsert idempotent).
#   3. report : stats appariées (Wilcoxon+FDR), outcomes (IC bootstrap), figures, STATUS.md.
#
# Ce qui est produit dans results/ est PETIT et diffable → à committer via git, sans LFS.
# Ce qui transite dans nnUNet_data/ est GROS → rsync uniquement, jamais git.
#
# ─── Usage ────────────────────────────────────────────────────────────────────
#   REMOTE=user@cluster REMOTE_DIR=pr_impact_gt ./scripts/sync_and_report.sh
#
#   Options :
#     --dry-run      montre ce que rsync copierait, sans rien écrire
#     --no-sync      saute le rsync (recalcule seulement à partir des volumes déjà là)
#     --no-report    s'arrête après collect_metrics (pas de figures/stats)
#     --kind KIND    oof (défaut) | test  — jeu évalué par report.py
#     --with-test    rsync aussi predictions/ (ensemble sur le test set), si présent
#
#   Variables d'environnement :
#     REMOTE       (obligatoire sauf --no-sync)  ex: mcodjia@cluster.univ.fr
#     REMOTE_DIR   (obligatoire sauf --no-sync)  ex: pr_impact_gt (relatif au home distant ; PAS de ~ qui s'expanserait localement)
#     SSH_OPTS     options ssh (rebonds/bastion). ex: "-J bastion" ou "-J b1,b2"
#                  (inutile si tu déclares le ProxyJump dans ~/.ssh/config)
#     PYTHON       interpréteur (défaut: .venv/bin/python s'il existe, sinon python3)
#     CONFIG       config expé (défaut: configs/experiment_config.yaml)
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Racine du repo (le script vit dans scripts/) ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Réglages ──────────────────────────────────────────────────────────────────
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

# nnU-Net lit ses volumes ici (défauts locaux) — on les fixe pour être explicite.
export nnUNet_results="${nnUNet_results:-$REPO_ROOT/nnUNet_data/nnUNet_results}"
export nnUNet_raw="${nnUNet_raw:-$REPO_ROOT/nnUNet_data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$REPO_ROOT/nnUNet_data/nnUNet_preprocessed}"

# Interpréteur Python : venv local si dispo, sinon python3.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then PYTHON="$REPO_ROOT/.venv/bin/python"; else PYTHON="python3"; fi
fi

# ── Parsing des options ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)   DRY_RUN="--dry-run"; shift ;;
    --no-sync)   DO_SYNC=0; shift ;;
    --no-report) DO_REPORT=0; shift ;;
    --with-test) WITH_TEST=1; shift ;;
    --kind)      KIND="${2:?--kind requiert une valeur}"; shift 2 ;;
    -h|--help)   sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "Option inconnue : $1" >&2; exit 2 ;;
  esac
done

log() { printf '\n\033[1;36m▸ %s\033[0m\n' "$*"; }

# ── 1. rsync : cluster → local (dossiers validation uniquement) ───────────────
if [[ "$DO_SYNC" -eq 1 ]]; then
  if [[ -z "$REMOTE" || -z "$REMOTE_DIR" ]]; then
    echo "ERREUR : REMOTE et REMOTE_DIR doivent être définis pour le rsync." >&2
    echo "  ex: REMOTE=user@cluster REMOTE_DIR=pr_impact_gt $0" >&2
    echo "  (ou lance avec --no-sync pour recalculer sans rapatrier)" >&2
    exit 2
  fi
  command -v rsync >/dev/null || { echo "ERREUR : rsync introuvable." >&2; exit 3; }

  # Transport SSH : ajoute les rebonds/bastion (SSH_OPTS) si fournis.
  RSH_ARGS=()
  [[ -n "$SSH_OPTS" ]] && RSH_ARGS=(-e "ssh $SSH_OPTS")

  DEST="$nnUNet_results/"
  mkdir -p "$DEST"

  # Garde-fou : la destination DOIT être ignorée par git (étanchéité du versionning).
  if git -C "$REPO_ROOT" check-ignore -q nnUNet_data; then
    :
  else
    echo "AVERTISSEMENT : nnUNet_data/ n'est pas gitignore — risque de mélange git/rsync." >&2
    echo "  Ajoute 'nnUNet_data' à .gitignore avant de continuer." >&2
    exit 4
  fi

  log "rsync des prédictions validation depuis $REMOTE:$REMOTE_DIR"
  # --prune-empty-dirs + include/exclude : ne descend QUE dans les fold_*/validation/.
  rsync -avz --partial --progress $DRY_RUN --prune-empty-dirs "${RSH_ARGS[@]+${RSH_ARGS[@]}}" \
    --include='*/' \
    --include='validation/**' \
    --exclude='*' \
    "$REMOTE:$REMOTE_DIR/nnUNet_data/nnUNet_results/" "$DEST"

  if [[ "$WITH_TEST" -eq 1 ]]; then
    log "rsync des prédictions test set (predictions/)"
    rsync -avz --partial --progress $DRY_RUN "${RSH_ARGS[@]+${RSH_ARGS[@]}}" \
      "$REMOTE:$REMOTE_DIR/predictions/" "$REPO_ROOT/predictions/" || \
      echo "  (pas de predictions/ côté cluster — ignoré)"
  fi

  [[ -n "$DRY_RUN" ]] && { log "--dry-run : arrêt avant calcul."; exit 0; }
fi

# ── 2. collect_metrics : volumes → table maîtresse (calcul lourd, local) ──────
log "collect_metrics → $RESULTS_DIR/metrics.csv"
"$PYTHON" scripts/collect_metrics.py --config "$CONFIG" --output "$RESULTS_DIR"

# ── 3. report : stats, outcomes, figures, STATUS.md ───────────────────────────
if [[ "$DO_REPORT" -eq 1 ]]; then
  log "report ($KIND) → stats, figures, STATUS.md"
  "$PYTHON" scripts/report.py --results_dir "$RESULTS_DIR" --kind "$KIND"
fi

# ── Récap : ce qui est prêt à committer (petit, sans LFS) ─────────────────────
log "Terminé. Artefacts à committer (git, sans LFS) :"
for f in \
  "$RESULTS_DIR/metrics.csv" \
  "$RESULTS_DIR/metrics.parquet" \
  "$RESULTS_DIR/metrics_stats_${KIND}.csv" \
  "$RESULTS_DIR/outcomes_${KIND}.csv" \
  "$RESULTS_DIR/STATUS.md"; do
  [[ -e "$f" ]] && printf '  %-40s %s\n' "$f" "$(du -h "$f" | cut -f1)"
done
echo
echo "  Figures : $RESULTS_DIR/figures/  (les volumes rsyncés dans nnUNet_data/ restent hors git)"
