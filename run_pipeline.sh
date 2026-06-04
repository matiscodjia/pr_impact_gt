#!/usr/bin/env bash
# ============================================================================
# run_pipeline.sh — Pipeline nnU-Net PARSE (préparation + orchestration)
#
# La PRÉPARATION des données (conversion, prétraitement, génération des labels
# dégradés, dataset fixe) est faite ici, séquentiellement. La boucle lourde
# (entraînement K-fold → collecte → rapport) est déléguée à orchestrator.py, qui
# est REPRENABLE : sur GPU partagé / coupures, il repart du dernier checkpoint
# sans perte et rafraîchit les résultats après chaque fold (anytime).
#
# Usage:
#   bash run_pipeline.sh /chemin/vers/PARSE              # complet, GPU
#   bash run_pipeline.sh /chemin/vers/PARSE --debug      # 2 époques (validation mécanique)
#   bash run_pipeline.sh /chemin/vers/PARSE --from orchestrate   # reprendre l'entraînement
#   bash run_pipeline.sh /chemin/vers/PARSE --tiers ABC  # inclure grid/spécificité (P2/P3)
#
# Conseil tmux pour sessions longues :
#   tmux new -s nnunet ; bash run_pipeline.sh /chemin/vers/PARSE
#   # Ctrl+B D pour détacher, tmux attach -t nnunet pour revenir
# ============================================================================

set -euo pipefail

check_dependencies() {
    local deps=("uv" "python3" "nnUNetv2_train")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            echo "ERREUR: Dépendance absente : $dep"; exit 1
        fi
    done
}
check_dependencies

DATA_DIR="${1:?Usage: $0 /chemin/vers/PARSE [--debug] [--from STEP] [--tiers A|ABC]}"
shift

# ── Arguments optionnels ──
DEBUG=false
FROM_STEP="convert"
DATASET_ID=100
FIXED_ID=101
FIXED_DIR_NAME="Dataset101_PARSE_Fixed"
TIERS="A"
CONFIG="configs/experiment_config.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)   DEBUG=true; shift ;;
        --from)    FROM_STEP="$2"; shift 2 ;;
        --tiers)   TIERS="$2"; shift 2 ;;
        --config)  CONFIG="$2"; shift 2 ;;
        *)         echo "Argument inconnu: $1"; exit 1 ;;
    esac
done

# ── Détection device (informatif ; orchestrator.py détecte aussi) ──
detect_device() {
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        local name vram
        name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        echo "cuda (${name}, $((vram / 1024)) Go)"
    elif python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        echo "mps (Apple Silicon)"
    else
        echo "cpu"
    fi
}
DEVICE_INFO=$(detect_device)

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  nnU-Net PARSE — préparation + orchestration              ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Data:   ${DATA_DIR}"
echo "║  Device: ${DEVICE_INFO}"
echo "║  Debug:  ${DEBUG}   Tiers: ${TIERS}   From: ${FROM_STEP}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if [[ -z "${nnUNet_raw:-}" ]]; then
    echo "ERREUR: variables nnU-Net non définies. Lancez d'abord : source .env_nnunet"
    exit 1
fi

# ── Ordre des étapes ──
STEPS=(convert preprocess generate_degraded create_fixed_dataset preprocess_fixed orchestrate)

START_IDX=0
for i in "${!STEPS[@]}"; do
    [[ "${STEPS[$i]}" == "${FROM_STEP}" ]] && { START_IDX=$i; break; }
done

step_header() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ▶ $1    $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}
should_run() {
    for i in "${!STEPS[@]}"; do
        [[ "${STEPS[$i]}" == "$1" && $i -ge $START_IDX ]] && return 0
    done
    return 1
}

# ── 1. Conversion PARSE → nnU-Net ──
if should_run "convert"; then
    step_header "1/6 — Conversion PARSE → format nnU-Net"
    python scripts/convert_parse_to_nnunet.py --data_dir "${DATA_DIR}" --dataset_id ${DATASET_ID}
fi

# ── 2. Prétraitement Dataset100 ──
if should_run "preprocess"; then
    step_header "2/6 — Prétraitement nnU-Net (Dataset${DATASET_ID})"
    nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity
fi

# ── 3. Labels dégradés (GT_minus) — train ET test, pour l'éval out-of-fold ──
if should_run "generate_degraded"; then
    step_header "3/6 — Génération des labels dégradés (labelsTr/labelsTs)"
    python scripts/generate_degraded_dataset.py \
        --dataset_dir "${nnUNet_raw}/Dataset${DATASET_ID}_PARSE" \
        --config "${CONFIG}" --seed 42
fi

# ── 4. Dataset fixe (labels train dégradés déterministes) pour Model_Minus_Fixed ──
if should_run "create_fixed_dataset"; then
    step_header "4/6 — Création ${FIXED_DIR_NAME}"
    python scripts/create_fixed_degraded_dataset.py \
        --dataset_id ${FIXED_ID} \
        --source_dir "${nnUNet_raw}/Dataset${DATASET_ID}_PARSE" \
        --output_dir "${nnUNet_raw}/${FIXED_DIR_NAME}" \
        --config "${CONFIG}" --seed 42
fi

# ── 5. Prétraitement Dataset101 (les splits seront alignés par l'orchestrateur) ──
if should_run "preprocess_fixed"; then
    step_header "5/6 — Prétraitement nnU-Net (Dataset${FIXED_ID})"
    nnUNetv2_plan_and_preprocess -d ${FIXED_ID} --verify_dataset_integrity
fi

# ── 6. Orchestration : entraînement reprenable → collecte → rapport ──
if should_run "orchestrate"; then
    step_header "6/6 — Orchestration (entraînement K-fold reprenable + anytime)"
    DEBUG_FLAG=""
    [[ "${DEBUG}" == "true" ]] && DEBUG_FLAG="--debug"
    python scripts/orchestrator.py --config "${CONFIG}" --tiers "${TIERS}" ${DEBUG_FLAG}
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Pipeline terminé !                                       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Tableau de bord : results/STATUS.md                      ║"
echo "║  Table maîtresse : results/metrics.csv                    ║"
echo "║  Stats           : results/outcomes_oof.csv              ║"
echo "║  Figures         : results/figures/                       ║"
echo "║  Reprise         : relancer la même commande (idempotent) ║"
echo "╚══════════════════════════════════════════════════════════╝"
