#!/usr/bin/env bash
# ============================================================================
# run_pipeline.sh — Pipeline complet nnU-Net PARSE
#
# Lancez ce script dans un terminal SSH sur votre machine GPU.
# Utilise uv pour la gestion de l'environnement Python.
#
# Usage:
#   # Sur macOS (debug/validation du pipeline)
#   bash run_pipeline.sh /chemin/vers/PARSE --debug
#
#   # Sur le serveur GPU A40 (expériences complètes)
#   bash run_pipeline.sh /chemin/vers/PARSE
#
#   # Reprendre à une étape spécifique
#   bash run_pipeline.sh /chemin/vers/PARSE --from train_star
#
# Conseil tmux pour sessions longues :
#   tmux new -s nnunet
#   bash run_pipeline.sh /chemin/vers/PARSE
#   # Ctrl+B D pour détacher, tmux attach -t nnunet pour revenir
# ============================================================================

set -euo pipefail

# ── Vérification des dépendances ──
check_dependencies() {
    local deps=("uv" "python3" "nnUNetv2_train")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            echo "ERREUR: Dépendance absente : $dep"
            exit 1
        fi
    done
}

check_dependencies

DATA_DIR="${1:?Usage: $0 /chemin/vers/PARSE [--debug] [--from STEP]}"
shift

# ── Parse des arguments optionnels ──
DEBUG=false
FROM_STEP="convert"
DATASET_ID=100

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) DEBUG=true; shift ;;
        --from)  FROM_STEP="$2"; shift 2 ;;
        *)       echo "Argument inconnu: $1"; exit 1 ;;
    esac
done

# ── Détection automatique du device ──
detect_device() {
    python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'cuda (détecté: {name}, {vram:.0f} Go)')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('mps (Apple Silicon)')
else:
    print('cpu')
" 2>/dev/null | head -1
}

DEVICE_INFO=$(detect_device)
DEVICE=$(echo "${DEVICE_INFO}" | cut -d' ' -f1)

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  nnU-Net PARSE — Pipeline de segmentation               ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Data:   ${DATA_DIR}"
echo "║  Device: ${DEVICE_INFO}"
echo "║  Debug:  ${DEBUG}"
echo "║  From:   ${FROM_STEP}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Vérification de l'environnement ──
if [[ -z "${nnUNet_raw:-}" ]]; then
    echo "ERREUR: Les variables nnU-Net ne sont pas définies."
    echo "Lancez d'abord : source setup_env.sh"
    exit 1
fi

# ── Ordre des étapes ──
STEPS=(convert preprocess generate_degraded train_star train_minus predict evaluate grid_search aggregate)

# Trouver l'index de départ
START_IDX=0
for i in "${!STEPS[@]}"; do
    if [[ "${STEPS[$i]}" == "${FROM_STEP}" ]]; then
        START_IDX=$i
        break
    fi
done

# Folds à entraîner
if [[ "${DEBUG}" == "true" ]]; then
    FOLDS=(0)
    echo "[DEBUG] Mode debug : fold 0 uniquement"
else
    FOLDS=(0 1 2 3 4)
fi

# ── Fonctions utilitaires ──
step_header() {
    local step_name="$1"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ▶ ${step_name}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

should_run() {
    local step="$1"
    for i in "${!STEPS[@]}"; do
        if [[ "${STEPS[$i]}" == "$step" && $i -ge $START_IDX ]]; then
            return 0
        fi
    done
    return 1
}

# ============================================================================
# ÉTAPE 1 : Conversion des données
# ============================================================================
if should_run "convert"; then
    step_header "1/9 — Conversion PARSE → format nnU-Net"

    uv run scripts/convert_parse_to_nnunet.py \
        --data_dir "${DATA_DIR}" \
        --dataset_id ${DATASET_ID}
fi

# ============================================================================
# ÉTAPE 2 : Prétraitement nnU-Net (fingerprint + plans + preprocess)
# ============================================================================
if should_run "preprocess"; then
    step_header "2/9 — Prétraitement nnU-Net"

    nnUNetv2_plan_and_preprocess \
        -d ${DATASET_ID} \
        --verify_dataset_integrity

    echo ""
    echo "Plans générés. Vérifiez les fichiers dans :"
    echo "  ${nnUNet_preprocessed}/Dataset${DATASET_ID}_PARSE/"
fi

# ============================================================================
# ÉTAPE 3 : Génération des labels dégradés (pour l'évaluation)
# ============================================================================
if should_run "generate_degraded"; then
    step_header "3/9 — Génération des labels dégradés (GT-)"

    uv run scripts/generate_degraded_dataset.py \
        --dataset_dir "${nnUNet_raw}/Dataset${DATASET_ID}_PARSE" \
        --config configs/experiment_config.yaml \
        --seed 42
fi

# ============================================================================
# ÉTAPE 4 : Entraînement Model_Star (baseline, GT propres)
# ============================================================================
if should_run "train_star" || should_run "train_minus"; then
    echo "Mise à jour du custom trainer..."
    NNUNET_TRAINER_DIR="$(python3 -c "import nnunetv2; import os; print(os.path.join(nnunetv2.__path__[0], 'training', 'nnUNetTrainer'))")"
    cp "custom_trainers/nnUNetTrainerDegraded.py" "${NNUNET_TRAINER_DIR}/"
fi

if should_run "train_star"; then
    step_header "4/9 — Entraînement Model_Star (nnUNetTrainer)"

    TRAINER="nnUNetTrainer"
    if [[ "${DEBUG}" == "true" ]]; then
        TRAINER="nnUNetTrainerDebug"
    fi

    for FOLD in "${FOLDS[@]}"; do
        echo ""
        echo "  ── Fold ${FOLD}/4 ──"
        nnUNetv2_train \
            ${DATASET_ID} \
            3d_fullres \
            ${FOLD} \
            -tr ${TRAINER} \
            --npz \
            -device "${DEVICE}"
    done

    echo ""
    echo "Model_Star : tous les folds terminés."
fi

# ============================================================================
# ÉTAPE 5 : Entraînement Model_Minus (GT dégradées on-the-fly)
# ============================================================================
if should_run "train_minus"; then
    step_header "5/9 — Entraînement Model_Minus (nnUNetTrainerDegraded)"

    export DEBUG_PIPELINE=$([[ "${DEBUG}" == "true" ]] && echo "1" || echo "0")

    for FOLD in "${FOLDS[@]}"; do
        echo ""
        echo "  ── Fold ${FOLD}/4 ──"
        nnUNetv2_train \
            ${DATASET_ID} \
            3d_fullres \
            ${FOLD} \
            -tr nnUNetTrainerDegraded \
            --npz \
            -device "${DEVICE}"
    done

    echo ""
    echo "Model_Minus : tous les folds terminés."
fi

# ============================================================================
# ÉTAPE 6 : Prédictions des deux modèles sur le test set
# ============================================================================
if should_run "predict"; then
    step_header "6/9 — Prédiction sur le test set"

    IMAGES_TS="${nnUNet_raw}/Dataset${DATASET_ID}_PARSE/imagesTs"
    FOLD_ARGS=$(printf "%s " "${FOLDS[@]}")

    STAR_TRAINER="nnUNetTrainer"
    if [[ "${DEBUG}" == "true" ]]; then
        STAR_TRAINER="nnUNetTrainerDebug"
    fi

    echo "  Prédiction Model_Star (${STAR_TRAINER})..."
    mkdir -p predictions/model_star
    nnUNetv2_predict \
        -i "${IMAGES_TS}" \
        -o predictions/model_star \
        -d ${DATASET_ID} \
        -c 3d_fullres \
        -tr ${STAR_TRAINER} \
        -f ${FOLD_ARGS} \
        -device "${DEVICE}"

    echo ""
    echo "  Prédiction Model_Minus (nnUNetTrainerDegraded)..."
    mkdir -p predictions/model_minus
    nnUNetv2_predict \
        -i "${IMAGES_TS}" \
        -o predictions/model_minus \
        -d ${DATASET_ID} \
        -c 3d_fullres \
        -tr nnUNetTrainerDegraded \
        -f ${FOLD_ARGS} \
        -device "${DEVICE}"
fi

# ============================================================================
# ÉTAPE 7 : Cross-évaluation
# ============================================================================
if should_run "evaluate"; then
    step_header "7/9 — Cross-évaluation Model_Star vs Model_Minus"

    uv run scripts/cross_evaluate.py \
        --config configs/experiment_config.yaml \
        --output results
fi

# ============================================================================
# ÉTAPE 8 : Grid search (optionnel, long)
# ============================================================================
if should_run "grid_search"; then
    step_header "8/9 — Grid search des paramètres de dégradation"

    if [[ "${DEBUG}" == "true" ]]; then
        echo "[DEBUG] Grid search skippé en mode debug"
        echo "        Lancez manuellement :"
        echo "        uv run scripts/grid_search.py --config configs/experiment_config.yaml --phase all"
    else
        uv run scripts/grid_search.py \
            --config configs/experiment_config.yaml \
            --phase all
    fi
fi

# ============================================================================
# ÉTAPE 9 : Agrégation et visualisation
# ============================================================================
if should_run "aggregate"; then
    step_header "9/9 — Agrégation des résultats"

    uv run scripts/aggregate_results.py \
        --results_dir results
fi

# ============================================================================
# Résumé final
# ============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Pipeline terminé !                                     ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Résultats dans :                                       ║"
echo "║    results/cross_evaluation.csv                         ║"
echo "║    results/grid_search_results.csv                      ║"
echo "║    results/figures/                                      ║"
echo "║                                                         ║"
echo "║  Modèles sauvegardés dans :                             ║"
echo "║    ${nnUNet_results}/Dataset${DATASET_ID}_PARSE/        ║"
echo "╚══════════════════════════════════════════════════════════╝"
