#!/usr/bin/env bash
# ============================================================================
# run_pipeline.sh -- the whole study, from the raw PARSE data to the scores
#
# Data preparation runs here, step by step. The heavy loop (k-fold training ->
# scoring -> report) is delegated to pipeline/orchestrator.py, which is resumable: after
# an interruption it restarts from the last checkpoint and refreshes the results after
# every fold. Every step is idempotent, so the same command can be rerun.
#
#   bash pipeline/run_pipeline.sh /path/to/PARSE                    # everything, on GPU
#   bash pipeline/run_pipeline.sh /path/to/PARSE --debug            # 2 epochs, checks the mechanics
#   bash pipeline/run_pipeline.sh /path/to/PARSE --from orchestrate # resume the training
#   bash pipeline/run_pipeline.sh /path/to/PARSE --tiers AB         # also train M3/M4 (Q3)
#
# Steps: convert -> preprocess -> eval_references -> q3_datasets -> orchestrate
# For long sessions, run it inside tmux (detach with Ctrl+B D, come back with tmux attach).
# ============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

check_dependencies() {
    local deps=("python3" "nnUNetv2_train")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            echo "ERROR: missing dependency: $dep"; exit 1
        fi
    done
}
check_dependencies

DATA_DIR="${1:?Usage: $0 /path/to/PARSE [--debug] [--from STEP] [--tiers A|AB]}"
shift

DEBUG=false
FROM_STEP="convert"
DATASET_ID=100
TIERS="A"
CONFIG="configs/experiment_config.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)   DEBUG=true; shift ;;
        --from)    FROM_STEP="$2"; shift 2 ;;
        --tiers)   TIERS="$2"; shift 2 ;;
        --config)  CONFIG="$2"; shift 2 ;;
        *)         echo "unknown argument: $1"; exit 1 ;;
    esac
done

# Device (for information; the orchestrator detects it too)
detect_device() {
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        local name vram
        name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        echo "cuda (${name}, $((vram / 1024)) GB)"
    elif python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        echo "mps (Apple Silicon)"
    else
        echo "cpu"
    fi
}

echo ""
echo "=== PARSE reference-noise study: preparation + training ==="
echo "  data   : ${DATA_DIR}"
echo "  device : $(detect_device)"
echo "  debug  : ${DEBUG}   tiers: ${TIERS}   from: ${FROM_STEP}"
echo ""

if [[ -z "${nnUNet_raw:-}" ]]; then
    echo "ERROR: nnU-Net variables not set. Run first: source .env_nnunet"
    exit 1
fi

STEPS=(convert preprocess eval_references q3_datasets orchestrate)

START_IDX=0
for i in "${!STEPS[@]}"; do
    [[ "${STEPS[$i]}" == "${FROM_STEP}" ]] && { START_IDX=$i; break; }
done

step_header() {
    echo ""
    echo "------------------------------------------------------------"
    echo "  > $1    $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"
}
should_run() {
    for i in "${!STEPS[@]}"; do
        [[ "${STEPS[$i]}" == "$1" && $i -ge $START_IDX ]] && return 0
    done
    return 1
}

# 1. PARSE -> nnU-Net raw format (Dataset100)
if should_run "convert"; then
    step_header "1/5 convert PARSE to the nnU-Net format"
    python pipeline/convert_parse.py --data_dir "${DATA_DIR}" --dataset_id ${DATASET_ID}
fi

# 2. nnU-Net planning and preprocessing of the clean dataset
if should_run "preprocess"; then
    step_header "2/5 nnU-Net preprocessing (Dataset${DATASET_ID})"
    nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity
fi

# 3. Degraded evaluation references GT-, for labelsTr (out-of-fold) and labelsTs
if should_run "eval_references"; then
    step_header "3/5 degraded evaluation references (labelsTr/labelsTs)"
    python pipeline/make_eval_references.py \
        --dataset_dir "${nnUNet_raw}/Dataset${DATASET_ID}_PARSE" \
        --config "${CONFIG}" --seed 42
fi

# 4. Frozen training sets of M3/M4 (Dataset103/104), with the plans of Dataset100.
#    Only needed for tier B.
if should_run "q3_datasets" && [[ "${TIERS}" == *B* ]]; then
    step_header "4/5 frozen training sets of M3/M4 (Dataset103/104)"
    bash cluster/prepare_q3.sh
fi

# 5. Resumable k-fold training -> scoring -> report
if should_run "orchestrate"; then
    step_header "5/5 resumable k-fold training, scoring and report"
    DEBUG_FLAG=""
    [[ "${DEBUG}" == "true" ]] && DEBUG_FLAG="--debug"
    python pipeline/orchestrator.py --config "${CONFIG}" --tiers "${TIERS}" ${DEBUG_FLAG}
fi

echo ""
echo "=== Done ==="
echo "  dashboard    : results/STATUS.md"
echo "  master table : results/metrics.csv"
echo "  paper        : run the scripts of analysis/ (see README.md)"
echo "  resume       : rerun the same command (every step is idempotent)"
