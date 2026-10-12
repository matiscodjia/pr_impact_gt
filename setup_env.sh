#!/usr/bin/env bash
# ============================================================================
# setup_env.sh -- install the environment with uv (once per machine)
#
#   bash setup_env.sh --local      # macOS (debugging, analyses)
#   bash setup_env.sh --cluster    # GPU node (CUDA)
#
# Creates .venv, installs PyTorch, nnU-Net v2 and this repository (package prgt, editable),
# writes .env_nnunet (nnU-Net paths of THIS machine, not tracked by git) and copies the
# custom trainers into nnU-Net.
# ============================================================================

set -euo pipefail

MODE="${1:?Usage: bash setup_env.sh [--local|--cluster]}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  Setup nnU-Net PARSE -- mode: ${MODE}"
echo "  Project: ${PROJECT_DIR}"
echo "============================================"

# 1. uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# 2. virtual environment
echo ""
echo "Creating the virtual environment..."
uv venv .venv --python 3.11
source .venv/bin/activate

# 3. PyTorch for the platform
if [[ "${MODE}" == "--cluster" ]]; then
    echo "Installing PyTorch CUDA 12.1..."
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [[ "$(uname)" == "Darwin" ]]; then
    echo "Installing PyTorch macOS (MPS)..."
    uv pip install torch torchvision
else
    echo "Installing PyTorch CPU..."
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# 4. nnU-Net v2, editable
echo ""
echo "Installing nnU-Net v2 (editable)..."
if [[ ! -d "/tmp/nnUNet_install" ]]; then
    git clone https://github.com/MIC-DKFZ/nnUNet.git /tmp/nnUNet_install
fi
cd /tmp/nnUNet_install
uv pip install -e .
cd "${PROJECT_DIR}"

# 5. this repository (package prgt + its dependencies, see pyproject.toml)
echo ""
echo "Installing the project..."
uv pip install -e ".[train,dev]"

# 6. nnU-Net paths
export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

# environment file sourced at every session
cat > "${PROJECT_DIR}/.env_nnunet" << EOF
export nnUNet_raw="${nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results}"
source "${PROJECT_DIR}/.venv/bin/activate"
EOF

# 7. custom trainers into nnU-Net (they import the noise operator from prgt)
echo ""
echo "Installing the custom trainers..."
NNUNET_TRAINER_DIR="$(python -c "import nnunetv2; import os; print(os.path.join(nnunetv2.__path__[0], 'training', 'nnUNetTrainer'))")"
cp "${PROJECT_DIR}/nnunet_trainers/nnUNetTrainerDegraded.py" "${NNUNET_TRAINER_DIR}/"
echo "  -> copied to ${NNUNET_TRAINER_DIR}/"

# 8. check
echo ""
echo "Checking the installation..."
python -c "
import torch
import nnunetv2
import prgt
print(f'  PyTorch:  {torch.__version__}')
print(f'  nnU-Net:  OK (path: {nnunetv2.__path__[0]})')
print(f'  prgt:     OK (path: {prgt.__path__[0]})')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:     {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'  GPU:      Apple MPS')
else:
    print(f'  GPU:      none (CPU)')
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerDegraded import nnUNetTrainerDegraded
print(f'  Trainer:  nnUNetTrainerDegraded OK')
"

echo ""
echo "============================================"
echo "  Installation done."
echo ""
echo "  At every new session:"
echo "    source .env_nnunet"
echo ""
echo "  Then run the pipeline:"
echo "    bash pipeline/run_pipeline.sh /path/to/PARSE"
echo ""
echo "  Debug mode (2 epochs):"
echo "    bash pipeline/run_pipeline.sh /path/to/PARSE --debug"
echo "============================================"
