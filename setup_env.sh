#!/usr/bin/env bash
# ============================================================================
# setup_env.sh — Installation et configuration avec uv
#
# Usage:
#   # Sur macOS (debug/validation)
#   bash setup_env.sh --local
#
#   # Sur le serveur GPU A40 via SSH
#   bash setup_env.sh --cluster
# ============================================================================

set -euo pipefail

MODE="${1:---local}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  Setup nnU-Net PARSE — mode: ${MODE}"
echo "  Project: ${PROJECT_DIR}"
echo "============================================"

# ── 1. Vérifier que uv est installé ──
if ! command -v uv &> /dev/null; then
    echo "Installation de uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# ── 2. Créer le venv et installer les dépendances de base ──
echo ""
echo "Création de l'environnement virtuel..."
uv venv .venv --python 3.11
source .venv/bin/activate

# ── 3. PyTorch selon la plateforme ──
if [[ "${MODE}" == "--cluster" ]]; then
    echo "Installation PyTorch CUDA 12.1..."
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [[ "$(uname)" == "Darwin" ]]; then
    echo "Installation PyTorch macOS (MPS)..."
    uv pip install torch torchvision
else
    echo "Installation PyTorch CPU..."
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# ── 4. nnU-Net v2 en mode éditable ──
echo ""
echo "Installation nnU-Net v2 (mode éditable)..."
if [[ ! -d "/tmp/nnUNet_install" ]]; then
    git clone https://github.com/MIC-DKFZ/nnUNet.git /tmp/nnUNet_install
fi
cd /tmp/nnUNet_install
uv pip install -e .
cd "${PROJECT_DIR}"

# ── 5. Dépendances du projet ──
echo ""
echo "Installation des dépendances du projet..."
uv pip install -r requirements.txt

# ── 6. Chemins nnU-Net ──
export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

# Fichier source pour les sessions futures
cat > "${PROJECT_DIR}/.env_nnunet" << EOF
export nnUNet_raw="${nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results}"
source "${PROJECT_DIR}/.venv/bin/activate"
EOF

# ── 7. Installer le custom trainer dans nnU-Net ──
echo ""
echo "Installation du custom trainer..."
NNUNET_TRAINER_DIR="$(python -c "import nnunetv2; import os; print(os.path.join(nnunetv2.__path__[0], 'training', 'nnUNetTrainer'))")"
cp "${PROJECT_DIR}/custom_trainers/nnUNetTrainerDegraded.py" "${NNUNET_TRAINER_DIR}/"
echo "  -> Copié dans ${NNUNET_TRAINER_DIR}/"

# ── 8. Vérification ──
echo ""
echo "Vérification de l'installation..."
python -c "
import torch
import nnunetv2
print(f'  PyTorch:  {torch.__version__}')
print(f'  nnU-Net:  OK (path: {nnunetv2.__path__[0]})')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:     {torch.cuda.get_device_properties(0).total_mem/1e9:.0f} Go')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'  GPU:      Apple MPS')
else:
    print(f'  GPU:      Aucun (CPU)')
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerDegraded import nnUNetTrainerDegraded
print(f'  Trainer:  nnUNetTrainerDegraded OK')
"

echo ""
echo "============================================"
echo "  Installation terminée !"
echo ""
echo "  Pour chaque nouvelle session SSH :"
echo "    source .env_nnunet"
echo ""
echo "  Puis lancez le pipeline :"
echo "    bash run_pipeline.sh /chemin/vers/PARSE"
echo ""
echo "  Mode debug (macOS) :"
echo "    bash run_pipeline.sh /chemin/vers/PARSE --debug"
echo "============================================"
