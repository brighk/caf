#!/bin/bash
################################################################################
# Virtual Environment Setup Script for CAF on SCIAMA
# Run this script ONCE on Sciama to set up your Python environment
################################################################################

set -e  # Exit on error

echo "=========================================="
echo "CAF Virtual Environment Setup for SCIAMA"
echo "=========================================="

# 1. Load required modules
echo "Loading system modules..."
module purge
module load cuda/11.8  # Adjust version based on what's available
module load python/3.12  # Or python3 - check available versions with: module avail python

# 2. Create virtual environment
VENV_DIR="${HOME}/caf_venv"
echo "Creating virtual environment at: ${VENV_DIR}"

if [ -d "${VENV_DIR}" ]; then
    echo "WARNING: Virtual environment already exists at ${VENV_DIR}"
    read -p "Do you want to remove it and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${VENV_DIR}"
    else
        echo "Keeping existing environment. Exiting."
        exit 0
    fi
fi

python3 -m venv "${VENV_DIR}"

# 3. Activate virtual environment
echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# 4. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# 5. Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 6. Install core dependencies for experiments
echo "Installing core experiment dependencies..."
pip install \
    numpy \
    transformers \
    accelerate \
    bitsandbytes \
    sentencepiece \
    protobuf

# 7. Install optional dependencies (if needed for full CAF framework)
echo "Installing optional dependencies..."
pip install \
    rdflib \
    SPARQLWrapper \
    requests \
    httpx \
    loguru \
    python-dotenv \
    tenacity

# 8. Verify GPU access with PyTorch
echo "=========================================="
echo "Verifying PyTorch CUDA installation..."
echo "=========================================="
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available! GPU jobs will fail.")
EOF

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment created at: ${VENV_DIR}"
echo ""
echo "To activate this environment in your job script, add:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To test manually:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
