#!/bin/bash
################################################################################
# SLURM Job Script for CAF Experiment on SCIAMA GPU
################################################################################

#SBATCH --partition=gpu.q          # REQUIRED: GPU partition
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem=32G                  # Memory allocation (adjust for model size)
#SBATCH --cpus-per-task=8          # CPU cores
#SBATCH --time=24:00:00            # Max runtime (24 hours)
#SBATCH --job-name=caf_experiment
#SBATCH --output=logs/caf_%j.out   # Output file (%j = job ID)
#SBATCH --error=logs/caf_%j.err    # Error file

################################################################################
# Setup
################################################################################

# Exit on error
set -e

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "CAF Experiment Starting"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Date: $(date)"
echo "=========================================="

################################################################################
# Environment Setup
################################################################################

echo "Loading modules..."
module purge
module load cuda/11.8

echo "Activating virtual environment..."
VENV_DIR="${HOME}/caf_venv"
source "${VENV_DIR}/bin/activate"

# Navigate to project directory
cd "${HOME}/Brightness Computing/PhD/Causal AI/CAF" || cd "${SLURM_SUBMIT_DIR}"

################################################################################
# GPU Verification
################################################################################

echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi

echo ""
echo "CUDA Environment Variables:"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo ""

echo "PyTorch GPU Check:"
python3 << 'EOF'
import torch
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU Memory: {mem_gb:.1f} GB")
EOF

################################################################################
# Run Experiment
################################################################################

echo "=========================================="
echo "Starting CAF Experiment"
echo "=========================================="

# Configuration
NUM_CHAINS=75
PERTURBATIONS=3
OUTPUT_DIR="experiments/results"
MODEL_SIZE="7b"  # Options: 7b, 8b, 13b

# Run the experiment
python -m experiments.run_experiment \
    --use-llm \
    --llm-model ${MODEL_SIZE} \
    --llm-4bit \
    --num-chains ${NUM_CHAINS} \
    --perturbations ${PERTURBATIONS} \
    --output-dir ${OUTPUT_DIR}

################################################################################
# Completion
################################################################################

echo "=========================================="
echo "Job Completed Successfully"
echo "End time: $(date)"
echo "=========================================="

# Show output files
echo "Results saved to:"
ls -lh ${OUTPUT_DIR}/*.json ${OUTPUT_DIR}/*.tex ${OUTPUT_DIR}/*.txt 2>/dev/null || echo "No results found"
