# CAF Setup Guide for SCIAMA GPU Computing

This guide will help you set up and run your CAF experiment on the SCIAMA GPU cluster.

## Prerequisites

- Access to SCIAMA cluster
- Your CAF project code uploaded to SCIAMA
- SSH access configured

## Step-by-Step Setup

### 1. Upload Your Code to SCIAMA

```bash
# From your local machine
rsync -avz --exclude='*.pyc' --exclude='__pycache__' \
    "/home/bright/Brightness Computing/PhD/Causal AI/CAF/" \
    your_username@sciama.icg.port.ac.uk:~/CAF/
```

### 2. SSH into SCIAMA

```bash
ssh your_username@sciama.icg.port.ac.uk
```

### 3. Set Up Virtual Environment (ONE-TIME SETUP)

```bash
# Navigate to your project
cd ~/CAF/deployment/sciama/

# Make setup script executable
chmod +x setup_venv.sh

# Run the setup script
./setup_venv.sh
```

This script will:
- Load CUDA and Python modules
- Create a virtual environment at `~/caf_venv`
- Install PyTorch with GPU support
- Install all required dependencies
- Verify GPU access

**Expected output:** You should see "CUDA available: True" at the end.

### 4. Verify Your Setup

```bash
# Activate the environment
source ~/caf_venv/bin/activate

# Test PyTorch GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test transformers
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 5. Submit Your GPU Job

```bash
# Navigate to deployment directory
cd ~/CAF/deployment/sciama/

# Create logs directory
mkdir -p logs

# Submit the job
sbatch run_caf_gpu.sbatch
```

### 6. Monitor Your Job

```bash
# Check job status
squeue -u $USER

# View real-time output
tail -f logs/caf_<job_id>.out

# View errors (if any)
tail -f logs/caf_<job_id>.err

# Check GPU usage (while job is running)
srun --jobid=<job_id> --pty nvidia-smi
```

## Configuration Options

### Model Size and Memory Requirements

Edit `run_caf_gpu.sbatch` to adjust:

```bash
# For 7B model (recommended for testing)
#SBATCH --mem=20G
MODEL_SIZE="7b"

# For 8B model
#SBATCH --mem=24G
MODEL_SIZE="8b"

# For 13B model
#SBATCH --mem=40G
MODEL_SIZE="13b"
```

### Experiment Parameters

In `run_caf_gpu.sbatch`:

```bash
NUM_CHAINS=75           # Number of causal chains (50-100)
PERTURBATIONS=3         # Perturbations per chain (2-3)
OUTPUT_DIR="experiments/results"
```

### Quantization Options

- `--llm-4bit`: Use 4-bit quantization (saves memory, recommended)
- `--llm-8bit`: Use 8-bit quantization (alternative)
- No flag: Full precision (requires more memory)

## Troubleshooting

### CUDA Not Available

If PyTorch can't see CUDA:
```bash
# Check available CUDA modules
module avail cuda

# Load specific version
module load cuda/11.8  # or cuda/12.1, etc.
```

### Out of Memory

1. Reduce model size: 13b → 8b → 7b
2. Enable 4-bit quantization: `--llm-4bit`
3. Reduce batch size in the code
4. Request more memory: `#SBATCH --mem=40G`

### Job Pending

Check GPU availability:
```bash
sinfo -p gpu.q
squeue -p gpu.q
```

SCIAMA has 4 GPUs total. If all are busy, your job will queue.

### Module Not Found

Make sure you're using the virtual environment:
```bash
source ~/caf_venv/bin/activate
```

### Permission Denied

Make scripts executable:
```bash
chmod +x setup_venv.sh
chmod +x run_caf_gpu.sbatch
```

## Interactive Testing

For quick tests, use an interactive GPU session:

```bash
# Start interactive session
srun --pty --mem=16G --gres=gpu:1 -p gpu.q /bin/bash

# Activate environment
source ~/caf_venv/bin/activate

# Navigate to project
cd ~/CAF

# Test with small dataset
python -m experiments.run_experiment \
    --use-llm \
    --llm-model 7b \
    --llm-4bit \
    --num-chains 10 \
    --perturbations 2
```

## Expected Runtime

- **Simulated mode** (no `--use-llm`): ~5-10 minutes for 75 chains
- **7B model with GPU**: ~2-4 hours for 75 chains
- **13B model with GPU**: ~4-8 hours for 75 chains

## Output Files

Results are saved to `experiments/results/`:
- `synthetic_dataset_*.json` - Generated dataset
- `experiment_metrics_*.json` - All metrics
- `results_table_*.tex` - LaTeX table
- `experiment_report_*.txt` - Human-readable report
- `algorithm_*.tex` - Algorithm pseudocode

## Useful Commands

```bash
# Cancel a job
scancel <job_id>

# Check job details
scontrol show job <job_id>

# View completed job info
sacct -j <job_id> --format=JobID,JobName,Partition,State,Elapsed,MaxRSS

# Check your quota
quota -s

# See your recent jobs
sacct -u $USER --starttime $(date -d '7 days ago' +%Y-%m-%d)
```

## Additional Resources

- SCIAMA GPU Documentation: `Using GPUs on SCIAMA – SCIAMA.html`
- SCIAMA Job Submission: `Submitting Jobs – SCIAMA.html`
- GPU Quick Reference: `sciama_gpu_quick_reference.md`
- GPU Cheatsheet: `SCIAMA_GPU_CHEATSHEET.txt`

## Contact

If you encounter issues specific to SCIAMA infrastructure:
- Check SCIAMA status page
- Contact SCIAMA support team

For CAF-specific issues:
- Review experiment logs in `logs/`
- Check Python traceback in `.err` file
