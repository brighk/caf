# START HERE - CAF Setup on Projects Partition

## ✅ Step 1: You've Already Copied CAF

Your CAF directory is now at:
```
/home/bright/projects/PhD/CAF
```

This partition has **159GB free space** - perfect for CAF! ✅

## Step 2: Run Automated Setup

```bash
# Navigate to CAF directory
cd /home/bright/projects/PhD/CAF

# Run setup script (this will take 5-10 minutes)
bash setup_env.sh
```

This script will:
- ✅ Create Python virtual environment
- ✅ Install all dependencies (PyTorch, Transformers, etc.)
- ✅ Download spaCy model
- ✅ Test GPU availability
- ✅ Create necessary directories

## Step 3: Activate Environment

Every time you open a new terminal:

```bash
cd /home/bright/projects/PhD/CAF
source venv/bin/activate
```

You should see `(venv)` in your prompt.

## Step 4: Quick Test

```bash
# Test simulation mode (no GPU needed)
python -m experiments.run_counterbench_experiment \
    --input data/counterbench_caf.json \
    --limit 5 \
    --output results/quick_test

# If GPU available, test TinyLlama (0.6GB)
python -m experiments.run_counterbench_experiment \
    --input data/counterbench_caf.json \
    --limit 5 \
    --use-llm \
    --llm-model tiny \
    --llm-4bit \
    --output results/gpu_test
```

## What If Setup Fails?

### Manual Setup

```bash
cd /home/bright/projects/PhD/CAF

# 1. Create venv
python3 -m venv venv

# 2. Activate
source venv/bin/activate

# 3. Install pip packages
pip install --upgrade pip
pip install torch transformers datasets spacy rdflib requests

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Test
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### If python3-venv not installed:

```bash
sudo apt-get update
sudo apt-get install python3-venv
```

### If using conda instead:

```bash
conda create -n caf python=3.12
conda activate caf
pip install -r requirements.txt
```

## Next Steps After Setup

### 1. Load CounterBench Dataset

```bash
cd /home/bright/projects/PhD/CAF
source venv/bin/activate

python scripts/load_counterbench.py \
    --output data/counterbench_caf.json \
    --limit 100 \
    --stats
```

### 2. Start Fuseki (for SPARQL verification)

```bash
# If you have Fuseki installed
cd ~/apache-jena-fuseki-4.10.0
./fuseki-server --mem /counterbench &
```

### 3. Run CAF Experiments

See these guides for complete instructions:

- **[README_4GB_GPU.md](README_4GB_GPU.md)** - Running on 4GB GPU
- **[SMALL_LLM_GUIDE.md](SMALL_LLM_GUIDE.md)** - Using small LLMs
- **[COUNTERBENCH_GUIDE.md](COUNTERBENCH_GUIDE.md)** - Complete guide
- **[COUNTERBENCH_QUICKSTART.md](COUNTERBENCH_QUICKSTART.md)** - Quick reference

## Quick Reference

### Activate Environment
```bash
cd /home/bright/projects/PhD/CAF
source venv/bin/activate
# or
source activate.sh
```

### Run Experiments

```bash
# Simulation (no GPU)
python -m experiments.run_counterbench_experiment \
    --input data/counterbench_caf.json \
    --output results/sim

# With TinyLlama (0.6GB)
python -m experiments.run_counterbench_experiment \
    --input data/counterbench_caf.json \
    --use-llm --llm-model tiny --llm-4bit \
    --output results/tiny

# With Llama-2-7B (3.5GB, recommended)
python -m experiments.run_counterbench_experiment \
    --input data/counterbench_caf.json \
    --use-llm --llm-model 7b --llm-4bit \
    --use-real-sparql --extract-kb \
    --sparql-endpoint http://localhost:3030/counterbench/query \
    --output results/llama7b
```

### Check GPU
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `command not found: python3` | Install Python 3: `sudo apt-get install python3` |
| `No module named 'venv'` | Install venv: `sudo apt-get install python3-venv` |
| CUDA not available | Check driver: `nvidia-smi`, reinstall PyTorch |
| Out of disk space | You're on projects partition (159GB free), should be fine |
| Permission denied | Make script executable: `chmod +x setup_env.sh` |

## Directory Structure

```
/home/bright/projects/PhD/CAF/
├── START_HERE.md              ← You are here
├── setup_env.sh              ← Run this to setup
├── activate.sh               ← Use this to activate
├── SETUP.md                  ← Detailed setup guide
├── README_4GB_GPU.md         ← How to run on 4GB GPU
├── SMALL_LLM_GUIDE.md        ← Small LLM guide
├── COUNTERBENCH_GUIDE.md     ← Complete CounterBench guide
├── venv/                     ← Virtual environment (created by setup)
├── data/                     ← Datasets
├── results/                  ← Experiment results
├── experiments/              ← CAF code
├── scripts/                  ← Helper scripts
└── ...
```

## Summary

**Setup**: `bash setup_env.sh` (one time)
**Activate**: `source venv/bin/activate` (every session)
**Run**: See README_4GB_GPU.md

Your 4GB GPU is perfect for CAF! 🎉
