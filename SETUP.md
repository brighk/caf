# CAF Environment Setup on Projects Partition

## ✅ Step 1: Directory Copied

You've already copied CAF to the projects partition:
```
Location: /home/bright/projects/PhD/CAF
Size: 24MB
Available space: 159GB
```

## Step 2: Create Virtual Environment

```bash
# Navigate to CAF directory
cd /home/bright/projects/PhD/CAF

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Verify activation (you should see (venv) in your prompt)
which python
# Should show: /home/bright/projects/PhD/CAF/venv/bin/python
```

## Step 3: Install Dependencies

### Option A: Basic Installation (Recommended First)

```bash
# Upgrade pip
pip install --upgrade pip

# Install basic dependencies
pip install -r requirements.txt
```

### Option B: With GPU Support (After basic install works)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers with optimizations
pip install transformers accelerate bitsandbytes

# Install other dependencies
pip install -r requirements.txt
```

## Step 4: Verify Installation

```bash
# Check Python packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check GPU
nvidia-smi

# Run pre-flight check
python tests/test_preflight_check.py
```

## Step 5: Quick Test

```bash
# Test without GPU (simulation mode)
python -m experiments.run_counterbench_experiment \
    --input data/counterbench_caf.json \
    --limit 5 \
    --output results/test

# Test with GPU (requires GPU setup)
python -m experiments.run_counterbench_experiment \
    --input data/counterbench_caf.json \
    --limit 5 \
    --use-llm \
    --llm-model tiny \
    --llm-4bit \
    --output results/test_gpu
```

## Troubleshooting

### If venv creation fails:

```bash
# Install python3-venv
sudo apt-get install python3-venv

# Or use conda instead
conda create -n caf python=3.12
conda activate caf
```

### If CUDA not available:

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### If out of disk space during installation:

```bash
# Clean pip cache
pip cache purge

# Install with --no-cache-dir
pip install --no-cache-dir -r requirements.txt
```

## Activation for Future Sessions

```bash
# Every time you start a new terminal session:
cd /home/bright/projects/PhD/CAF
source venv/bin/activate

# Or use the helper script:
source activate.sh
```

## Update activate.sh

The activate.sh file needs to be updated since venv will be in a different location:

```bash
#!/bin/bash
# Activation helper for CAF virtual environment

# Change to CAF directory
cd /home/bright/projects/PhD/CAF

echo "Activating CAF virtual environment..."
source venv/bin/activate

echo "✓ Environment activated!"
echo ""
echo "Current directory: $(pwd)"
echo ""
echo "Installed packages:"
echo "  - Python: $(python --version)"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  - CUDA: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "N/A")' 2>/dev/null || echo 'N/A')"
echo "  - GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>/dev/null || echo 'N/A')"
echo "  - Transformers: $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'Not installed')"
echo ""
echo "Quick start:"
echo "  bash scripts/test_small_llm.sh          # Test all models"
echo "  python -m experiments.run_counterbench_experiment --help"
echo ""
```

## Space Considerations

### Current Setup
- CAF code: 24MB
- Virtual environment: ~500MB - 1GB
- PyTorch + CUDA: ~2-3GB
- Downloaded models cache: 3-7GB per model

### Total Estimated Space Needed
- Minimal (no LLM): ~2GB
- With TinyLlama: ~3GB
- With Llama-2-7B: ~6GB
- With multiple models: ~10-15GB

### Your projects partition has 159GB available - plenty of space! ✅

## Next Steps After Setup

1. Download CounterBench dataset:
   ```bash
   python scripts/load_counterbench.py \
       --output data/counterbench_caf.json \
       --limit 100
   ```

2. Start Fuseki (if using SPARQL):
   ```bash
   cd ~/apache-jena-fuseki-4.10.0
   ./fuseki-server --mem /counterbench &
   ```

3. Run experiments:
   ```bash
   # See README_4GB_GPU.md for complete guide
   ```

## Quick Reference

```bash
# Setup (once)
cd /home/bright/projects/PhD/CAF
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Activate (every session)
cd /home/bright/projects/PhD/CAF
source venv/bin/activate

# Run
python -m experiments.run_counterbench_experiment --help
```
