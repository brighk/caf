# CAF Testing Guide

Complete guide for testing CAF before deploying to GPU hardware.

## Quick Pre-Flight Check (5 minutes)

```bash
# Basic check (no GPU required)
python tests/test_preflight_check.py

# This will test:
# ✓ Python version and dependencies
# ✓ spaCy model installed
# ✓ Mini experiment with 2 chains (simulation)
# ✓ Resource estimation for full run
```

## Pre-Flight Check with GPU (15 minutes)

```bash
# Full check including GPU and LLM loading
python tests/test_preflight_check.py --test-gpu

# This will test:
# ✓ CUDA availability
# ✓ GPU memory (needs 4GB+ for 7B model)
# ✓ LLM loading (Llama-2-7b with 4-bit)
# ✓ Inference test
# ✓ Full mini experiment
```

## Pre-Flight Check with SPARQL (10 minutes)

```bash
# 1. Start Fuseki (in separate terminal)
cd ~/apache-jena-fuseki-4.10.0
./fuseki-server --mem /conceptnet &

# 2. Load minimal test data (1000 triples)
python scripts/convert_conceptnet_to_rdf.py \
    ~/data/conceptnet-assertions-5.7.0.csv \
    --limit 1000 \
    --output /tmp/test_conceptnet.nt

curl -X POST -H "Content-Type: application/n-triples" \
    --data-binary @/tmp/test_conceptnet.nt \
    http://localhost:3030/conceptnet/data

# 3. Run pre-flight with SPARQL test
python tests/test_preflight_check.py --test-gpu

# Or SPARQL only
python tests/test_preflight_check.py --sparql-only
```

## What Each Test Checks

### 1. Python Environment ✅
- Python version >= 3.12
- All required packages installed
- Optional packages (warns if missing)

### 2. GPU/CUDA ✅
- CUDA availability
- GPU device count and names
- VRAM capacity (needs 4GB+ for 7B model)
- Compute capability

### 3. spaCy Model ✅
- `en_core_web_sm` downloaded
- Can load and parse text

### 4. SPARQL Endpoint (optional) ✅
- Server reachable at localhost:3030
- Datasets available
- Can execute queries
- Triple count validation

### 5. LLM Loading (if --test-gpu) ✅
- Can load Llama-2-7b-chat with 4-bit quantization
- Model fits in GPU memory
- Inference works correctly

### 6. Mini Experiment ✅
- Generate 2 synthetic chains
- Run CAF verification loop
- Compute metrics
- Validate outputs

### 7. Resource Estimation ✅
- Time estimate for full experiment
- GPU memory requirements
- Disk space needed

## Expected Output

### ✅ Success (All Passed)
```
======================================================================
                    Pre-Flight Check Summary
======================================================================

✓ ALL CHECKS PASSED
System is ready for full experiment deployment!

Errors: 0
Warnings: 0

Report saved to: tests/preflight_report.json
```

### ⚠️ Warnings (Can proceed)
```
======================================================================
                    Pre-Flight Check Summary
======================================================================

⚠ CHECKS PASSED WITH WARNINGS
System can run but some features may be limited

Errors: 0
Warnings: 2
  ⚠ Optional package not installed: fuzzywuzzy
  ⚠ SPARQL endpoint not reachable at http://localhost:3030
```

### ✗ Errors (Do NOT proceed)
```
======================================================================
                    Pre-Flight Check Summary
======================================================================

✗ CHECKS FAILED
System is NOT ready - fix errors before deployment

Errors: 2
  ✗ Missing required package: transformers
  ✗ No GPU available for LLM inference

Warnings: 0
```

## Common Issues and Fixes

### Issue: "CUDA not available"

**Cause**: PyTorch not installed with CUDA support

**Fix**:
```bash
# Reinstall PyTorch with CUDA 12.1
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "GPU memory insufficient"

**Cause**: GPU has < 4GB VRAM

**Fix**: Use smaller model or CPU-only mode
```bash
# Try CPU-only (slow but works)
python -m experiments.run_experiment --num-chains 5

# Or use even smaller model
python tests/test_preflight_check.py --test-gpu --model-size 7b
```

### Issue: "spaCy model not found"

**Cause**: Model not downloaded

**Fix**:
```bash
python -m spacy download en_core_web_sm

# Or larger model for better accuracy
python -m spacy download en_core_web_lg
```

### Issue: "SPARQL connection refused"

**Cause**: Fuseki not running

**Fix**:
```bash
# Start Fuseki
cd ~/apache-jena-fuseki-4.10.0
./fuseki-server --mem /conceptnet &

# Verify it's running
curl http://localhost:3030/$/ping
```

### Issue: "Out of memory during LLM loading"

**Cause**: GPU too small or other processes using VRAM

**Fix**:
```bash
# Clear GPU memory
nvidia-smi  # Check what's using GPU
kill <process-id>  # Kill processes using GPU

# Or reduce batch size (if supported)
export CUDA_VISIBLE_DEVICES=0
python tests/test_preflight_check.py --test-gpu
```

### Issue: "Model download timeout"

**Cause**: Slow internet or HuggingFace down

**Fix**:
```bash
# Pre-download model
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Model will be cached at ~/.cache/huggingface/
```

## Interpreting Resource Estimates

After pre-flight check, you'll see estimates like:

```
Simulation mode (no GPU):
  - Estimated time: 1.9 minutes
  - GPU memory: Not required
  - Disk space: ~50MB

Real LLM mode (Llama-2-7b, 4-bit):
  - Estimated time: 56.3 minutes (0.9 hours)
  - GPU memory needed: ~4.5GB
  - Disk space: ~100MB
```

### Time Estimates

| Mode | Chains | Est. Time | Notes |
|------|--------|-----------|-------|
| Simulation | 75 | ~2 min | Fast, no GPU |
| Simulation | 300 | ~8 min | Large dataset |
| LLM 7B + Sim FVL | 75 | ~56 min | GPU required |
| LLM 7B + Real SPARQL | 75 | ~70 min | GPU + triplestore |

### GPU Memory Requirements

| Model | Quantization | VRAM Needed | Your GPU |
|-------|--------------|-------------|----------|
| Llama-2-7B | 4-bit | ~4.5GB | ✓ (4GB may work) |
| Llama-2-7B | 8-bit | ~8GB | ✗ (insufficient) |
| Llama-2-13B | 4-bit | ~7GB | ✗ (insufficient) |

**Your 4GB GPU**: Can run Llama-2-7B with 4-bit quantization, but it will be tight. Close all other GPU applications.

## Step-by-Step: First Time Setup

### 1. Basic Setup (No GPU)

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run basic check
python tests/test_preflight_check.py

# If passed, run small experiment
python -m experiments.run_experiment --num-chains 5
```

### 2. GPU Setup

```bash
# Check GPU
nvidia-smi

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Test GPU
python tests/test_preflight_check.py --test-gpu

# If passed, run with real LLM
python -m experiments.run_experiment \
    --use-llm \
    --llm-4bit \
    --num-chains 5
```

### 3. Full Production Setup

```bash
# Start Fuseki
./fuseki-server --mem /conceptnet &

# Load ConceptNet data (10K triples for testing)
python scripts/convert_conceptnet_to_rdf.py \
    conceptnet-assertions-5.7.0.csv \
    --limit 10000 \
    --output conceptnet_10k.nt

curl -X POST -H "Content-Type: application/n-triples" \
    --data-binary @conceptnet_10k.nt \
    http://localhost:3030/conceptnet/data

# Full pre-flight check
python tests/test_preflight_check.py --test-gpu

# If passed, run full production experiment
python -m experiments.run_experiment \
    --use-llm \
    --llm-4bit \
    --use-real-sparql \
    --num-chains 25
```

## Continuous Monitoring

While running experiments, monitor resources:

```bash
# GPU usage
watch -n 1 nvidia-smi

# CPU and RAM
htop

# Disk space
df -h

# SPARQL endpoint health
curl http://localhost:3030/$/ping
```

## Recommended Test Sequence

Before running on expensive GPU hardware, follow this sequence:

1. ✅ **Basic check** (2 min)
   ```bash
   python tests/test_preflight_check.py
   ```

2. ✅ **Small simulation** (5 min)
   ```bash
   python -m experiments.run_experiment --num-chains 5
   ```

3. ✅ **GPU check** (15 min)
   ```bash
   python tests/test_preflight_check.py --test-gpu
   ```

4. ✅ **Small GPU experiment** (15 min)
   ```bash
   python -m experiments.run_experiment --use-llm --llm-4bit --num-chains 5
   ```

5. ✅ **SPARQL test** (10 min)
   ```bash
   # Setup and test as shown above
   python tests/test_preflight_check.py --sparql-only
   ```

6. ✅ **Medium experiment** (30 min)
   ```bash
   python -m experiments.run_experiment --use-llm --llm-4bit --num-chains 25
   ```

7. 🚀 **Full deployment** (1-2 hours)
   ```bash
   python -m experiments.run_experiment \
       --use-llm \
       --llm-4bit \
       --use-real-sparql \
       --num-chains 75
   ```

## Exit Codes

The pre-flight check returns exit codes for scripting:

- `0`: All checks passed, ready to deploy
- `1`: Errors found, NOT ready to deploy

Example in shell script:
```bash
#!/bin/bash

if python tests/test_preflight_check.py --test-gpu; then
    echo "Pre-flight passed, starting full experiment..."
    python -m experiments.run_experiment --use-llm --num-chains 75
else
    echo "Pre-flight failed, aborting!"
    exit 1
fi
```

## Report Files

Pre-flight check generates a JSON report:

```json
{
  "status": "READY",
  "errors": [],
  "warnings": [
    "Optional package not installed: fuzzywuzzy"
  ],
  "timestamp": "2026-02-15 12:34:56"
}
```

Use this for automated CI/CD or deployment pipelines.

## See Also

- [REAL_SPARQL_SETUP.md](REAL_SPARQL_SETUP.md) - SPARQL setup guide
- [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - Usage examples
- [README.md](README.md) - Main documentation
