# CAF Testing Suite

## Quick Start

```bash
# One-command test (interactive)
./quick_test.sh

# Or manual pre-flight check
python tests/test_preflight_check.py
```

## Test Files

| File | Purpose | Duration |
|------|---------|----------|
| `test_preflight_check.py` | Comprehensive pre-deployment validation | 2-15 min |
| `test_real_fvl_integration.py` | Real SPARQL FVL integration tests | 5 min |
| `quick_test.sh` | Interactive one-command test script | 3-15 min |

## Pre-Flight Check Options

```bash
# Basic check (no GPU, no SPARQL)
python tests/test_preflight_check.py --no-sparql

# With GPU test (requires 4GB+ GPU)
python tests/test_preflight_check.py --test-gpu

# SPARQL only
python tests/test_preflight_check.py --sparql-only

# Full check (GPU + SPARQL)
python tests/test_preflight_check.py --test-gpu
```

## What Gets Tested

### Basic Checks (Always)
- ✅ Python version >= 3.12
- ✅ All required packages installed
- ✅ spaCy model downloaded
- ✅ Mini experiment (2 chains)
- ✅ Resource estimation

### GPU Checks (--test-gpu)
- ✅ CUDA availability
- ✅ GPU memory (4GB+ for 7B model)
- ✅ LLM loading (Llama-2-7b, 4-bit)
- ✅ Inference test

### SPARQL Checks (default, unless --no-sparql)
- ✅ Fuseki server reachable
- ✅ Datasets available
- ✅ Query execution
- ✅ Data loaded

## Expected Results

### ✓ All Passed
```
✓ ALL CHECKS PASSED
System is ready for full experiment deployment!
```
→ **You can proceed with GPU experiments**

### ⚠ Warnings
```
⚠ CHECKS PASSED WITH WARNINGS
System can run but some features may be limited
```
→ **You can proceed but some features won't work**

### ✗ Failed
```
✗ CHECKS FAILED
System is NOT ready - fix errors before deployment
```
→ **DO NOT proceed - fix errors first**

## For Your 4GB GPU

Your setup can run:

| Configuration | Compatible | Notes |
|---------------|------------|-------|
| Simulation only | ✅ Yes | Fast, no GPU needed |
| Llama-2-7B + 4-bit | ✅ Yes | Should fit in 4GB |
| Llama-2-7B + 8-bit | ❌ No | Needs ~8GB |
| Llama-2-13B | ❌ No | Needs ~7GB+ |
| Real SPARQL | ✅ Yes | CPU-based, independent of GPU |

### Recommended Command for 4GB GPU

```bash
# Test first
python tests/test_preflight_check.py --test-gpu

# If passed, run small experiment
python -m experiments.run_experiment \
    --use-llm \
    --llm-4bit \
    --num-chains 10

# If that works, scale up
python -m experiments.run_experiment \
    --use-llm \
    --llm-4bit \
    --num-chains 75
```

## Troubleshooting

See [TESTING_GUIDE.md](../TESTING_GUIDE.md) for detailed troubleshooting.

Common issues:
- **CUDA not available**: Reinstall PyTorch with CUDA support
- **OOM error**: Close other GPU apps, use 4-bit quantization
- **spaCy model not found**: Run `python -m spacy download en_core_web_sm`
- **SPARQL connection refused**: Start Fuseki server

## Output Files

After running tests, check:

```bash
# Pre-flight report
cat tests/preflight_report.json

# Mini experiment results
ls tests/preflight_results/

# Logs (if errors occurred)
tail -f tests/preflight_results/*.log
```

## CI/CD Integration

```bash
# In your CI/CD pipeline
python tests/test_preflight_check.py --no-sparql
if [ $? -eq 0 ]; then
    echo "Tests passed, deploying..."
    python -m experiments.run_experiment --num-chains 75
else
    echo "Tests failed, aborting deployment"
    exit 1
fi
```

## See Also

- [TESTING_GUIDE.md](../TESTING_GUIDE.md) - Comprehensive testing guide
- [REAL_SPARQL_SETUP.md](../REAL_SPARQL_SETUP.md) - SPARQL setup
- [USAGE_EXAMPLES.md](../USAGE_EXAMPLES.md) - Usage examples
