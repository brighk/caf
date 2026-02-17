#!/bin/bash
# Activation helper for CAF virtual environment

echo "Activating CAF virtual environment..."
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
else
  echo "No virtual environment found (.venv or venv)."
  return 1 2>/dev/null || exit 1
fi

# Keep model/cache/temp writes on /workspace (large volume), not root overlay.
export HF_HOME="/workspace/caf/.hf-cache"
export HUGGINGFACE_HUB_CACHE="/workspace/caf/.hf-cache/hub"
export TRANSFORMERS_CACHE="/workspace/caf/.hf-cache/transformers"
export HF_XET_CACHE="/workspace/caf/.hf-cache/xet"
export TMPDIR="/workspace/caf/.tmp"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_XET_CACHE" "$TMPDIR"

echo "✓ Environment activated!"
echo "✓ HF cache: $HF_HOME"
echo ""
echo "Installed packages:"
echo "  - PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "  - CUDA: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "N/A")')"
echo "  - GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "  - Transformers $(python -c 'import transformers; print(transformers.__version__)')"
echo ""
echo "Quick test commands:"
echo "  python test_llm.py                                  # Test LLM loading"
echo "  python -m experiments.run_experiment --use-llm --llm-4bit --num-chains 5"
echo ""
