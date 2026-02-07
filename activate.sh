#!/bin/bash
# Activation helper for CAF virtual environment

echo "Activating CAF virtual environment..."
source venv/bin/activate

echo "✓ Environment activated!"
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
