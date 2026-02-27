#!/bin/bash
# Setup CAF environment on projects partition

set -e  # Exit on error

echo "=========================================="
echo "CAF Environment Setup"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"
echo ""

# Check disk space
echo "Checking disk space..."
df -h "$SCRIPT_DIR" | tail -1
echo ""

# Step 1: Create virtual environment
if [ -d "venv" ]; then
    echo "✓ Virtual environment already exists"
else
    echo "Creating virtual environment..."
    python3 -m venv venv || {
        echo "✗ Failed to create venv. Install python3-venv:"
        echo "  sudo apt-get install python3-venv"
        exit 1
    }
    echo "✓ Virtual environment created"
fi

# Step 2: Activate
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Activated"

# Step 3: Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ Pip upgraded"

# Step 4: Install dependencies
echo ""
echo "Installing dependencies..."
echo "(This may take several minutes...)"

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt || {
        echo "⚠ Warning: Some packages failed to install"
        echo "Continuing anyway..."
    }
else
    echo "⚠ requirements.txt not found, installing minimal dependencies..."
    pip install torch transformers datasets spacy rdflib requests
fi

echo "✓ Dependencies installed"

# Step 5: Download spaCy model (if needed)
echo ""
echo "Checking spaCy model..."
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null && {
    echo "✓ spaCy model already installed"
} || {
    echo "Downloading spaCy model..."
    python -m spacy download en_core_web_sm
    echo "✓ spaCy model downloaded"
}

# Step 6: Test installation
echo ""
echo "=========================================="
echo "Testing Installation"
echo "=========================================="
echo ""

echo "Python version:"
python --version

echo ""
echo "Checking packages:"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"
python -c "import datasets; print(f'✓ Datasets {datasets.__version__}')"
python -c "import spacy; print(f'✓ spaCy {spacy.__version__}')"

echo ""
echo "Checking GPU:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
else
    echo "⚠ No GPU detected (CPU mode will work for simulation)"
fi

# Step 7: Create data directory
echo ""
echo "Creating directories..."
mkdir -p data results logs
echo "✓ Directories created"

# Step 8: Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Location: $SCRIPT_DIR"
echo ""
echo "To activate environment in new terminal sessions:"
echo "  cd $SCRIPT_DIR"
echo "  source venv/bin/activate"
echo ""
echo "Or use the activation helper:"
echo "  source $SCRIPT_DIR/activate.sh"
echo ""
echo "Next steps:"
echo "  1. Load CounterBench dataset:"
echo "     python scripts/load_counterbench.py --output data/counterbench.json --limit 100"
echo ""
echo "  2. Test simulation mode (no GPU needed):"
echo "     python -m experiments.run_counterbench_experiment \\"
echo "         --input data/counterbench.json --limit 5 --output results/test"
echo ""
echo "  3. See README_4GB_GPU.md for GPU setup and usage"
echo ""
echo "Documentation:"
echo "  - SETUP.md              # This setup process explained"
echo "  - README_4GB_GPU.md     # Running on 4GB GPU"
echo "  - SMALL_LLM_GUIDE.md    # Using small LLMs"
echo "  - COUNTERBENCH_GUIDE.md # Complete CounterBench guide"
echo ""
