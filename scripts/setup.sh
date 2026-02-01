#!/bin/bash
# CAF Setup Script
# Sets up the development environment

set -e

echo "=========================================="
echo "CAF Setup - Causal Autonomy Framework"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.12"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "ERROR: Python 3.12+ is required. Found: $python_version"
    exit 1
fi
echo "✓ Python version: $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "✓ Virtual environment created"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_lg
echo "✓ spaCy model downloaded"
echo ""

# Create data directories
echo "Creating data directories..."
mkdir -p data/rdf data/vectors logs
echo "✓ Data directories created"
echo ""

# Copy environment template
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo "⚠️  WARNING: Please edit .env and configure your settings!"
else
    echo "✓ .env file already exists"
fi
echo ""

# Check for GPU
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✓ NVIDIA GPU detected"
else
    echo "⚠️  WARNING: No NVIDIA GPU detected. CPU inference will be slow."
fi
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Edit .env file with your configuration"
echo "3. Start services with: docker-compose up -d"
echo "4. Run the API: python -m uvicorn api.main:app --reload"
echo ""
