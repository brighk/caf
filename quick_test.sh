#!/bin/bash
# Quick CAF Test Script
# ======================
# Run this before deploying to GPU hardware

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           CAF Quick Test (4GB GPU Compatible)                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Basic check
echo -e "${BLUE}[1/4] Running basic dependency check...${NC}"
python tests/test_preflight_check.py --no-sparql

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Basic check failed! Fix errors before proceeding.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Basic check passed${NC}"
echo ""

# Step 2: Ask about GPU test
echo -e "${YELLOW}Do you want to test GPU/LLM loading? (requires 4GB+ GPU) [y/N]${NC}"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${BLUE}[2/4] Testing GPU and LLM loading (this may take 5-10 min)...${NC}"
    python tests/test_preflight_check.py --test-gpu --no-sparql

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ GPU test failed!${NC}"
        echo -e "${YELLOW}You can still run in simulation mode (no GPU)${NC}"
    else
        echo -e "${GREEN}✓ GPU test passed - you can use --use-llm flag${NC}"
    fi
else
    echo -e "${YELLOW}Skipping GPU test. You can run it later with:${NC}"
    echo "  python tests/test_preflight_check.py --test-gpu"
fi

echo ""

# Step 3: Ask about SPARQL test
echo -e "${YELLOW}Do you want to test SPARQL endpoint? (requires Fuseki running) [y/N]${NC}"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${BLUE}[3/4] Testing SPARQL endpoint...${NC}"
    python tests/test_preflight_check.py --sparql-only

    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}SPARQL test skipped or failed${NC}"
        echo -e "${YELLOW}Start Fuseki and load data, then run:${NC}"
        echo "  python tests/test_preflight_check.py --sparql-only"
    else
        echo -e "${GREEN}✓ SPARQL test passed - you can use --use-real-sparql flag${NC}"
    fi
else
    echo -e "${YELLOW}Skipping SPARQL test${NC}"
fi

echo ""

# Step 4: Run mini experiment
echo -e "${BLUE}[4/4] Running mini experiment (2 chains, simulation)...${NC}"
python -m experiments.run_experiment --num-chains 2 --quiet

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Mini experiment failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Mini experiment passed${NC}"
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                     Test Summary                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}✓ Your system is ready for CAF experiments!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Run small experiment (simulation, fast):"
echo -e "   ${BLUE}python -m experiments.run_experiment --num-chains 10${NC}"
echo ""
echo "2. Run with real LLM (requires GPU):"
echo -e "   ${BLUE}python -m experiments.run_experiment --use-llm --llm-4bit --num-chains 10${NC}"
echo ""
echo "3. Run full production (GPU + SPARQL):"
echo -e "   ${BLUE}python -m experiments.run_experiment --use-llm --llm-4bit --use-real-sparql --num-chains 75${NC}"
echo ""
echo "See TESTING_GUIDE.md for detailed testing procedures."
echo ""
