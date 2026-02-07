#!/usr/bin/env python3
"""
Quick test script for LLM integration.
Tests that Llama loads correctly and can generate responses.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from experiments.llm_integration import create_llama_layer


def test_basic_generation():
    """Test basic LLM generation."""
    print("=" * 70)
    print("CAF LLM Integration Test")
    print("=" * 70)

    # Create LLM with 4-bit quantization for efficiency
    print("\n[1/3] Loading Llama model (this may take 1-2 minutes)...")
    llm = create_llama_layer(
        model_size="7b",
        use_4bit=True,  # Use 4-bit to fit on A40
        open_source=True  # Use NousResearch/Llama-2-7b-chat-hf (no gating)
    )

    # Test 1: Simple generation
    print("\n[2/3] Testing basic generation...")
    print("-" * 70)
    prompt1 = "What causes rain to form? Explain in 2-3 sentences."
    print(f"Prompt: {prompt1}\n")
    response1 = llm.generate(prompt1)
    print(f"Response: {response1}")

    # Test 2: Generation with constraints
    print("\n[3/3] Testing constraint injection...")
    print("-" * 70)
    prompt2 = "Explain photosynthesis."
    constraints = [
        "Include the chemical equation CO2 + H2O → C6H12O6 + O2",
        "Mention chlorophyll",
        "Keep it under 4 sentences"
    ]
    print(f"Prompt: {prompt2}")
    print(f"Constraints: {constraints}\n")
    response2 = llm.generate(prompt2, constraints)
    print(f"Response: {response2}")

    print("\n" + "=" * 70)
    print("✓ LLM Integration Test Passed!")
    print("=" * 70)
    print("\nYou can now run the full experiment with:")
    print("  python -m experiments.run_experiment --use-llm --llm-4bit --num-chains 10")
    print("\nFor a quick test with fewer chains:")
    print("  python -m experiments.run_experiment --use-llm --llm-4bit --num-chains 5 --perturbations 2")


if __name__ == "__main__":
    test_basic_generation()
