#!/usr/bin/env python3
"""
Test script for Step B: Constraint Satisfaction

Demonstrates token-level KB grounding during LLM generation.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.inference_engine.engine import InferenceEngine, GenerationConfig
from modules.inference_engine.constrained_generation import ConstrainedInferenceEngine
from utils.config import get_settings
from loguru import logger


async def test_without_constraints():
    """Test standard generation without KB constraints"""
    print("\n" + "="*60)
    print("TEST 1: Standard Generation (No Constraints)")
    print("="*60)

    settings = get_settings()
    engine = InferenceEngine(
        model_name=settings.model_name,
        use_vllm=settings.use_vllm
    )

    config = GenerationConfig(
        max_tokens=100,
        temperature=0.7
    )

    result = await engine.generate(
        prompt="What does rain cause on roads?",
        config=config
    )

    print(f"\nPrompt: What does rain cause on roads?")
    print(f"\nGenerated Answer:")
    print(f"  {result['text']}")
    print(f"\nCausal Assertions:")
    for assertion in result.get('causal_assertions_raw', []):
        print(f"  - {assertion}")


async def test_with_constraints():
    """Test constrained generation with KB grounding"""
    print("\n" + "="*60)
    print("TEST 2: Constrained Generation (With KB Grounding)")
    print("="*60)

    settings = get_settings()

    # Initialize base engine
    base_engine = InferenceEngine(
        model_name=settings.model_name,
        use_vllm=False  # Use HF to enable logit processors
    )

    # Wrap with constraint satisfaction
    constrained_engine = ConstrainedInferenceEngine(
        base_engine=base_engine,
        fuseki_endpoint=settings.fuseki_endpoint,
        constraint_strength=15.0,  # Strong KB grounding
        enable_constraint_logging=True
    )

    config = GenerationConfig(
        max_tokens=100,
        temperature=0.7
    )

    result = await constrained_engine.generate_constrained(
        prompt="What does rain cause on roads?",
        config=config
    )

    print(f"\nPrompt: What does rain cause on roads?")
    print(f"\nGenerated Answer (KB-Grounded):")
    print(f"  {result['text']}")
    print(f"\nCausal Assertions:")
    for assertion in result.get('causal_assertions_raw', []):
        print(f"  - {assertion}")
    print(f"\nMetadata:")
    print(f"  Constraint Satisfaction: {result['metadata'].get('constraint_satisfaction')}")
    print(f"  Fuseki Endpoint: {result['metadata'].get('fuseki_endpoint')}")


async def test_contradiction_prevention():
    """Test that constraints prevent contradictions"""
    print("\n" + "="*60)
    print("TEST 3: Contradiction Prevention")
    print("="*60)
    print("Testing if KB prevents model from saying 'rain causes dry roads'")

    settings = get_settings()

    base_engine = InferenceEngine(
        model_name=settings.model_name,
        use_vllm=False
    )

    constrained_engine = ConstrainedInferenceEngine(
        base_engine=base_engine,
        fuseki_endpoint=settings.fuseki_endpoint,
        constraint_strength=20.0,  # Very strong KB enforcement
        enable_constraint_logging=True
    )

    config = GenerationConfig(
        max_tokens=80,
        temperature=0.3  # Lower temperature for more deterministic results
    )

    # Try to trick the model with a misleading prompt
    result = await constrained_engine.generate_constrained(
        prompt="If rain falls, what condition do roads have?",
        config=config
    )

    print(f"\nPrompt: If rain falls, what condition do roads have?")
    print(f"\nKB-Grounded Answer:")
    print(f"  {result['text']}")
    print(f"\nExpected: Model should say 'wet' not 'dry' due to KB constraint")


async def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*10 + "STEP B: CONSTRAINT SATISFACTION TEST" + " "*10 + "║")
    print("╚" + "═"*58 + "╝")

    try:
        # Test 1: Standard generation
        await test_without_constraints()

        # Test 2: Constrained generation
        await test_with_constraints()

        # Test 3: Contradiction prevention
        await test_contradiction_prevention()

        print("\n" + "="*60)
        print("✓ All tests completed")
        print("="*60)
        print("\nKEY INSIGHT:")
        print("Constrained generation queries Fuseki during token generation")
        print("and biases the model to prefer KB-aligned tokens.")
        print("This prevents hallucinations and ensures factual grounding.")
        print()

    except Exception as e:
        logger.exception("Test failed")
        print(f"\n✗ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
