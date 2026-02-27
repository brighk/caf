#!/usr/bin/env python3
"""
Minimal test to debug intervention calculus integration bug.
"""
import sys
import traceback

# Test 1: Can we import the modules?
try:
    print("Test 1: Importing modules...")
    from experiments.real_fvl_with_intervention import RealFVLWithIntervention
    from experiments.llm_integration import create_llama_layer
    from experiments.caf_algorithm import CAFLoop, CAFConfig
    print("✓ All imports successful\n")
except Exception as e:
    print(f"✗ Import failed: {e}\n")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Can we create the FVL?
try:
    print("Test 2: Creating intervention FVL...")
    fvl = RealFVLWithIntervention(
        sparql_endpoint="http://localhost:3030/test/query",
        enable_fuzzy_match=False
    )
    print(f"✓ FVL created: {type(fvl)}\n")
except Exception as e:
    print(f"✗ FVL creation failed: {e}\n")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Can we create a tiny LLM?
try:
    print("Test 3: Creating LLM layer (this takes time)...")
    llm = create_llama_layer(
        model_name='tiny',
        use_4bit=True,
        max_iterations=5
    )
    print(f"✓ LLM created: {type(llm)}")
    print(f"  - Has max_iterations: {hasattr(llm, 'max_iterations')}\n")
except Exception as e:
    print(f"✗ LLM creation failed: {e}\n")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Can we create CAFConfig and CAFLoop?
try:
    print("Test 4: Creating CAF loop...")
    config = CAFConfig(max_iterations=5, verification_threshold=0.7)
    caf_loop = CAFLoop(llm, fvl, config)
    print(f"✓ CAF loop created: {type(caf_loop)}\n")
except Exception as e:
    print(f"✗ CAF loop creation failed: {e}\n")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Can we execute a simple query?
try:
    print("Test 5: Executing CAF loop...")
    query = "Would Lumbo occur if not Ziklo instead of Ziklo?"
    context = "Ziklo causes Lumbo."

    # Set intervention context
    fvl.set_causal_context(context)
    fvl.set_current_query(query)

    result = caf_loop.execute(query, context)
    print(f"✓ CAF execution completed")
    print(f"  - Result: {result.final_response}")
    print(f"  - Decision: {result.decision}")
    print(f"  - Iterations: {result.iterations_used}\n")
except Exception as e:
    print(f"✗ CAF execution failed: {e}\n")
    traceback.print_exc()
    sys.exit(1)

print("=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
