# Intervention Calculus Integration - Complete Summary

## What We Built

I've integrated **Pearl's intervention calculus (do-calculus)** into CAF to properly handle counterfactual reasoning.

## The Problem

**Before** (Basic SPARQL):
- CAF accuracy: **30%** on CounterBench
- RAG accuracy: **60%** on CounterBench
- **Issue**: SPARQL can't handle counterfactual queries like "Would Y occur if NOT X?"

**Example Failure**:
```
Query: "Would Lumbo occur if NOT Ziklo instead of Ziklo?"
Context: Ziklo → Blaf → Trune → Vork → Lumbo

Basic SPARQL: ASK { <Ziklo> <causes> <Lumbo> }
Result: TRUE (via transitive closure)
Problem: This doesn't tell us what happens when we PREVENT Ziklo!
CAF Answer: "yes" ❌ (Wrong!)
Expected: "no"
```

## The Solution

**After** (Intervention Calculus):
```
Query: "Would Lumbo occur if NOT Ziklo instead of Ziklo?"

Intervention Calculus:
1. Parse: do(Ziklo=False) - prevent Ziklo
2. Graph Surgery: Remove incoming edges to Ziklo
3. Check Descendants: Lumbo depends on Ziklo
4. Answer: NO ✓ (Correct!)
```

## Files Created

### 1. Core Implementation
**[experiments/intervention_calculus.py](experiments/intervention_calculus.py)**
- `CausalGraph` class with graph surgery
- `intervene()` method for do-calculus
- `would_occur()` for counterfactual queries
- `parse_counterfactual_query()` to detect "if NOT X" patterns

### 2. Enhanced FVL
**[experiments/real_fvl_with_intervention.py](experiments/real_fvl_with_intervention.py)**
- Extends `RealFVL` with intervention support
- Auto-detects counterfactual vs factual queries
- Uses do-calculus for counterfactuals, SPARQL for facts
- Provides explanations of reasoning

### 3. Experiment Runner
**[experiments/run_counterbench_with_intervention.py](experiments/run_counterbench_with_intervention.py)**
- Modified CAF experiment runner
- Sets causal context for each query
- Tracks intervention vs SPARQL usage
- Compatible with existing evaluation framework

### 4. Documentation
- **[INTERVENTION_CALCULUS_GUIDE.md](INTERVENTION_CALCULUS_GUIDE.md)** - Complete theory + implementation
- **[INTERVENTION_INTEGRATION_SUMMARY.md](INTERVENTION_INTEGRATION_SUMMARY.md)** - This file

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CAF with Intervention                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query + Context                                            │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────┐                                       │
│  │ Query Classifier │                                       │
│  └──────────────────┘                                       │
│       │        │                                            │
│       │        │                                            │
│   Factual  Counterfactual                                   │
│       │        │                                            │
│       ▼        ▼                                            │
│  ┌────────┐  ┌────────────────────┐                        │
│  │ SPARQL │  │ Intervention       │                        │
│  │  ASK   │  │ Calculus           │                        │
│  └────────┘  │ (do-calculus)      │                        │
│              └────────────────────┘                        │
│                    │                                        │
│                    ▼                                        │
│              Verification Result                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Example Flow

**Input**:
```python
query = "Would Lumbo occur if not Ziklo instead of Ziklo?"
context = "Ziklo causes Blaf, Blaf causes Trune, Trune causes Vork, Vork causes Lumbo"
```

**Step 1**: Build Causal Graph
```python
fvl.set_causal_context(context)
# Graph: Ziklo → Blaf → Trune → Vork → Lumbo
```

**Step 2**: Set Query
```python
fvl.set_current_query(query)
# Detected: Counterfactual (contains "if not X instead of X")
```

**Step 3**: Verify with Intervention
```python
fvl.verify(triplets)
# 1. Parse: do(Ziklo=False)
# 2. Graph surgery: Remove Ziklo
# 3. Check: Is Lumbo descendant of Ziklo? YES
# 4. Result: Lumbo would NOT occur
# 5. Return: VerificationStatus.CONTRADICTION
```

**Output**: Answer "no" ✓

## Usage

### Basic Usage

```bash
cd /home/bright/projects/PhD/CAF

# Test on 3 examples
python -m experiments.run_counterbench_with_intervention \
    --input data/counterbench.json \
    --limit 3 \
    --use-llm --llm-model tiny --llm-4bit \
    --use-intervention \
    --output results/caf_intervention_test

# Full experiment (10 examples)
python -m experiments.run_counterbench_with_intervention \
    --input data/counterbench.json \
    --limit 10 \
    --use-llm --llm-model tiny --llm-4bit \
    --use-intervention \
    --output results/caf_intervention_full
```

### Programmatic Usage

```python
from experiments.real_fvl_with_intervention import RealFVLWithIntervention
from experiments.caf_algorithm import CAFLoop, CAFConfig
from experiments.llm_integration import create_llama_layer

# Create components
llm = create_llama_layer("tiny", use_4bit=True)
fvl = RealFVLWithIntervention()
caf = CAFLoop(llm, fvl, CAFConfig(max_iterations=5))

# Set context
fvl.set_causal_context("Ziklo causes Blaf, Blaf causes Trune...")
fvl.set_current_query("Would Lumbo occur if not Ziklo?")

# Run CAF
result = caf.execute(query, context)

# Get explanation
print(fvl.get_explanation())
```

## Expected Results

### Hypothesis

With intervention calculus, CAF should:
1. **Correctly answer counterfactuals** (do-calculus handles "if NOT X")
2. **Improve accuracy** from 30% to >60%
3. **Match or beat RAG** (60% accuracy)

### Comparison

| System | Accuracy | Method |
|--------|----------|--------|
| RAG | 60% | Retrieval + Single-shot LLM |
| CAF (Basic SPARQL) | 30% | SPARQL verification (broken for counterfactuals) |
| **CAF (Intervention)** | **?%** | **do-calculus + SPARQL** |

### Test In Progress

Currently running:
```
bash ID: 70a798
Command: run_counterbench_with_intervention (3 examples)
Status: Running
```

## Key Advantages

### 1. Proper Counterfactual Reasoning
- **do-calculus** models interventions correctly
- Graph surgery removes causal pathways
- Descendant analysis determines effects

### 2. Hybrid Approach
- Counterfactual queries → Intervention calculus
- Factual queries → SPARQL
- Best of both worlds

### 3. Explainability
```python
fvl.get_explanation()
```
Returns:
```
Counterfactual Reasoning via Intervention Calculus:

Query: Would Lumbo occur if NOT Ziklo?
Intervention: do(Ziklo=False)
Target: Lumbo

Causal Graph:
  Ziklo → Blaf
  Blaf → Trune
  Trune → Vork
  Vork → Lumbo

Intervention Effect:
- After do(Ziklo=False):
  Lumbo is a descendant of Ziklo → won't occur

Answer: No
```

### 4. No Additional Dependencies
- Uses existing causal graph from context
- No new external services needed
- Pure Python implementation

## Limitations

### 1. Still Requires Good LLM
- Intervention calculus doesn't fix LLM errors
- TinyLlama still struggles with complex reasoning
- Larger models (Phi-2, Llama-2-7B) would help

### 2. Context Parsing
- Relies on regex to extract causal relations
- May miss complex causal structures
- Could be improved with better NLP

### 3. Query Pattern Matching
- Detects "if not X instead of X" patterns
- May miss other counterfactual phrasings
- Could expand to more patterns

## Next Steps

1. ✅ Implement intervention calculus
2. ✅ Integrate into CAF
3. ⏳ Test on 3 examples (running)
4. ⏳ Run full 10-example experiment
5. ⏳ Compare CAF+intervention vs RAG vs CAF-basic
6. ⏳ Scale to 100+ examples
7. ⏳ Test with larger models (Phi-2, Llama-2-7B)

## Conclusion

Intervention calculus is the **missing piece** that makes CAF properly handle counterfactual reasoning.

**Before**: CAF relied on basic SPARQL → Failed on counterfactuals (30% accuracy)
**After**: CAF uses do-calculus → Should handle counterfactuals correctly (expected >60%)

This represents a **fundamental upgrade** from pattern-matching (RAG) to **causal reasoning** (CAF).

---

**Status**: Integration complete, testing in progress.
**ETA**: Results available in ~5-10 minutes.
