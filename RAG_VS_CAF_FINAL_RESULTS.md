# RAG vs CAF: Empirical Comparison Results

## Executive Summary

We compared **Pure RAG** (Retrieval-Augmented Generation) against **CAF** (Causal Autonomy Framework) on 10 CounterBench examples using the same LLM (TinyLlama-1.1B-4bit).

**Key Finding**: RAG outperformed CAF by 30 percentage points (60% vs 30% accuracy).

## Experimental Setup

### Common Configuration
- **LLM**: TinyLlama-1.1B-Chat-v1.0
- **Quantization**: 4-bit (fits in 4GB GPU)
- **Hardware**: NVIDIA GeForce GTX 1650 (3.9GB)
- **Dataset**: CounterBench (10 examples, basic reasoning)
- **Task**: Counterfactual causal reasoning (yes/no questions)

### System Architectures

#### Pure RAG Baseline
**Implementation**: [experiments/rag_baseline.py](experiments/rag_baseline.py)

**Pipeline**:
1. Extract causal facts from context (regex)
2. Retrieve relevant facts (keyword matching)
3. Augment LLM prompt with retrieved facts
4. **Single-shot generation** (1 iteration)
5. **NO verification** or iteration

**Characteristics**:
- Fast (1 iteration)
- Pure neural
- No symbolic reasoning

#### CAF (Neuro-Symbolic)
**Pipeline**:
1. Extract causal facts → RDF triples
2. Store in Apache Jena Fuseki (SPARQL endpoint)
3. LLM generates draft answer
4. **SPARQL verification** of extracted claims
5. **Iterative refinement** (up to 5 iterations)
6. Decision engine adjudicates final answer

**Characteristics**:
- Slower (avg 5 iterations)
- Neuro-symbolic hybrid
- Formal logic verification

## Results

### Performance Metrics

| Metric | RAG (Pure Neural) | CAF (Neuro-Symbolic) | Difference |
|--------|-------------------|----------------------|------------|
| **Accuracy** | **60.0%** | **30.0%** | **RAG +30%** |
| Correct / Total | 6 / 10 | 3 / 10 | - |
| Avg Iterations | 1.0 | 5.0 | RAG 5x faster |
| Verification Score | N/A | 0.00 | CAF verification failed |

### Answer Distribution

**RAG**:
- "no": 7 (70%)
- "yes": 3 (30%)

**CAF**:
- "no": 4 (40%)
- "yes": 6 (60%)

## Analysis

### Why RAG Outperformed CAF

1. **SPARQL Verification Too Strict**
   - CAF's real SPARQL verification scored 0.00 on all examples
   - Verification couldn't handle counterfactual reasoning ("if NOT X")
   - Without proper intervention calculus, SPARQL ASK queries fail

2. **TinyLlama's Counterfactual Limitation**
   - Small 1.1B model struggles with "if not X, would Y happen?" logic
   - Both systems suffer from this, but CAF's iteration doesn't help
   - Iteration requires LLM to understand the error—TinyLlama can't

3. **RAG's Simplicity Advantage**
   - Single-shot generation avoids compounding errors
   - No verification means no false negatives from strict logic
   - Retrieval provides enough context for simple reasoning

### Example Error Pattern

**Query**: "Would Lumbo occur if NOT Ziklo instead of Ziklo?"
**Context**: Ziklo → Blaf → Trune → Vork → Lumbo
**Expected**: "no" (breaking causal chain)

**RAG**: Correctly answers "no" (60% of the time)
**CAF**: Often answers "yes" (60% yes answers overall)

→ CAF's iterations reinforce incorrect reasoning rather than fixing it

## When Each Approach Works

### Use RAG When:
✓ Simple reasoning tasks
✓ Speed is critical
✓ Small LLM (< 3B parameters)
✓ No formal verification available
✓ Approximate answers acceptable

### Use CAF When:
✓ LLM capable of complex reasoning (7B+ models)
✓ Formal verification possible (not counterfactuals with basic SPARQL)
✓ Accuracy > Speed
✓ Interpretability required
✓ Domain has structured knowledge base

## Implications

### For CAF Development

1. **Need Better LLM**: TinyLlama insufficient for counterfactual reasoning
   - Try Phi-2 (2.7B) or Llama-2-7B
   - Or use better prompting techniques

2. **Need Better Verification**: SPARQL ASK queries inadequate
   - Implement intervention calculus (do-calculus)
   - Use probabilistic verification instead of Boolean

3. **Iteration Helps Only When**: LLM understands errors
   - Current setup: garbage in → garbage out (5x over)
   - Need smarter constraint injection

### For RAG Research

1. **RAG Works Well**: On simple causal reasoning
   - Retrieval + generation sufficient
   - No verification needed for basic tasks

2. **Scalability Question**: Will RAG maintain advantage on:
   - Longer causal chains?
   - More complex reasoning (nested, conditional)?
   - Larger datasets?

## Next Steps

### Immediate
1. ✓ Compare RAG vs CAF on same dataset ✓
2. ⏳ Test on larger LLM (Phi-2 or Llama-2-7B)
3. ⏳ Implement intervention calculus for CAF
4. ⏳ Scale to 100+ examples

### Research Questions
1. At what LLM size does CAF outperform RAG?
2. Does CAF help more on complex reasoning (nested/joint)?
3. Can hybrid approach (RAG retrieval + CAF verification) combine benefits?

## Conclusion

On this small-scale experiment with TinyLlama (1.1B):

**RAG wins**: 60% vs 30% accuracy, 5x faster

**Why**: Single-shot simplicity beats over-iterating with a weak LLM and strict verification

**But**: This doesn't invalidate CAF—it shows the importance of:
- Matching LLM capability to task complexity
- Verification that fits the reasoning type
- Knowing when symbolic reasoning helps vs hurts

**Future Work**: Test CAF with better LLM and proper counterfactual verification logic.

---

## Files

- **RAG Implementation**: [experiments/rag_baseline.py](experiments/rag_baseline.py)
- **CAF Implementation**: [experiments/run_counterbench_experiment.py](experiments/run_counterbench_experiment.py)
- **Comparison Script**: [scripts/compare_rag_caf.py](scripts/compare_rag_caf.py)
- **RAG Results**: [results/rag_pure/](results/rag_pure/)
- **CAF Results**: [results/counterbench_full/](results/counterbench_full/)

## Reproduction

```bash
# RAG baseline
python -m experiments.rag_baseline \
    --input data/counterbench.json \
    --limit 10 \
    --llm-model tiny --llm-4bit \
    --output results/rag_pure

# CAF full pipeline
python -m experiments.run_counterbench_experiment \
    --input data/counterbench.json \
    --limit 10 \
    --use-llm --llm-model tiny --llm-4bit \
    --use-real-sparql \
    --sparql-endpoint http://localhost:3030/counterbench/query \
    --extract-kb \
    --output results/counterbench_full

# Compare
python scripts/compare_rag_caf.py \
    --rag-dir results/rag_pure \
    --caf-dir results/counterbench_full
```
