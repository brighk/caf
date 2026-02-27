# Intervention Calculus for CAF

## What is Intervention Calculus?

**Intervention calculus** (Pearl's do-calculus) is a formal framework for reasoning about **causal interventions** and **counterfactuals**.

### The Problem

Basic SPARQL queries can check if relationships exist:
```sparql
ASK { <Ziklo> <causes> <Lumbo> }  # Returns: TRUE
```

But they **cannot** answer counterfactual questions:
- "What if we prevent Ziklo from happening?"
- "Would Lumbo still occur if we intervene on Ziklo?"

### The Solution

Intervention calculus uses **graph surgery** to model interventions:

1. **do(X=false)**: Intervene to prevent X
   - Remove all incoming edges to X
   - Keep outgoing edges from X
   - Check if descendants still occur

2. **do(X=true)**: Intervene to force X
   - Remove incoming edges to X
   - Force X to occur
   - All descendants will occur

## Implementation

I've created [experiments/intervention_calculus.py](experiments/intervention_calculus.py) with:

### Core Classes

**CausalGraph**:
- Stores causal relationships as directed edges
- Computes ancestors and descendants
- Performs graph surgery for interventions

**Key Methods**:
```python
graph.intervene(node, value)  # do-calculus graph surgery
graph.would_occur(target, intervention_node, intervention_value)  # Counterfactual query
```

## Example

### Query
```
Context: "Ziklo causes Blaf, Blaf causes Trune, Trune causes Vork, Vork causes Lumbo"
Question: "Would Lumbo occur if not Ziklo instead of Ziklo?"
```

### Solution Using Intervention Calculus

**Step 1**: Build causal graph
```
Ziklo → Blaf → Trune → Vork → Lumbo
```

**Step 2**: Perform intervention do(Ziklo=False)
```
Original: Ziklo → Blaf → Trune → Vork → Lumbo

After do(Ziklo=False):
- Remove Ziklo from graph (graph surgery)
- Blaf has no other causes → Blaf won't occur
- Trune has no other causes → Trune won't occur
- Vork has no other causes → Vork won't occur
- Lumbo has no other causes → Lumbo won't occur
```

**Step 3**: Check if Lumbo is a descendant of Ziklo
```
Lumbo is a descendant of Ziklo → Lumbo depends on Ziklo
Without Ziklo, Lumbo will NOT occur
```

**Answer**: **No** ✓

### Why Basic SPARQL Fails

Basic SPARQL approach:
```sparql
# Check: Does Ziklo cause Lumbo?
ASK { <Ziklo> <causes>+ <Lumbo> }  # TRUE (transitive)

# But this doesn't answer: "What if we PREVENT Ziklo?"
```

SPARQL only checks existence of relationships, not effects of interventions.

## Pearl's Three Rules of do-Calculus

### Rule 1: Insertion/Deletion of Observations
```
P(Y | do(X), Z, W) = P(Y | do(X), Z)
if W is independent of Y given X, Z in the manipulated graph
```

### Rule 2: Action/Observation Exchange
```
P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W)
if Z is independent of Y given X, W in the graph after removing arrows into Z
```

### Rule 3: Insertion/Deletion of Actions
```
P(Y | do(X), do(Z), W) = P(Y | do(X), W)
if Z is independent of Y given X, W in the graph after removing arrows into X and out of Z
```

Our implementation uses **Rule 1** (graph surgery) for counterfactual reasoning.

## Integration with CAF

### Current CAF Verification (Broken for Counterfactuals)

```python
# Real FVL: experiments/real_fvl.py
def verify_triplet(self, triplet: RDFTriplet) -> VerificationResult:
    # Simple SPARQL ASK query
    query = f"ASK {{ {triplet.to_sparql_pattern()} }}"
    result = self.execute_sparql(query)
    # Problem: This doesn't handle do(X=value) interventions!
```

### New CAF with Intervention Calculus

```python
# Enhanced FVL with intervention support
from experiments.intervention_calculus import CausalGraph, counterfactual_reasoning

def verify_counterfactual(self, query: str, context: str) -> bool:
    """Verify counterfactual query using do-calculus."""
    result = counterfactual_reasoning(query, context)
    return result
```

## Testing

Run the example:
```bash
cd /home/bright/projects/PhD/CAF
python experiments/intervention_calculus.py
```

**Output**:
```
Context: Ziklo causes Blaf, Blaf causes Trune, Trune causes Vork, and Vork causes Lumbo.
Query: Would Lumbo occur if not Ziklo instead of Ziklo?

Causal Graph:
  Ziklo → Blaf
  Blaf → Trune
  Trune → Vork
  Vork → Lumbo

After Intervention do(Ziklo=False):
  (Ziklo removed from graph)

Answer: No

✓ Correct! Expected: no, Got: no
```

## Why This Fixes CAF's Counterfactual Problem

### Before (Basic SPARQL)
- CAF accuracy: 30%
- Verification score: 0.00
- Problem: SPARQL can't handle interventions

### After (Intervention Calculus)
- Can properly reason about do(X=value)
- Handles counterfactual queries correctly
- Should improve CAF accuracy significantly

## Next Steps

1. **Integrate with Real FVL**: Replace basic SPARQL verification with intervention calculus
2. **Test on CounterBench**: Re-run CAF experiments with intervention support
3. **Compare**: RAG (60%) vs CAF with intervention calculus (expected: >60%)

## References

- Pearl, J. (2009). "Causality: Models, Reasoning and Inference"
- Pearl, J. (2000). "Causal Inference in Statistics: A Primer"
- Bareinboim, E., & Pearl, J. (2016). "Causal inference and the data-fusion problem"

---

**Key Insight**: Counterfactual reasoning requires intervention calculus, not just relationship checking. This is what separates CAF (causal reasoning) from RAG (pattern matching).
