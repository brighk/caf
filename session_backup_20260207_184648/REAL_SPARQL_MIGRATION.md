# Migration Guide: Simulation → Real SPARQL

This document outlines what needs to change to replace the simulated FVL with real SPARQL/RDF verification.

## Cost Summary

| Component | Time Estimate | Difficulty | Dependencies |
|-----------|---------------|------------|--------------|
| **Infrastructure Setup** | 1-2 hours | Easy | Jena/GraphDB install |
| **Load Knowledge Base** | 30 min - 2 hours | Easy | Download ConceptNet |
| **Code Changes** | 4-8 hours | Medium | Python, SPARQL |
| **Testing & Debugging** | 4-8 hours | Medium | SPARQL queries |
| **Dataset Adaptation** | 2-4 hours | Medium | Replace synthetic data |
| **TOTAL** | **~12-24 hours** | **Medium** | |

## Detailed Changes Required

### 1. Install Dependencies

```bash
# Add to requirements.txt
pip install rdflib SPARQLWrapper spacy fuzzywuzzy python-Levenshtein

# Download spaCy model for NER
python -m spacy download en_core_web_sm
```

### 2. Infrastructure Setup

#### Option A: Apache Jena Fuseki (Recommended for testing)

```bash
# Download and start Jena
wget https://archive.apache.org/dist/jena/binaries/apache-jena-fuseki-4.10.0.tar.gz
tar -xzf apache-jena-fuseki-4.10.0.tar.gz
cd apache-jena-fuseki-4.10.0

# Start server (in-memory dataset)
./fuseki-server --update --mem /conceptnet

# Or with persistent storage
./fuseki-server --update --loc=../data/conceptnet /conceptnet
```

**Endpoint:** `http://localhost:3030/conceptnet/query`

#### Option B: GraphDB (Better for production)

```bash
# Download from https://graphdb.ontotext.com/
# Free edition supports 200M triples

# Start GraphDB
./graphdb -Dgraphdb.home=/path/to/data

# Access at http://localhost:7200
# Create repository via web UI
```

### 3. Load ConceptNet Data

```bash
# Download ConceptNet assertions
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
gunzip conceptnet-assertions-5.7.0.csv.gz

# Convert to RDF (N-Triples format)
python convert_conceptnet_to_rdf.py conceptnet-assertions-5.7.0.csv > conceptnet.nt

# Load into Jena via CLI
./tdb2.tdbloader --loc=../data/conceptnet conceptnet.nt

# Or load via SPARQL Update (slower)
curl -X POST \
  -H "Content-Type: application/n-triples" \
  --data-binary @conceptnet.nt \
  http://localhost:3030/conceptnet/data
```

**ConceptNet → RDF Conversion Script:**
```python
import csv
import hashlib

def uri(text):
    """Create URI from text."""
    return f"http://conceptnet.io/c/en/{text.replace(' ', '_')}"

with open('conceptnet-assertions-5.7.0.csv') as f:
    reader = csv.reader(f, delimiter='\t')

    for row in reader:
        rel = row[1]  # /r/RelatedTo
        subj = row[2]  # /c/en/dog
        obj = row[3]   # /c/en/animal

        # Extract English entities
        if '/c/en/' not in subj or '/c/en/' not in obj:
            continue

        s = subj.split('/c/en/')[-1]
        o = obj.split('/c/en/')[-1]
        r = rel.split('/r/')[-1]

        # Output N-Triple
        print(f"<{uri(s)}> <http://conceptnet.io/r/{r}> <{uri(o)}> .")
```

### 4. Code Changes

#### A. Replace `SimulatedFVL` in `caf_algorithm.py`

**Current:**
```python
from experiments.caf_algorithm import SimulatedFVL

fvl = SimulatedFVL(kb_coverage=0.8)
```

**New:**
```python
from experiments.real_fvl import RealFVL

fvl = RealFVL(
    sparql_endpoint="http://localhost:3030/conceptnet/query",
    entity_threshold=0.7,
    enable_fuzzy_match=True
)
```

#### B. Update `run_experiment.py`

**Lines 76-84 (CAFLoop initialization):**

```python
# OLD - Simulation
self.caf = CAFLoop(
    config=CAFConfig(
        max_iterations=5,
        verification_threshold=0.8,
    ),
    inference_layer=inference_layer  # Simulated or real LLM
)

# NEW - Real SPARQL
from experiments.real_fvl import RealFVL

real_fvl = RealFVL(
    sparql_endpoint="http://localhost:3030/conceptnet/query",
    entity_threshold=0.7
)

self.caf = CAFLoop(
    config=CAFConfig(
        max_iterations=5,
        verification_threshold=0.8,
    ),
    inference_layer=inference_layer,
    fvl=real_fvl  # Real FVL instead of simulated
)
```

#### C. Update `CAFLoop` constructor in `caf_algorithm.py`

**Add optional FVL parameter:**

```python
class CAFLoop:
    def __init__(
        self,
        config: CAFConfig,
        inference_layer: Optional[InferenceLayer] = None,
        fvl: Optional[FormalVerificationLayer] = None  # NEW
    ):
        self.config = config
        self.il = inference_layer or SimulatedInferenceLayer()
        self.fvl = fvl or SimulatedFVL()  # Use provided or default
        self.de = SimulatedDecisionEngine()
```

### 5. Dataset Changes

**Current:** Synthetic causal chains

**New:** Real-world evaluation datasets

```python
# Option 1: Use existing NLI datasets
from datasets import load_dataset

# FEVER (Fact Extraction and VERification)
fever = load_dataset("fever", "v1.0")

# HotpotQA (Multi-hop reasoning)
hotpot = load_dataset("hotpot_qa", "distractor")

# Convert to prompts for CAF evaluation
def fever_to_prompt(example):
    claim = example['claim']
    return f"Verify the following claim: {claim}"

# Option 2: Manual curation
causal_queries = [
    "Explain how deforestation leads to climate change.",
    "What causes economic recession?",
    "How does vaccination prevent disease spread?"
]
```

### 6. Evaluation Metrics Updates

**No changes needed** - metrics.py works the same, just gets real verification results instead of simulated ones.

### 7. Testing Strategy

```python
# Step 1: Test triplestore connectivity
def test_triplestore():
    from SPARQLWrapper import SPARQLWrapper, JSON

    sparql = SPARQLWrapper("http://localhost:3030/conceptnet/query")
    sparql.setQuery("SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }")
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    count = results["results"]["bindings"][0]["count"]["value"]
    print(f"Triplestore has {count} triples")

# Step 2: Test entity linking
def test_entity_linking():
    from experiments.real_fvl import RealFVL

    fvl = RealFVL(sparql_endpoint="http://localhost:3030/conceptnet/query")

    uri = fvl._link_entity("dog")
    assert uri is not None, "Failed to link 'dog'"
    print(f"Linked 'dog' to {uri}")

# Step 3: Test full pipeline
def test_verification():
    fvl = RealFVL(sparql_endpoint="http://localhost:3030/conceptnet/query")

    triplets = [
        RDFTriplet(subject="dog", predicate="IsA", object="animal")
    ]

    results = fvl.verify(triplets)
    assert results[0].status == VerificationStatus.VERIFIED
    print("Verification successful!")

# Run tests
test_triplestore()
test_entity_linking()
test_verification()
```

## Performance Considerations

### Latency
- **Simulated FVL:** ~1-5ms per triplet (random number generation)
- **Real SPARQL:** ~50-200ms per triplet (depends on KB size, indexing)
- **Mitigation:**
  - Cache entity URI mappings
  - Batch SPARQL queries
  - Use SPARQL UNION for parallel verification

### Memory
- **ConceptNet:** ~300MB CSV → ~2GB in triplestore
- **Wikidata subset:** ~10-50GB (depends on subset)
- **Recommendation:** Start with ConceptNet, expand as needed

### Scalability
```python
# Optimize with batching
def verify_batch(self, triplets: List[RDFTriplet]) -> List[VerificationResult]:
    """Verify multiple triplets in single SPARQL query."""

    # Build UNION query for parallel verification
    query = "ASK { "
    for i, t in enumerate(triplets):
        subj_uri = self._link_entity(t.subject)
        obj_uri = self._link_entity(t.object)

        if i > 0:
            query += " UNION "
        query += f"{{ <{subj_uri}> ?p{i} <{obj_uri}> }}"
    query += " }"

    # Single query for all triplets
    return self._execute_batch_verification(query, triplets)
```

## Migration Checklist

- [ ] Install Apache Jena Fuseki / GraphDB
- [ ] Download ConceptNet dataset
- [ ] Convert ConceptNet to RDF format
- [ ] Load data into triplestore
- [ ] Install Python dependencies (rdflib, SPARQLWrapper, spacy)
- [ ] Implement `RealFVL` class (see `real_fvl_example.py`)
- [ ] Update `CAFLoop` to accept custom FVL
- [ ] Update `run_experiment.py` to use `RealFVL`
- [ ] Test triplestore connectivity
- [ ] Test entity linking
- [ ] Test verification pipeline
- [ ] Replace synthetic dataset with real evaluation data
- [ ] Run experiments and compare with simulation results
- [ ] Update paper to reflect real-world evaluation

## Expected Results

After migration, you'll have:
- ✅ **Real SPARQL verification** against ConceptNet
- ✅ **Entity linking** using NLP (spaCy)
- ✅ **Production-ready** architecture
- ✅ **Publishable results** on real datasets
- ✅ **Academic integrity** - no simulation disclaimer needed

## Cost-Benefit Analysis

| Aspect | Simulation | Real SPARQL |
|--------|-----------|-------------|
| **Setup time** | 0 hours | 12-24 hours |
| **Maintenance** | None | Triplestore management |
| **Reproducibility** | Perfect | Good (KB version-dependent) |
| **Academic credibility** | Requires disclaimer | Full credibility |
| **Real-world applicability** | Limited | Production-ready |
| **Debugging ease** | Easy | Moderate |
| **Performance** | Fast (~1ms) | Slower (~100ms) |

**Recommendation:** If publishing in a top-tier venue (NeurIPS, ICML, ACL), the 12-24 hour investment in real SPARQL is worth it for academic credibility.

## Quick Start (Fast Path)

```bash
# 1. Install Jena (5 min)
wget https://archive.apache.org/dist/jena/binaries/apache-jena-fuseki-4.10.0.tar.gz
tar -xzf apache-jena-fuseki-4.10.0.tar.gz
cd apache-jena-fuseki-4.10.0
./fuseki-server --update --mem /conceptnet &

# 2. Install dependencies (2 min)
pip install rdflib SPARQLWrapper spacy
python -m spacy download en_core_web_sm

# 3. Load minimal ConceptNet subset (10 min)
# Use first 10K triples for testing
head -10000 conceptnet-assertions-5.7.0.csv | python convert_to_rdf.py | \
  curl -X POST -H "Content-Type: application/n-triples" \
  --data-binary @- http://localhost:3030/conceptnet/data

# 4. Update experiment code (30 min)
# - Copy real_fvl_example.py to experiments/real_fvl.py
# - Update run_experiment.py CAFLoop initialization
# - Add --use-real-sparql flag

# 5. Run test experiment (5 min)
python -m experiments.run_experiment \
  --use-llm \
  --llm-4bit \
  --use-real-sparql \
  --num-chains 5

# Total: ~1 hour for proof-of-concept
```
