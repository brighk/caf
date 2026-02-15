# CAF Usage Examples

Quick examples for common CAF tasks.

## Real SPARQL Verification

### 1. Basic Setup

```bash
# Install dependencies
pip install SPARQLWrapper spacy fuzzywuzzy python-Levenshtein
python -m spacy download en_core_web_sm

# Start Fuseki with test data
./fuseki-server --mem /conceptnet &

# Load sample ConceptNet data
python scripts/convert_conceptnet_to_rdf.py \
    conceptnet-assertions-5.7.0.csv \
    --limit 10000 \
    --output conceptnet_10k.nt

curl -X POST -H "Content-Type: application/n-triples" \
    --data-binary @conceptnet_10k.nt \
    http://localhost:3030/conceptnet/data
```

### 2. Test Real FVL

```python
from experiments.real_fvl import RealFVL

# Initialize
fvl = RealFVL(sparql_endpoint="http://localhost:3030/conceptnet/query")

# Parse text
text = "Water causes erosion. Climate change leads to extreme weather."
triplets = fvl.parse(text)

print(f"Extracted {len(triplets)} triplets:")
for t in triplets:
    print(f"  ({t.subject}) --[{t.predicate}]--> ({t.obj})")

# Verify
results = fvl.verify(triplets)

print("\nVerification results:")
for triplet, result in zip(triplets, results):
    print(f"  {triplet.subject} -> {result.status.value} ({result.confidence_score:.2f})")

# Statistics
stats = fvl.get_stats()
print(f"\nCache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Avg query time: {stats['avg_query_time_ms']:.1f}ms")
```

### 3. Run Experiment with Real SPARQL

```bash
# With simulation (fast, no setup)
python -m experiments.run_experiment \
    --num-chains 25 \
    --output-dir results/simulated

# With real SPARQL (requires triplestore)
python -m experiments.run_experiment \
    --use-real-sparql \
    --sparql-endpoint http://localhost:3030/conceptnet/query \
    --num-chains 25 \
    --output-dir results/real_sparql

# With real LLM + real SPARQL (full production mode)
python -m experiments.run_experiment \
    --use-llm \
    --llm-4bit \
    --use-real-sparql \
    --num-chains 25 \
    --output-dir results/full_production
```

### 4. CAF Loop with Real Components

```python
from experiments.real_fvl import RealFVL
from experiments.llm_integration import create_llama_layer
from experiments.caf_algorithm import CAFLoop, CAFConfig

# Initialize real components
fvl = RealFVL(sparql_endpoint="http://localhost:3030/conceptnet/query")
llm = create_llama_layer(model_size="7b", use_4bit=True)

# Create CAF loop
caf = CAFLoop(
    config=CAFConfig(
        max_iterations=5,
        verification_threshold=0.8
    ),
    inference_layer=llm,
    verification_layer=fvl
)

# Execute
result = caf.execute("What are the causes of climate change?")

print(f"Response: {result.final_response}")
print(f"Score: {result.final_score:.2f}")
print(f"Iterations: {result.iterations_used}")
print(f"Decision: {result.decision.value}")
```

## ConceptNet Data Processing

### Convert Full Dataset

```bash
# English only, high-quality relations
python scripts/convert_conceptnet_to_rdf.py \
    conceptnet-assertions-5.7.0.csv \
    --output conceptnet_en_full.nt \
    --language en \
    --min-weight 2.0 \
    --verbose

# Causal relations only
python scripts/convert_conceptnet_to_rdf.py \
    conceptnet-assertions-5.7.0.csv \
    --output conceptnet_causal.nt \
    --language en \
    --relations "Causes,HasSubevent,HasPrerequisite" \
    --min-weight 3.0 \
    --include-weights
```

### Load into Fuseki

```bash
# HTTP POST (< 100K triples)
curl -X POST \
    -H "Content-Type: application/n-triples" \
    --data-binary @conceptnet_10k.nt \
    http://localhost:3030/conceptnet/data

# tdbloader (> 1M triples)
./tdb2.tdbloader --loc=../databases/conceptnet conceptnet_en_full.nt
```

## Testing

### Unit Tests

```bash
# Test parsing
python -c "
from experiments.real_fvl import RealFVL
fvl = RealFVL()
triplets = fvl.parse('Dogs are animals.')
print(f'Parsed {len(triplets)} triplets')
"

# Test SPARQL connection
curl http://localhost:3030/$/ping
```

### Integration Tests

```bash
# Run all tests
pytest tests/test_real_fvl_integration.py -v

# Run specific test
pytest tests/test_real_fvl_integration.py::test_parsing -v

# With coverage
pytest tests/test_real_fvl_integration.py --cov=experiments.real_fvl
```

## Performance Optimization

### Increase Cache Size

```python
fvl = RealFVL(
    cache_size=10000,  # Default: 1000
    fuzzy_match_limit=20  # Search more candidates
)
```

### Pre-warm Cache

```python
# Load common entities
common = ["water", "air", "fire", "earth", "climate", "temperature"]
for entity in common:
    fvl._link_entity(entity)
```

### Tune Fuzzy Matching

```python
# Strict matching (faster, less recall)
fvl = RealFVL(entity_threshold=0.9)

# Loose matching (slower, higher recall)
fvl = RealFVL(entity_threshold=0.6)

# Disable fuzzy matching
fvl = RealFVL(enable_fuzzy_match=False)
```

## Troubleshooting

### Connection Errors

```python
# Test endpoint
from experiments.real_fvl import RealFVL

fvl = RealFVL()
result = fvl._execute_sparql_query("ASK { ?s ?p ?o }")

if result.success:
    print("✓ Connected to SPARQL endpoint")
else:
    print(f"✗ Connection failed: {result.error}")
```

### Check Data Loaded

```bash
# Count triples
curl -X POST \
    -H "Content-Type: application/sparql-query" \
    --data "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }" \
    http://localhost:3030/conceptnet/query

# Sample data
curl -X POST \
    -H "Content-Type: application/sparql-query" \
    --data "SELECT * WHERE { ?s ?p ?o } LIMIT 5" \
    http://localhost:3030/conceptnet/query
```

### Debugging Verification

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

fvl = RealFVL()
triplets = fvl.parse("Water causes erosion.")
results = fvl.verify(triplets)

# Check statistics
print(fvl.get_stats())
```

## Production Deployment

### Persistent Triplestore

```bash
# Start with persistent storage
./fuseki-server --loc=../databases/conceptnet /conceptnet

# Load full dataset
./tdb2.tdbloader --loc=../databases/conceptnet conceptnet_en_full.nt

# Build statistics for faster queries
./tdb2.tdbstats --loc=../databases/conceptnet > stats.opt
```

### Resource Limits

```bash
# Increase JVM heap
export JVM_ARGS="-Xmx8g -Xms4g"
./fuseki-server --loc=../databases/conceptnet /conceptnet
```

### Monitoring

```python
# Track FVL statistics
stats = fvl.get_stats()

metrics = {
    "total_queries": stats["queries_executed"],
    "avg_latency_ms": stats["avg_query_time_ms"],
    "cache_hit_rate": stats["cache_hit_rate"],
    "exact_matches": stats["exact_matches"],
    "fuzzy_matches": stats["fuzzy_matches"],
    "failed_links": stats["failed_links"]
}

# Log to monitoring system
print(json.dumps(metrics))
```

## See Also

- [REAL_SPARQL_SETUP.md](REAL_SPARQL_SETUP.md) - Detailed setup guide
- [REAL_SPARQL_MIGRATION.md](REAL_SPARQL_MIGRATION.md) - Migration from simulation
- [README.md](README.md) - Project overview
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
