# Real SPARQL Integration Setup Guide

Complete guide for setting up CAF with real SPARQL verification using Apache Jena Fuseki and ConceptNet.

## Quick Start (30 minutes)

```bash
# 1. Download and start Jena Fuseki
wget https://archive.apache.org/dist/jena/binaries/apache-jena-fuseki-4.10.0.tar.gz
tar -xzf apache-jena-fuseki-4.10.0.tar.gz
cd apache-jena-fuseki-4.10.0
./fuseki-server --update --mem /conceptnet &

# 2. Install Python dependencies
pip install SPARQLWrapper spacy fuzzywuzzy python-Levenshtein
python -m spacy download en_core_web_sm

# 3. Download and convert ConceptNet (sample)
cd ~/data
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
gunzip conceptnet-assertions-5.7.0.csv.gz

# Convert first 10K triples for testing
python scripts/convert_conceptnet_to_rdf.py \
    conceptnet-assertions-5.7.0.csv \
    --limit 10000 \
    --language en \
    --verbose \
    > conceptnet_10k.nt

# 4. Load into Fuseki
curl -X POST \
    -H "Content-Type: application/n-triples" \
    --data-binary @conceptnet_10k.nt \
    http://localhost:3030/conceptnet/data

# 5. Test Real FVL
python -m experiments.real_fvl
```

## Detailed Setup

### 1. Install Apache Jena Fuseki

#### Download and Extract

```bash
# Create data directory
mkdir -p ~/caf-data/fuseki
cd ~/caf-data/fuseki

# Download Jena Fuseki (latest stable)
wget https://archive.apache.org/dist/jena/binaries/apache-jena-fuseki-4.10.0.tar.gz
tar -xzf apache-jena-fuseki-4.10.0.tar.gz
cd apache-jena-fuseki-4.10.0
```

#### Start Fuseki Server

**Option A: In-memory (for testing)**
```bash
# Start with in-memory dataset
./fuseki-server --update --mem /conceptnet

# Server will run on http://localhost:3030
# Dataset endpoint: http://localhost:3030/conceptnet/query
```

**Option B: Persistent storage (for production)**
```bash
# Create persistent database
mkdir -p ../databases/conceptnet

# Start with persistent storage
./fuseki-server --update --loc=../databases/conceptnet /conceptnet

# Data persists across restarts
```

**Option C: With configuration file**
```bash
# Create config file: config.ttl
cat > config.ttl << 'EOF'
@prefix fuseki:  <http://jena.apache.org/fuseki#> .
@prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix tdb2:    <http://jena.apache.org/2016/tdb#> .
@prefix ja:      <http://jena.hpl.hp.com/2005/11/Assembler#> .

:service rdf:type fuseki:Service ;
    fuseki:name "conceptnet" ;
    fuseki:endpoint [ fuseki:operation fuseki:query ] ;
    fuseki:endpoint [ fuseki:operation fuseki:update ] ;
    fuseki:dataset :dataset .

:dataset rdf:type tdb2:DatasetTDB2 ;
    tdb2:location "../databases/conceptnet" ;
    tdb2:unionDefaultGraph true .
EOF

# Start with config
./fuseki-server --config=config.ttl
```

#### Verify Fuseki is Running

```bash
# Check server status
curl http://localhost:3030/$/ping

# List datasets
curl http://localhost:3030/$/datasets

# Access web UI
open http://localhost:3030
```

### 2. Install Python Dependencies

```bash
# Core dependencies
pip install SPARQLWrapper>=2.0.0
pip install rdflib>=7.0.0

# NLP for entity linking
pip install spacy>=3.7.0
python -m spacy download en_core_web_sm

# Optional: fuzzy matching (faster than difflib)
pip install fuzzywuzzy>=0.18.0
pip install python-Levenshtein>=0.12.0

# Optional: progress bars
pip install tqdm
```

### 3. Download ConceptNet Dataset

```bash
# Create data directory
mkdir -p ~/caf-data/conceptnet
cd ~/caf-data/conceptnet

# Download ConceptNet 5.7 (full dataset ~1.5GB)
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz

# Uncompress
gunzip conceptnet-assertions-5.7.0.csv.gz

# File info
wc -l conceptnet-assertions-5.7.0.csv
# ~34 million assertions
```

### 4. Convert ConceptNet to RDF

#### Small Test Dataset (10K triples)

```bash
python scripts/convert_conceptnet_to_rdf.py \
    conceptnet-assertions-5.7.0.csv \
    --output conceptnet_10k.nt \
    --limit 10000 \
    --language en \
    --min-weight 1.0 \
    --verbose
```

#### Medium Dataset (100K triples, common relations only)

```bash
python scripts/convert_conceptnet_to_rdf.py \
    conceptnet-assertions-5.7.0.csv \
    --output conceptnet_100k.nt \
    --limit 100000 \
    --language en \
    --relations "IsA,PartOf,Causes,RelatedTo,HasA,UsedFor" \
    --min-weight 2.0 \
    --verbose
```

#### Full English Dataset (~8M triples)

```bash
python scripts/convert_conceptnet_to_rdf.py \
    conceptnet-assertions-5.7.0.csv \
    --output conceptnet_en_full.nt \
    --language en \
    --min-weight 1.0 \
    --verbose

# This will take ~10-20 minutes
```

#### Custom Filtering

```bash
# High-quality causal relations only
python scripts/convert_conceptnet_to_rdf.py \
    conceptnet-assertions-5.7.0.csv \
    --output conceptnet_causal.nt \
    --language en \
    --relations "Causes,HasSubevent,HasPrerequisite,CausesDesire" \
    --min-weight 3.0 \
    --include-weights \
    --verbose
```

### 5. Load RDF Data into Fuseki

#### Using HTTP POST (recommended for < 100K triples)

```bash
# Load N-Triples file
curl -X POST \
    -H "Content-Type: application/n-triples" \
    --data-binary @conceptnet_10k.nt \
    http://localhost:3030/conceptnet/data

# Check triple count
curl -X POST \
    -H "Content-Type: application/sparql-query" \
    --data "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }" \
    http://localhost:3030/conceptnet/query
```

#### Using tdbloader (faster for > 1M triples)

```bash
# Stop Fuseki first
# Then load directly into TDB2 database

cd apache-jena-fuseki-4.10.0

# Load data
./tdb2.tdbloader --loc=../databases/conceptnet conceptnet_en_full.nt

# Restart Fuseki
./fuseki-server --loc=../databases/conceptnet /conceptnet &
```

#### Using Fuseki Web UI

1. Open http://localhost:3030
2. Click "manage datasets"
3. Click "add data" for /conceptnet
4. Upload .nt file
5. Select "N-Triples" format
6. Click "upload"

### 6. Verify Data is Loaded

```bash
# Count triples
curl -X POST \
    -H "Content-Type: application/sparql-query" \
    --data "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }" \
    http://localhost:3030/conceptnet/query | jq

# Sample triples
curl -X POST \
    -H "Content-Type: application/sparql-query" \
    --data "SELECT * WHERE { ?s ?p ?o } LIMIT 10" \
    http://localhost:3030/conceptnet/query | jq

# Test specific query
curl -X POST \
    -H "Content-Type: application/sparql-query" \
    --data "ASK { ?s <http://conceptnet.io/r/IsA> ?o }" \
    http://localhost:3030/conceptnet/query | jq
```

## Using Real FVL in CAF

### 1. Basic Usage

```python
from experiments.real_fvl import RealFVL

# Initialize with Fuseki endpoint
fvl = RealFVL(
    sparql_endpoint="http://localhost:3030/conceptnet/query",
    entity_threshold=0.7,
    enable_fuzzy_match=True
)

# Parse text into triplets
text = "Water causes erosion. Erosion leads to soil degradation."
triplets = fvl.parse(text)

# Verify against knowledge base
results = fvl.verify(triplets)

# Check results
for triplet, result in zip(triplets, results):
    print(f"{triplet}: {result.status.value} (confidence: {result.confidence_score:.2f})")

# Get statistics
stats = fvl.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Avg query time: {stats['avg_query_time_ms']:.1f}ms")
```

### 2. Integration with CAF Loop

```python
from experiments.caf_algorithm import CAFLoop, CAFConfig
from experiments.real_fvl import RealFVL
from experiments.llm_integration import RealInferenceLayer

# Initialize components
fvl = RealFVL(sparql_endpoint="http://localhost:3030/conceptnet/query")
inference_layer = RealInferenceLayer(model="meta-llama/Llama-2-7b-chat-hf")

# Create CAF loop with Real FVL
caf = CAFLoop(
    config=CAFConfig(
        max_iterations=5,
        verification_threshold=0.8
    ),
    inference_layer=inference_layer,
    verification_layer=fvl  # Use Real FVL instead of simulated
)

# Execute verification loop
result = caf.execute("What causes climate change?")

print(f"Final response: {result.final_response}")
print(f"Verification score: {result.final_score:.2f}")
print(f"Iterations: {result.iterations_used}")
```

### 3. Running Experiments with Real SPARQL

```bash
# Update run_experiment.py to accept --use-real-sparql flag

python -m experiments.run_experiment \
    --use-llm \
    --llm-4bit \
    --use-real-sparql \
    --sparql-endpoint http://localhost:3030/conceptnet/query \
    --num-chains 25 \
    --output-dir results/real_sparql
```

## Performance Optimization

### 1. Enable Query Result Caching

Fuseki caches query results automatically, but you can tune it:

```bash
# Start with larger cache
./fuseki-server --mem /conceptnet \
    --set tdb:query-cache-size=10000
```

### 2. Use Indexes

For persistent datasets, build indexes for faster queries:

```bash
# After loading data
./tdb2.tdbstats --loc=../databases/conceptnet --graph urn:x-arq:UnionGraph > stats.opt

# Restart Fuseki to use stats
```

### 3. Optimize Entity Linking

```python
# Increase cache size
fvl = RealFVL(
    sparql_endpoint="http://localhost:3030/conceptnet/query",
    cache_size=10000,  # Default: 1000
    fuzzy_match_limit=20  # Search more candidates
)

# Pre-warm cache with common entities
common_entities = ["water", "air", "fire", "earth", ...]
for entity in common_entities:
    fvl._link_entity(entity)
```

### 4. Batch Queries

For large-scale experiments, batch entity linking:

```python
# Instead of linking entities one by one, batch them
def batch_link_entities(fvl, entities):
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?entity ?uri WHERE {{
        VALUES ?label {{ {' '.join([f'"{e}"' for e in entities])} }}
        ?uri rdfs:label ?label .
        BIND(?label as ?entity)
    }}
    """
    # Execute and cache results
    ...
```

## Troubleshooting

### Issue: "Connection refused" when querying Fuseki

**Solution:**
```bash
# Check if Fuseki is running
ps aux | grep fuseki

# Check port
lsof -i :3030

# Restart Fuseki
./fuseki-server --update --mem /conceptnet
```

### Issue: "Dataset not found: /conceptnet"

**Solution:**
```bash
# List datasets
curl http://localhost:3030/$/datasets

# Create dataset via API
curl -X POST http://localhost:3030/$/datasets \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "dbName=conceptnet&dbType=mem"
```

### Issue: Slow query performance

**Solution:**
```bash
# Use persistent TDB2 instead of in-memory
./fuseki-server --update --loc=../databases/conceptnet /conceptnet

# Build statistics
./tdb2.tdbstats --loc=../databases/conceptnet > stats.opt

# Increase JVM heap size
export JVM_ARGS="-Xmx8g -Xms4g"
./fuseki-server --loc=../databases/conceptnet /conceptnet
```

### Issue: spaCy model not found

**Solution:**
```bash
# Download model
python -m spacy download en_core_web_sm

# Or use larger model for better accuracy
python -m spacy download en_core_web_lg

# Update RealFVL initialization
fvl = RealFVL(spacy_model="en_core_web_lg")
```

### Issue: Out of memory when loading large dataset

**Solution:**
```bash
# Use tdbloader instead of HTTP POST
./tdb2.tdbloader --loc=../databases/conceptnet conceptnet_en_full.nt

# Or load in chunks
split -l 1000000 conceptnet_en_full.nt chunk_
for chunk in chunk_*; do
    curl -X POST -H "Content-Type: application/n-triples" \
        --data-binary @$chunk http://localhost:3030/conceptnet/data
done
```

## Testing Real FVL

### Unit Tests

```python
# tests/test_real_fvl.py
import pytest
from experiments.real_fvl import RealFVL

@pytest.fixture
def fvl():
    return RealFVL(sparql_endpoint="http://localhost:3030/conceptnet/query")

def test_parsing(fvl):
    text = "Dogs are animals. Cats are mammals."
    triplets = fvl.parse(text)
    assert len(triplets) > 0
    assert any(t.predicate == "be" for t in triplets)

def test_entity_linking(fvl):
    uri = fvl._link_entity("dog")
    assert uri is not None
    assert "dog" in uri.lower()

def test_verification(fvl):
    triplets = [RDFTriplet(subject="dog", predicate="IsA", obj="animal")]
    results = fvl.verify(triplets)
    assert len(results) == 1
    # Status depends on KB content
    assert results[0].status in [VerificationStatus.VERIFIED, VerificationStatus.FAILED]
```

### Integration Test

```bash
# Start test Fuseki instance
./fuseki-server --mem /test &

# Load test data
python scripts/convert_conceptnet_to_rdf.py \
    conceptnet-assertions-5.7.0.csv \
    --limit 1000 \
    --output test_data.nt

curl -X POST -H "Content-Type: application/n-triples" \
    --data-binary @test_data.nt \
    http://localhost:3030/test/data

# Run integration test
pytest tests/test_real_fvl_integration.py -v
```

## Next Steps

1. **Expand Knowledge Base**: Load full ConceptNet or add Wikidata
2. **Tune Parameters**: Adjust entity_threshold, cache_size based on experiments
3. **Run Benchmarks**: Compare Real FVL vs Simulated FVL performance
4. **Update Paper**: Replace simulation results with real SPARQL results
5. **Deploy**: Set up persistent Fuseki instance for production

## Resources

- **Apache Jena Fuseki**: https://jena.apache.org/documentation/fuseki2/
- **ConceptNet**: https://conceptnet.io/
- **SPARQL 1.1**: https://www.w3.org/TR/sparql11-query/
- **spaCy**: https://spacy.io/
- **CAF Migration Guide**: [REAL_SPARQL_MIGRATION.md](REAL_SPARQL_MIGRATION.md)
