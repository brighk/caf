# CAF Architecture Deep Dive

## System Overview

The Causal Autonomy Framework implements a **hybrid neural-symbolic architecture** that achieves deterministic AI outputs through formal verification and causal grounding.

## Design Principles

### 1. Separation of Concerns

Each module has a single, well-defined responsibility:
- **Module A**: Hypothesis generation (stochastic)
- **Module B**: Semantic extraction (deterministic)
- **Module C**: Truth verification (deterministic)
- **Module D**: Causal validation (deterministic)

### 2. Recursive Refinement

The system implements a feedback loop:

```
┌─────────────────────────────────────────┐
│         Recursive Refinement Loop       │
└─────────────────────────────────────────┘

  ┌───────────┐
  │  Prompt   │
  └─────┬─────┘
        │
        ▼
  ┌───────────────┐
  │ Generate R_c  │ ◄──────┐
  │  (Module A)   │        │
  └───────┬───────┘        │
          │                │
          ▼                │
  ┌───────────────┐        │
  │ Extract (s,p,o)│       │
  │  (Module B)   │        │
  └───────┬───────┘        │
          │                │
          ▼                │
  ┌───────────────┐        │
  │ Verify in KB  │        │
  │  (Module C)   │        │
  └───────┬───────┘        │
          │                │
          ▼                │
  ┌───────────────┐        │
  │   Valid?      │────N───┤ Add constraints
  └───────┬───────┘        │
          │                │
          Y                │
          │                │
          ▼                │
  ┌───────────────┐        │
  │ Causal Check  │        │
  │  (Module D)   │        │
  └───────┬───────┘        │
          │                │
          ▼                │
  ┌───────────────┐        │
  │ Consistent?   │────N───┘
  └───────┬───────┘
          │
          Y
          │
          ▼
  ┌───────────────┐
  │ Output R_f    │
  └───────────────┘
```

### 3. Microservices Architecture

CAF uses containerization to isolate:
- GPU-heavy inference from CPU-heavy parsing
- Stateful graph database from stateless API
- Monitoring from core services

## Data Flow

### Request Pipeline

```
HTTP POST /v1/infer
    ↓
┌─────────────────────┐
│  FastAPI Gateway    │ ← Pydantic validation
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Inference Engine   │ ← vLLM on GPU
│  (gRPC/HTTP)        │
└──────────┬──────────┘
           │
           ▼ (response + assertions)
┌─────────────────────┐
│  Semantic Parser    │ ← spaCy NER
│  + Entity Linker    │ ← ChromaDB lookup
└──────────┬──────────┘
           │
           ▼ (triplets)
┌─────────────────────┐
│  Truth Anchor       │ ← SPARQL query
│  (Fuseki)           │ ← Levenshtein match
└──────────┬──────────┘
           │
           ▼ (verification_result)
┌─────────────────────┐
│  Causal Validator   │ ← DoWhy axioms
│  (NetworkX)         │ ← Cycle detection
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Metrics Collector  │ ← Prometheus
└──────────┬──────────┘
           │
           ▼
      JSON Response
```

## Module Specifications

### Module A: Inference Engine

**Purpose**: Generate natural language responses with causal assertions

**Technology Stack**:
- PyTorch 2.1+
- vLLM for inference optimization
- CUDA 12.1
- Llama-3-70B (default model)

**Key Features**:
- PagedAttention for memory efficiency
- Continuous batching for throughput
- Structured prompt engineering for causal extraction

**Performance**:
- Latency: 1-3s per request
- Throughput: 10-20 req/s (single A100)
- VRAM: 40-80GB

**API**:
```python
POST /generate
{
    "prompt": str,
    "max_tokens": int,
    "temperature": float,
    "constraints": List[str]  # Feedback from verification
}
→ {
    "text": str,
    "causal_assertions_raw": List[str],
    "metadata": dict
}
```

### Module B: Semantic Parser

**Purpose**: Extract structured RDF triplets from natural language

**Technology Stack**:
- spaCy 3.7 (NER)
- Sentence-Transformers (embeddings)
- ChromaDB (vector similarity)
- LangChain (orchestration)

**Pipeline**:
1. **NER**: Extract entities (PERSON, ORG, GPE, etc.)
2. **Dependency Parsing**: Find subject-predicate-object patterns
3. **Entity Linking**: Map entities to URIs via vector similarity
4. **SPARQL Generation**: Create queries from triplets

**Performance**:
- Latency: 50-200ms
- Accuracy: Depends on entity coverage in ChromaDB

**API**:
```python
await parser.parse(text, causal_assertions)
→ ParsedResult(
    triplets: List[Triplet],
    sparql_query: str
)
```

### Module C: Truth Anchor

**Purpose**: Verify triplets against ground truth knowledge

**Technology Stack**:
- Apache Jena Fuseki (RDF triplestore)
- SPARQL 1.1
- RDFLib (Python interface)
- Levenshtein distance (fuzzy matching)

**Data Sources**:
- ConceptNet 5.7 (common sense)
- Wikidata (factual knowledge)
- Custom domain ontologies

**Verification Methods**:
1. **Exact Match**: URI comparison
2. **Fuzzy Match**: Levenshtein ratio ≥ threshold
3. **Semantic Match**: Embedding similarity (planned)

**Performance**:
- Latency: 10-100ms per query
- Scalability: Millions of triplets

**API**:
```python
await truth_anchor.verify(triplets, threshold=0.8)
→ VerificationResult(
    is_valid: bool,
    matched_triplets: List[Triplet],
    contradictions: List[str],
    similarity_score: float
)
```

### Module D: Causal Validator

**Purpose**: Ensure logical consistency of causal assertions

**Technology Stack**:
- NetworkX (graph analysis)
- DoWhy (causal inference - planned)

**Validation Axioms**:
1. **Acyclicity**: No causal loops (A → B → A)
2. **Transitivity**: If A → B → C, then A influences C
3. **Consistency**: No contradictory assertions

**Graph Analysis**:
- Detects cycles using DFS
- Identifies bidirectional causation
- Validates temporal ordering (if metadata available)

**Performance**:
- Latency: 5-50ms
- Graph size: Thousands of nodes

**API**:
```python
await causal_validator.validate(assertions, verified_triplets)
→ ValidationResult(
    is_valid: bool,
    violations: List[str],
    causal_graph_nodes: List[str],
    causal_graph_edges: List[tuple]
)
```

## Configuration Management

### Environment Variables

All configuration via `.env`:

```
# Model
MODEL_NAME=meta-llama/Llama-3-70b-chat-hf
TENSOR_PARALLEL_SIZE=1

# Thresholds
VERIFICATION_THRESHOLD=0.8
MAX_REFINEMENT_ITERATIONS=3

# Endpoints
FUSEKI_ENDPOINT=http://fuseki:3030/dataset/query
CHROMADB_HOST=chromadb
INFERENCE_ENGINE_URL=http://inference-engine:8001
```

### Pydantic Settings

Type-safe configuration with validation:
- Auto-loads from `.env`
- Validates types and ranges
- Provides defaults
- Singleton pattern via `@lru_cache`

## Monitoring & Observability

### Prometheus Metrics

**Request Metrics**:
- `caf_inference_requests_total` (Counter)
- `caf_verification_attempts_total` (Counter)

**Latency Metrics**:
- `caf_inference_latency_seconds` (Histogram)
- `caf_verification_latency_seconds` (Histogram)
- `caf_total_pipeline_latency_seconds` (Histogram)

**Quality Metrics**:
- `caf_refinement_iterations` (Histogram)
- `caf_verification_similarity_score` (Histogram)
- `caf_causal_violations_total` (Counter)

**System Metrics**:
- `caf_component_health` (Gauge)
- `caf_gpu_memory_usage_bytes` (Gauge)

### Grafana Dashboards

Pre-configured dashboards for:
1. **System Health**: Component status, uptime
2. **Performance**: Latencies, throughput
3. **Quality**: Verification scores, refinement counts
4. **Resources**: GPU/CPU/Memory usage

## Scalability

### Horizontal Scaling

**Stateless Components** (can scale easily):
- API Gateway (load balanced)
- Inference Engine (GPU pool)

**Stateful Components** (require coordination):
- Fuseki (replication)
- ChromaDB (sharding)

### Vertical Scaling

**GPU Scaling**:
- Tensor parallelism: `TENSOR_PARALLEL_SIZE=2` (multi-GPU)
- Pipeline parallelism: Split model layers across GPUs

**Memory Optimization**:
- Quantization: Use 4-bit/8-bit models
- Offloading: CPU offload for large models

## Security

### API Authentication

Optional API key authentication:
```python
headers = {"X-API-Key": "your-secret-key"}
```

### Network Isolation

Services communicate within Docker network:
- No external exposure except API Gateway
- Internal DNS resolution

### Data Privacy

- No data persistence by default
- Optional request logging
- GDPR-compliant data handling

## Error Handling

### Graceful Degradation

If verification fails after `MAX_REFINEMENT_ITERATIONS`:
1. Return HTTP 422 with detailed contradictions
2. Allow client to retry with relaxed threshold
3. Log failure for analysis

### Circuit Breaker

Services implement health checks:
- `/health`: Component status
- `/ready`: Kubernetes readiness probe

## Future Enhancements

### Planned Features

1. **Causal Discovery**: Automatic extraction of causal graphs
2. **Counterfactual Reasoning**: "What if" queries
3. **Temporal Reasoning**: Time-aware verification
4. **Multi-modal Input**: Image + text grounding
5. **Federated Learning**: Distributed knowledge bases

### Research Directions

1. **Neuro-symbolic Integration**: Tighter coupling of neural and symbolic
2. **Probabilistic Verification**: Bayesian confidence intervals
3. **Active Learning**: KB expansion from user feedback

## References

- [vLLM Documentation](https://vllm.readthedocs.io/)
- [Apache Jena](https://jena.apache.org/)
- [ConceptNet](https://conceptnet.io/)
- [DoWhy](https://microsoft.github.io/dowhy/)
