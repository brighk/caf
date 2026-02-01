# Causal Autonomy Framework (CAF)

**Sovereign Agent with Deterministic Output via Causal Grounding**

## Overview

The Causal Autonomy Framework (CAF) is a production-grade system that combines neural inference with symbolic verification to achieve **deterministic, causally-grounded AI outputs**. Unlike traditional LLMs that produce stochastic results, CAF anchors responses in formal knowledge bases, ensuring factual accuracy and logical consistency.

### Key Innovation

CAF implements a **recursive refinement loop** where:
1. Neural models generate initial hypotheses
2. Symbolic verifiers check assertions against ground truth
3. Causal validators ensure logical consistency
4. The system iteratively refines until **K ⊨ φ** (knowledge base entails the assertion)

This architecture minimizes variance σ² by grounding the LLM's latent space in the fixed coordinates of an RDF knowledge graph.

## Architecture

### Four-Component Design

```
┌─────────────────────────────────────────────────────────────┐
│                    CAF Pipeline                              │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐│
│  │ Module A │──▶│ Module B │──▶│ Module C │──▶│ Module D ││
│  │Inference │   │ Semantic │   │  Truth   │   │  Causal  ││
│  │ Engine   │   │  Parser  │   │  Anchor  │   │Validator ││
│  │ (Neural) │   │(Middleware)│ │(Symbolic)│   │(Verifier)││
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘│
│       │              │               │              │       │
│    Llama-3      spaCy+LCEL      SPARQL/RDF      DoWhy      │
│    vLLM         ChromaDB      Jena Fuseki     NetworkX    │
└─────────────────────────────────────────────────────────────┘
```

#### Module A: Inference Engine (Neural)
- **Hardware**: NVIDIA A100/H100 GPU (80GB VRAM)
- **Model**: Llama-3-70B with vLLM for optimized inference
- **Role**: Generates semantic hypotheses and causal assertions

#### Module B: Semantic Parser (Middleware)
- **Framework**: LangChain + spaCy
- **Task**: Text-to-SPARQL mapping via NER and entity linking
- **Storage**: ChromaDB for vector similarity search

#### Module C: Truth Anchor (Symbolic)
- **Database**: Apache Jena Fuseki (RDF triplestore)
- **Data**: ConceptNet 5.7 + Wikidata subset
- **Protocol**: SPARQL 1.1 over HTTP

#### Module D: Causal Validator (Verification)
- **Library**: DoWhy + NetworkX
- **Mechanism**: Axiomatic verification of causal relationships
- **Checks**: Acyclicity, transitivity, consistency

## Technical Stack

| Layer | Technology | Specification |
|-------|-----------|---------------|
| OS | Ubuntu 22.04 LTS | Kernel 5.15+ |
| Runtime | Python 3.12+ | Async I/O |
| API | FastAPI | Pydantic validation |
| Inference | PyTorch + vLLM | CUDA 12.1 |
| Knowledge Graph | Apache Jena Fuseki | SPARQL 1.1 |
| Vector Index | FAISS + ChromaDB | HNSW indexing |
| Orchestration | Docker + Kubernetes | Microservices |
| Monitoring | Prometheus + Grafana | Real-time metrics |

## Installation

### Prerequisites

- Ubuntu 22.04 LTS
- Python 3.12+
- NVIDIA GPU with CUDA 12.1+
- Docker & Docker Compose
- 32GB+ RAM (80GB+ for GPU)

### Quick Start

```bash
# Clone repository
cd CAF

# Run setup script
bash scripts/setup.sh

# Activate virtual environment
source venv/bin/activate

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start services
cd deployment/docker
docker-compose up -d

# Load sample knowledge base
python scripts/load_knowledge_base.py --sample

# Run API server
uvicorn api.main:app --reload
```

### Kubernetes Deployment

```bash
# Deploy to K8s cluster
bash scripts/deploy.sh kubernetes

# Verify deployment
kubectl get pods -n caf-system

# Port forward API
kubectl port-forward -n caf-system svc/api-gateway-service 8000:8000
```

## Usage

### API Example

```python
import requests

# Send inference request
response = requests.post(
    "http://localhost:8000/v1/infer",
    json={
        "prompt": "What causes rain?",
        "max_refinement_iterations": 3,
        "verification_threshold": 0.8,
        "enable_causal_validation": True
    }
)

result = response.json()
print(f"Response: {result['final_response']['text']}")
print(f"Confidence: {result['final_response']['confidence']}")
print(f"Refinements: {result['final_response']['refinement_iterations']}")
```

### Data Pipeline

```
Input: "What causes rain?"
  ↓
[Module A] LLM generates:
  ANSWER: Rain is caused by water condensation in clouds.
  CAUSAL_ASSERTIONS:
  - Water vapor causes condensation
  - Condensation causes precipitation
  ↓
[Module B] Extracts triplets:
  (water_vapor, causes, condensation)
  (condensation, causes, precipitation)
  ↓
[Module C] Verifies against KB:
  SELECT ?o WHERE { <water_vapor> <causes> ?o }
  Result: Match found (similarity=0.95)
  ↓
[Module D] Validates causality:
  Check: No cycles ✓
  Check: No contradictions ✓
  ↓
Output: FinalResponse(
  text="Rain is caused by water condensation in clouds.",
  confidence=0.95,
  verification_status=VALID
)
```

## Configuration

Key settings in `.env`:

```bash
# Model selection
MODEL_NAME=meta-llama/Llama-3-70b-chat-hf
TENSOR_PARALLEL_SIZE=1  # Multi-GPU parallelism

# Verification thresholds
VERIFICATION_THRESHOLD=0.8
MAX_REFINEMENT_ITERATIONS=3

# Service endpoints
FUSEKI_ENDPOINT=http://fuseki:3030/dataset/query
CHROMADB_HOST=chromadb
INFERENCE_ENGINE_URL=http://inference-engine:8001
```

## Monitoring

Access monitoring dashboards:

- **API Gateway**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

Key metrics tracked:
- Verification latency (`caf_verification_latency_seconds`)
- Refinement iterations (`caf_refinement_iterations`)
- Similarity scores (`caf_verification_similarity_score`)
- Component health (`caf_component_health`)

## Knowledge Base

### Loading ConceptNet

```bash
# Download ConceptNet 5.7
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
gunzip conceptnet-assertions-5.7.0.csv.gz

# Load into CAF
python scripts/load_knowledge_base.py --conceptnet conceptnet-assertions-5.7.0.csv
```

### Loading Wikidata

```bash
# Download Wikidata subset (filtered)
# See: https://www.wikidata.org/wiki/Wikidata:Database_download

# Load into Fuseki
python scripts/load_knowledge_base.py --wikidata wikidata-subset.nt
```

## Development

### Project Structure

```
CAF/
├── api/                    # FastAPI gateway
│   ├── main.py            # Main application
│   └── models.py          # Pydantic models
├── modules/               # Core modules
│   ├── inference_engine/  # Module A
│   ├── semantic_parser/   # Module B
│   ├── truth_anchor/      # Module C
│   └── causal_validator/  # Module D
├── config/                # Configuration files
├── deployment/            # Docker & K8s configs
├── monitoring/            # Prometheus metrics
├── scripts/               # Utility scripts
└── tests/                 # Unit & integration tests
```

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=modules --cov=api
```

## Performance

Typical latencies on A100 GPU:

- **Inference**: 1-3 seconds (Llama-3-70B)
- **Parsing**: 50-200ms (spaCy + entity linking)
- **Verification**: 10-100ms (SPARQL query)
- **Validation**: 5-50ms (graph analysis)
- **Total Pipeline**: 2-5 seconds

Throughput: ~10-20 requests/second (single GPU)

## Formal Definitions

### Deterministic Output

An output where the variance σ² is minimized by grounding the latent space of the LLM in the fixed coordinates of the RDF Graph.

```
σ²(output) → 0 as K ⊨ φ
```

### Recursive Refinement

The algorithmic loop where the error signal from the FVL (Formal Verification Layer) is fed back into the IL (Inference Layer) to adjust the output until K ⊨ φ is satisfied.

```
while ¬(K ⊨ φ) and iterations < max:
    φ' ← refine(φ, contradictions)
    verify(φ')
```

## Citation

If you use CAF in your research, please cite:

```bibtex
@software{caf2024,
  title={Causal Autonomy Framework: Deterministic AI via Symbolic Grounding},
  author={CAF Development Team},
  year={2024},
  url={https://github.com/your-org/caf}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Support

- Issues: [GitHub Issues](https://github.com/your-org/caf/issues)
- Documentation: [Full Docs](https://caf-docs.readthedocs.io)
- Email: support@caf-framework.org
