# CAF Quick Start Guide

Get the Causal Autonomy Framework running in 15 minutes.

## Prerequisites Check

```bash
# Check Python version (need 3.12+)
python3 --version

# Check CUDA/GPU
nvidia-smi

# Check Docker
docker --version
docker-compose --version
```

## Step 1: Initial Setup (5 minutes)

```bash
cd CAF

# Run automated setup
bash scripts/setup.sh

# Activate environment
source venv/bin/activate
```

## Step 2: Configure Environment (2 minutes)

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (use your preferred editor)
nano .env
```

**Minimum required changes:**
- Set `MODEL_NAME` to your desired LLM (or keep default)
- Set `API_KEY` to a secure value
- Adjust `GPU_MEMORY_UTILIZATION` based on your GPU

## Step 3: Start Services (5 minutes)

```bash
# Start Docker services
make docker-up

# Wait for services to initialize (~30 seconds)
sleep 30

# Check service health
docker-compose ps
```

You should see all services as "Up":
- fuseki
- chromadb
- inference-engine (if GPU available)
- api-gateway
- prometheus
- grafana

## Step 4: Load Sample Knowledge Base (2 minutes)

```bash
# Load sample data for testing
make load-kb
```

This creates sample entities in ChromaDB and sample RDF triplets in Fuseki.

## Step 5: Test the System (1 minute)

### Option A: Using cURL

```bash
curl -X POST http://localhost:8000/v1/infer \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is water made of?",
    "max_refinement_iterations": 2,
    "verification_threshold": 0.7
  }'
```

### Option B: Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/infer",
    json={
        "prompt": "What is water made of?",
        "max_refinement_iterations": 2,
        "verification_threshold": 0.7
    }
)

print(response.json())
```

### Option C: Interactive API Docs

Open in browser: http://localhost:8000/docs

## Troubleshooting

### Services won't start

```bash
# Check logs
docker-compose logs

# Restart specific service
docker-compose restart fuseki
```

### GPU not detected

If running without GPU, set in `.env`:
```
USE_VLLM=false
MODEL_NAME=meta-llama/Llama-3-8b-chat-hf  # Smaller model for CPU
```

### Port conflicts

Change ports in `.env`:
```
API_PORT=8080  # instead of 8000
```

### Out of memory

Reduce GPU memory usage in `.env`:
```
GPU_MEMORY_UTILIZATION=0.7  # instead of 0.9
```

## Next Steps

1. **Load Real Data**: Download ConceptNet and Wikidata
   ```bash
   python scripts/load_knowledge_base.py --conceptnet path/to/conceptnet.csv
   ```

2. **Monitor System**: Access Grafana at http://localhost:3000
   - Username: `admin`
   - Password: `admin`

3. **Customize Prompts**: Edit the prompt template in:
   - `modules/inference_engine/engine.py` → `_build_causal_prompt()`

4. **Add Custom Entities**: Use the Entity Linker API:
   ```python
   from modules.semantic_parser import EntityLinker
   linker = EntityLinker()
   linker.add_entity("custom:my_entity", "My Entity", "custom")
   ```

5. **Deploy to Production**: See Kubernetes deployment guide
   ```bash
   make k8s-deploy
   ```

## Key Endpoints

- **API Gateway**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Fuseki**: http://localhost:3030
- **ChromaDB**: http://localhost:8001
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## Common Commands

```bash
# Stop all services
make docker-down

# View logs
docker-compose logs -f api-gateway

# Run tests
make test

# Format code
make format

# Clean build artifacts
make clean
```

## Getting Help

- Check logs: `docker-compose logs [service-name]`
- Read full docs: [README.md](README.md)
- Report issues: GitHub Issues
- Email support: support@caf-framework.org

## Verification of Installation

Run this test to verify everything works:

```bash
# Test all components
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "components": {
#     "inference_engine": true,
#     "semantic_parser": true,
#     "truth_anchor": true,
#     "causal_validator": true
#   }
# }
```

If all components show `true`, your installation is complete!
