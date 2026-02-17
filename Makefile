# CAF Makefile
# Automation for common development tasks

.PHONY: help setup install test lint format clean docker-build docker-up docker-down k8s-deploy

COMPOSE := $(shell if command -v docker-compose >/dev/null 2>&1; then echo docker-compose; elif command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then echo "docker compose"; fi)

help:
	@echo "CAF - Causal Autonomy Framework"
	@echo ""
	@echo "Available targets:"
	@echo "  setup        - Initial setup (venv, dependencies, data dirs)"
	@echo "  install      - Install Python dependencies"
	@echo "  test         - Run test suite"
	@echo "  lint         - Run linters (mypy, black, isort)"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Remove build artifacts and cache"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-up    - Start Docker Compose services"
	@echo "  docker-down  - Stop Docker Compose services"
	@echo "  k8s-deploy   - Deploy to Kubernetes"
	@echo "  load-kb      - Load sample knowledge base"
	@echo "  run-api      - Run API server locally"

setup:
	@echo "Running setup script..."
	@bash scripts/setup.sh

install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt
	@pip install -e ".[dev]"

test:
	@echo "Running tests..."
	@pytest tests/ -v --cov=modules --cov=api

lint:
	@echo "Running linters..."
	@mypy modules/ api/
	@black --check modules/ api/
	@isort --check modules/ api/

format:
	@echo "Formatting code..."
	@black modules/ api/ scripts/
	@isort modules/ api/ scripts/

clean:
	@echo "Cleaning build artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@rm -rf build/ dist/ *.egg-info .coverage htmlcov/ .pytest_cache/
	@echo "✓ Cleaned"

docker-build:
	@echo "Building Docker images..."
	@if [ -z "$(COMPOSE)" ]; then echo "Docker Compose not found (expected 'docker compose' or 'docker-compose')"; exit 127; fi
	@cd deployment/docker && $(COMPOSE) build

docker-up:
	@echo "Starting Docker services..."
	@if [ -z "$(COMPOSE)" ]; then echo "Docker Compose not found (expected 'docker compose' or 'docker-compose')"; exit 127; fi
	@cd deployment/docker && $(COMPOSE) up -d
	@echo "Services started. Check status with: $(COMPOSE) ps"

docker-down:
	@echo "Stopping Docker services..."
	@if [ -z "$(COMPOSE)" ]; then echo "Docker Compose not found (expected 'docker compose' or 'docker-compose')"; exit 127; fi
	@cd deployment/docker && $(COMPOSE) down

k8s-deploy:
	@echo "Deploying to Kubernetes..."
	@bash scripts/deploy.sh kubernetes

load-kb:
	@echo "Loading sample knowledge base..."
	@python scripts/load_knowledge_base.py --sample

run-api:
	@echo "Starting API server..."
	@uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-inference:
	@echo "Starting Inference Engine server..."
	@python -m modules.inference_engine.server
