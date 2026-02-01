#!/bin/bash
# CAF Deployment Script
# Deploys CAF to Kubernetes or Docker Compose

set -e

DEPLOYMENT_MODE=${1:-docker}

echo "=========================================="
echo "CAF Deployment Script"
echo "Mode: $DEPLOYMENT_MODE"
echo "=========================================="
echo ""

if [ "$DEPLOYMENT_MODE" = "docker" ]; then
    echo "Deploying with Docker Compose..."

    # Build images
    echo "Building Docker images..."
    cd deployment/docker
    docker-compose build
    echo "✓ Images built"
    echo ""

    # Start services
    echo "Starting services..."
    docker-compose up -d
    echo "✓ Services started"
    echo ""

    # Wait for services to be ready
    echo "Waiting for services to be ready..."
    sleep 10

    # Check service health
    echo "Checking service health..."
    docker-compose ps
    echo ""

    echo "Services available at:"
    echo "  - API Gateway: http://localhost:8000"
    echo "  - Inference Engine: http://localhost:8002"
    echo "  - Fuseki: http://localhost:3030"
    echo "  - ChromaDB: http://localhost:8001"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000"
    echo ""

elif [ "$DEPLOYMENT_MODE" = "kubernetes" ]; then
    echo "Deploying to Kubernetes..."

    # Create namespace
    echo "Creating namespace..."
    kubectl apply -f deployment/kubernetes/namespace.yaml
    echo "✓ Namespace created"
    echo ""

    # Deploy inference engine
    echo "Deploying inference engine..."
    kubectl apply -f deployment/kubernetes/inference-deployment.yaml
    echo "✓ Inference engine deployed"
    echo ""

    # Wait for pods to be ready
    echo "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=inference-engine -n caf-system --timeout=300s
    echo "✓ Pods ready"
    echo ""

    # Show deployment status
    echo "Deployment status:"
    kubectl get pods -n caf-system
    echo ""

    echo "To access services:"
    echo "  kubectl port-forward -n caf-system svc/api-gateway-service 8000:8000"
    echo ""

else
    echo "ERROR: Invalid deployment mode: $DEPLOYMENT_MODE"
    echo "Usage: ./deploy.sh [docker|kubernetes]"
    exit 1
fi

echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
