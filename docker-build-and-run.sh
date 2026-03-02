#!/bin/bash

# NHL Moneyline Model - Docker Build & Run Script
# This script builds the Docker image and runs the FastAPI server

set -e

PROJECT_NAME="nhl-model"
IMAGE_TAG="${PROJECT_NAME}:latest"
CONTAINER_NAME="${PROJECT_NAME}-api"

echo "================================================"
echo "NHL Moneyline Model - Docker Build & Run"
echo "================================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker from https://www.docker.com/"
    exit 1
fi

echo "🔨 Building Docker image: $IMAGE_TAG..."
docker build -t "$IMAGE_TAG" .

echo ""
echo "✅ Build complete!"
echo ""
echo "🚀 Starting container: $CONTAINER_NAME..."
echo ""

# Stop and remove existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Run container
docker run \
    --name "$CONTAINER_NAME" \
    -p 8000:8000 \
    -v "$(pwd)/data/processed:/app/data/processed:ro" \
    -v "$(pwd)/models:/app/models:ro" \
    --rm \
    "$IMAGE_TAG"

