#!/bin/bash
# Start PEST++ agents on a helper machine
# Run this script on Linux/WSL2 machines to contribute workers to calibration

set -e

# Configuration - modify these for your setup
MANAGER_IP="${MANAGER_IP:-192.168.1.100}"
MANAGER_PORT="${MANAGER_PORT:-4004}"
MINIO_HOST="${MINIO_HOST:-${MANAGER_IP}}"
MINIO_PORT="${MINIO_PORT:-9000}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"
NUM_AGENTS="${NUM_AGENTS:-4}"
PROJECT_ID="${PROJECT_ID:-}"
RUN_ID="${RUN_ID:-}"

# Docker image - use local registry or pull from Docker Hub
IMAGE="${IMAGE:-modflow-pest-agent:latest}"

echo "=============================================="
echo "PEST++ Agent Launcher"
echo "=============================================="
echo "Manager: ${MANAGER_IP}:${MANAGER_PORT}"
echo "MinIO: ${MINIO_HOST}:${MINIO_PORT}"
echo "Agents to start: ${NUM_AGENTS}"
echo "Project ID: ${PROJECT_ID:-<not set>}"
echo "Run ID: ${RUN_ID:-<not set>}"
echo "=============================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    exit 1
fi

# Check if we can reach the manager
echo "Checking connectivity to manager..."
if ! nc -z ${MANAGER_IP} ${MANAGER_PORT} 2>/dev/null; then
    echo "WARNING: Cannot reach manager at ${MANAGER_IP}:${MANAGER_PORT}"
    echo "Make sure the calibration is running and port ${MANAGER_PORT} is open"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Stop any existing agents
echo "Stopping existing agents..."
docker ps -q --filter "name=pest-agent-" | xargs -r docker stop 2>/dev/null || true
docker ps -aq --filter "name=pest-agent-" | xargs -r docker rm 2>/dev/null || true

# Create a shared work directory
WORK_DIR="/tmp/pest-agents-work"
mkdir -p ${WORK_DIR}

# Start agents
echo "Starting ${NUM_AGENTS} agents..."
for i in $(seq 1 $NUM_AGENTS); do
    CONTAINER_NAME="pest-agent-${i}"
    AGENT_WORK="${WORK_DIR}/agent-${i}"
    mkdir -p ${AGENT_WORK}

    echo "  Starting ${CONTAINER_NAME}..."
    docker run -d \
        --name ${CONTAINER_NAME} \
        --restart on-failure:5 \
        -v ${AGENT_WORK}:/work \
        -e MANAGER_HOST=${MANAGER_IP} \
        -e MANAGER_PORT=${MANAGER_PORT} \
        -e MINIO_HOST=${MINIO_HOST} \
        -e MINIO_PORT=${MINIO_PORT} \
        -e MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY} \
        -e MINIO_SECRET_KEY=${MINIO_SECRET_KEY} \
        -e PROJECT_ID=${PROJECT_ID} \
        -e RUN_ID=${RUN_ID} \
        ${IMAGE}

    # Small delay to avoid overwhelming the manager
    sleep 1
done

echo ""
echo "=============================================="
echo "Started ${NUM_AGENTS} PEST++ agents"
echo "=============================================="
echo ""
echo "To view agent logs:"
echo "  docker logs -f pest-agent-1"
echo ""
echo "To stop all agents:"
echo "  docker stop \$(docker ps -q --filter 'name=pest-agent-')"
echo ""
echo "To check agent status:"
echo "  docker ps --filter 'name=pest-agent-'"
echo ""
