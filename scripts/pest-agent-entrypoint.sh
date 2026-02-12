#!/bin/bash
# PEST++ Agent Entrypoint Script
# Downloads model files from MinIO and connects to PEST++ manager

set -e

echo "=============================================="
echo "PEST++ Agent Starting"
echo "=============================================="
echo "Manager: ${MANAGER_HOST}:${MANAGER_PORT}"
echo "MinIO: ${MINIO_HOST}:${MINIO_PORT}"
echo "Project: ${PROJECT_ID}"
echo "Run: ${RUN_ID}"
echo "=============================================="

# Configure MinIO client
mc alias set minio http://${MINIO_HOST}:${MINIO_PORT} ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY} --api S3v4

# Wait for manager to be available
echo "Waiting for PEST++ manager at ${MANAGER_HOST}:${MANAGER_PORT}..."
max_attempts=60
attempt=0
while ! nc -z ${MANAGER_HOST} ${MANAGER_PORT} 2>/dev/null; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo "ERROR: Manager not available after ${max_attempts} attempts"
        exit 1
    fi
    echo "  Attempt ${attempt}/${max_attempts}..."
    sleep 2
done
echo "Manager is available!"

# Download model files if PROJECT_ID and RUN_ID are set
if [ -n "${PROJECT_ID}" ] && [ -n "${RUN_ID}" ]; then
    echo "Downloading model files..."

    # Download from pest workspace path
    WORKSPACE_PATH="projects/${PROJECT_ID}/pest/${RUN_ID}/workspace"

    mc cp --recursive minio/modflow-models/${WORKSPACE_PATH}/ /work/ 2>/dev/null || {
        echo "Workspace not found at ${WORKSPACE_PATH}, trying template..."
        # Fallback to project storage path
        STORAGE_PATH=$(mc cat minio/modflow-models/projects/${PROJECT_ID}/pest/${RUN_ID}/storage_path.txt 2>/dev/null || echo "")
        if [ -n "${STORAGE_PATH}" ]; then
            mc cp --recursive minio/modflow-models/${STORAGE_PATH}/ /work/
        else
            echo "WARNING: Could not find model files to download"
        fi
    }

    echo "Files in work directory:"
    ls -la /work/
fi

# Find the panther_agent or pestpp-panther executable
AGENT_EXE=""
for exe in panther_agent pestpp-panther /usr/local/bin/panther_agent /usr/local/bin/pestpp-panther; do
    if command -v $exe &> /dev/null; then
        AGENT_EXE=$exe
        break
    fi
done

# If no panther agent found, use pestpp-glm/ies in agent mode
if [ -z "$AGENT_EXE" ]; then
    echo "No dedicated panther agent found, using PEST++ in agent mode"
    # PEST++ can run as an agent by connecting to a manager
    # The /h flag with host:port makes it act as an agent

    # Find any PST file to use
    PST_FILE=$(find /work -name "*.pst" -type f | head -1)

    if [ -n "$PST_FILE" ]; then
        echo "Found PST file: ${PST_FILE}"
        cd /work

        # Run pestpp-glm or pestpp-ies as agent
        # The /h host:port syntax connects as a worker/agent
        if command -v pestpp-glm &> /dev/null; then
            echo "Running pestpp-glm as agent..."
            exec pestpp-glm $(basename ${PST_FILE}) /h ${MANAGER_HOST}:${MANAGER_PORT}
        elif command -v pestpp-ies &> /dev/null; then
            echo "Running pestpp-ies as agent..."
            exec pestpp-ies $(basename ${PST_FILE}) /h ${MANAGER_HOST}:${MANAGER_PORT}
        else
            echo "ERROR: No PEST++ executable found"
            exit 1
        fi
    else
        echo "ERROR: No PST file found in /work"
        exit 1
    fi
else
    echo "Running ${AGENT_EXE}..."
    cd /work
    exec ${AGENT_EXE} ${MANAGER_HOST} ${MANAGER_PORT}
fi
