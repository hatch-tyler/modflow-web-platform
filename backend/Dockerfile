# Backend Dockerfile for MODFLOW Web Platform
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    procps \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install MODFLOW 6 executables
RUN mkdir -p /usr/local/bin && \
    wget -q https://github.com/MODFLOW-USGS/executables/releases/download/14.0/linux.zip -O /tmp/mf6.zip && \
    unzip -q /tmp/mf6.zip -d /tmp/mf6 && \
    cp /tmp/mf6/mf6 /usr/local/bin/ && \
    cp /tmp/mf6/mf2005 /usr/local/bin/ && \
    cp /tmp/mf6/mfnwt /usr/local/bin/ && \
    cp /tmp/mf6/mfusgdbl /usr/local/bin/mfusg 2>/dev/null || true && \
    chmod +x /usr/local/bin/mf* && \
    rm -rf /tmp/mf6 /tmp/mf6.zip && \
    echo "MODFLOW executables installed"

# Create non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Ensure scripts are executable
RUN chmod +x scripts/*.py 2>/dev/null || true

# Change ownership to non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check - uses our custom script that supports both API and worker modes
# The mode is set via HEALTH_CHECK_MODE environment variable in docker-compose
# Increased start-period to allow for service initialization with retries
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python scripts/healthcheck.py ${HEALTH_CHECK_MODE:-api}

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
