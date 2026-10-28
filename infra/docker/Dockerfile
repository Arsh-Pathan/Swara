# Multi-stage production Dockerfile for Swara
# Stage 1: Build & Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY swara ./swara

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build && \
    python -m build --wheel

# Stage 2: Production Minimal Runtime
FROM python:3.11-slim AS runtime

# Security: Install minimal runtime dependencies & create non-root user
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10001 swara \
    && useradd -u 10001 -g swara -s /bin/bash -m swara

WORKDIR /app

# Copy built wheel from builder and install
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Create data directories with appropriate permissions
RUN mkdir -p /app/data/speakers /app/models/cache /app/logs && \
    chown -R swara:swara /app

USER swara

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    SWARA_MODELS_DIR=/app/models/cache \
    SWARA_SPEAKER_STORE_PATH=/app/data/speakers \
    SWARA_API_HOST=0.0.0.0 \
    SWARA_API_PORT=8000 \
    SWARA_INTRA_OP_NUM_THREADS=4

EXPOSE 8000

# Container health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["swara"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
