# ─────────────────────────────────────────────────────────────────────────────
# VecDB — Multi-stage Dockerfile
#
# Stage 1 (builder): installs Python deps into a venv to keep the final image lean.
# Stage 2 (runtime): copies only the venv + application code into a slim image.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: build ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools (for numpy wheel compilation if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ───────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY vecdb/ ./vecdb/

# Create persistent data volume mount point
RUN mkdir -p /app/vdb_data

# Expose REST API port
EXPOSE 8088

# Health check — polls /health every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8088/health')" || exit 1

# Run FastAPI server with uvicorn
CMD ["uvicorn", "vecdb.server.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8088", \
     "--workers", "1"]
