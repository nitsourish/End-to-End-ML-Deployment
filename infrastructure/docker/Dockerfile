# ------------------------------------------------------------------
# Multi-stage Dockerfile for the Fraud Detection API
# Stage 1: builder  — install deps + compile artifacts
# Stage 2: runtime  — lean production image
# ------------------------------------------------------------------

# ---- Stage 1: Builder ------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for scikit-learn / evidently
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: Runtime ------------------------------------------------
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Expose port (AWS App Runner default)
EXPOSE 8000

# Healthcheck for App Runner / ECS
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

USER appuser

# Uvicorn production server with 2 workers
CMD ["uvicorn", "src.serving.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
