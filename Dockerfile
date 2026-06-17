# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix /install -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Non-root user for security
RUN useradd -m -u 1000 appuser

# Copy installed packages from builder into system prefix (stable across minor Python versions)
COPY --from=builder /install /usr/local

USER appuser

COPY --chown=appuser:appuser . .

# Expose Streamlit and FastAPI ports
EXPOSE 8501 8000

# Health check for the dashboard
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: run the Streamlit dashboard
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
