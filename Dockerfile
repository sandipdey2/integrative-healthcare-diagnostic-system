# IHDS: MCP tool server + FastAPI + Streamlit, for Azure Container Apps / Azure Container Instances.
# scripts/start_all.sh starts all three; Streamlit (8501) is the public port, the API (8000) is optional.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Prometheus binary for the optional in-container metrics push (agent mode; only started when
# IHDS_REMOTE_WRITE_URL is set, see scripts/metrics_agent.py). Static Go binary, ~160 MB.
COPY --from=prom/prometheus:v3.5.0 /bin/prometheus /usr/local/bin/prometheus

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Pre-generate the Evidently reports so the Monitoring tab has content on first load.

# log directory for the compose api service: IHDS_LOG_FILE=/app/logs/ihds.jsonl on a named volume (inherits this owner)
RUN mkdir -p /app/logs && useradd --create-home appuser && chown -R appuser /app
USER appuser
RUN python -m ihds.pipeline
# Bake chromadb's all-MiniLM-L6-v2 ONNX model (~80 MB) into the image for the evidence retriever, so the
# container does not download it at start (it lands in /home/appuser/.cache/chroma).
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as E; E()(['warm-up'])"

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

CMD ["sh", "scripts/start_all.sh"]
