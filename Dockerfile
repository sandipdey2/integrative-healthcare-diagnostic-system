# IHDS: MCP tool server + FastAPI + Streamlit, for Azure Container Apps / Azure Container Instances.
# scripts/start_all.sh starts all three; Streamlit (8501) is the public port, the API (8000) is optional.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Pre-generate the Evidently reports so the Monitoring tab has content on first load.
RUN python -m ihds.pipeline

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

CMD ["sh", "scripts/start_all.sh"]
