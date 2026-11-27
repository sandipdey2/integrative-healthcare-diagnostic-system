#!/usr/bin/env sh
# Start the three layers in one container (or locally). The MCP server gets no OPENAI_API_KEY:
#   1. MCP tool server (streamable HTTP)  127.0.0.1:${IHDS_MCP_PORT:-8765}/mcp
#   2. FastAPI service                    0.0.0.0:${IHDS_API_PORT:-8000}   (agents get tools from 1)
#   3. Streamlit UI                       0.0.0.0:${PORT:-8501}            (calls 2 via IHDS_API_URL)
# Optional: if GRAFANA_RW_URL is set, a Prometheus agent (127.0.0.1:9091) scrapes 2's /metrics and
# remote_writes the ihds_* series there (Grafana Cloud or the self-hosted ACA Prometheus). Off when unset;
# if the agent fails, the app keeps running. See scripts/metrics_agent.py and docs/MLOPS.md.
set -eu
MCP_PORT="${IHDS_MCP_PORT:-8765}"
API_PORT="${IHDS_API_PORT:-8000}"
UI_PORT="${PORT:-8501}"

wait_for() {  # wait_for <url> <seconds>
  python - "$1" "$2" <<'PY'
import sys, time, urllib.request, urllib.error
url, limit = sys.argv[1], float(sys.argv[2])
end = time.time() + limit
while time.time() < end:
    try:
        urllib.request.urlopen(url, timeout=2); sys.exit(0)
    except urllib.error.HTTPError:
        sys.exit(0)  # server is answering (MCP returns 4xx to a bare GET)
    except Exception:
        time.sleep(1)
sys.exit(f"timed out waiting for {url}")
PY
}

env -u OPENAI_API_KEY -u GRAFANA_RW_TOKEN -u LANGSMITH_API_KEY -u LANGCHAIN_API_KEY -u ARIZE_API_KEY -u ARIZE_SPACE_ID -u DATABASE_URL python -m mcp_server.server --transport streamable-http --host 127.0.0.1 --port "$MCP_PORT" &
wait_for "http://127.0.0.1:$MCP_PORT/mcp" 180

IHDS_MCP_URL="http://127.0.0.1:$MCP_PORT/mcp" IHDS_TOOLS=mcp \
  env -u GRAFANA_RW_TOKEN uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT" &
wait_for "http://127.0.0.1:$API_PORT/health" 180

if [ -n "${GRAFANA_RW_URL:-}" ]; then
  IHDS_API_PORT="$API_PORT" python scripts/metrics_agent.py &
fi

IHDS_EMBED_API=0 IHDS_API_URL="http://127.0.0.1:$API_PORT" exec env -u GRAFANA_RW_TOKEN streamlit run app.py \
  --server.port="$UI_PORT" --server.address=0.0.0.0 --server.headless=true
