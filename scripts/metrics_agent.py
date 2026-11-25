"""Optional in-container metrics push: Prometheus in agent mode scrapes the local FastAPI /metrics and
remote_writes the ihds_* series to a Prometheus-compatible endpoint.

Off unless GRAFANA_RW_URL is set (scripts/start_all.sh only starts it then). Used for:
  A) Grafana Cloud:  GRAFANA_RW_URL=https://prometheus-prod-XX-....grafana.net/api/prom/push
                     GRAFANA_RW_USER=<numeric instance ID>  GRAFANA_RW_TOKEN=<glc_... metrics:write token>
                     (all three are Container Apps secrets: deploy/azure/enable_metrics_push.sh)
  B) self-hosted:    GRAFANA_RW_URL=http://ihds-prometheus/api/v1/write   (no auth, same ACA environment)
The token is read from the environment (a Container Apps secretref), written to a 0600 file and passed to
Prometheus as `password_file`; it is never written into the config or logged. Only `ihds_*` series and `up`
are forwarded (Python process/GC metrics are dropped) to stay far below the Grafana Cloud free-tier series limit.

    python scripts/metrics_agent.py --print-config     # rendered config, password redacted
    python scripts/metrics_agent.py                    # render + exec `prometheus --agent`
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

AGENT_DIR = Path(os.environ.get("IHDS_AGENT_DIR", "/tmp/ihds-agent"))
KEEP_SERIES = "ihds_.*|up"


def render(env: dict | None = None) -> dict:
    env = dict(os.environ if env is None else env)
    url = env.get("GRAFANA_RW_URL", "").strip()
    if not url:
        raise SystemExit("GRAFANA_RW_URL is not set: metrics push stays off")
    api_port = env.get("IHDS_API_PORT", "8000")
    rw: dict = {
        "url": url,
        "write_relabel_configs": [
            {"source_labels": ["__name__"], "regex": KEEP_SERIES, "action": "keep"},
            # prometheus_client's *_created timestamps are not used by the dashboard; dropping them saves series
            {"source_labels": ["__name__"], "regex": "ihds_.*_created", "action": "drop"},
        ],
        "queue_config": {"max_samples_per_send": 2000, "batch_send_deadline": "15s"},
    }
    user = env.get("GRAFANA_RW_USER", "").strip()
    if user:
        rw["basic_auth"] = {"username": user, "password_file": str(AGENT_DIR / "remote_write_token")}
    return {
        "global": {
            "scrape_interval": env.get("IHDS_SCRAPE_INTERVAL", "15s"),
            "external_labels": {"service": "ihds",
                                "deployment": env.get("IHDS_METRICS_DEPLOYMENT", "azure-container-apps")},
        },
        "scrape_configs": [{"job_name": "ihds-api", "metrics_path": "/metrics",
                            "static_configs": [{"targets": [f"127.0.0.1:{api_port}"]}]}],
        "remote_write": [rw],
    }


def write_files(cfg: dict, env: dict | None = None) -> Path:
    env = dict(os.environ if env is None else env)
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    if "basic_auth" in cfg["remote_write"][0]:
        pw = env.get("GRAFANA_RW_TOKEN", "")
        if not pw:
            raise SystemExit("GRAFANA_RW_USER is set but GRAFANA_RW_TOKEN is empty")
        pw_file = Path(cfg["remote_write"][0]["basic_auth"]["password_file"])
        fd = os.open(pw_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(pw)
    path = AGENT_DIR / "agent.yml"
    path.write_text(json.dumps(cfg, indent=2))  # JSON is valid YAML
    return path


def main(argv: list[str]) -> None:
    cfg = render()
    if "--print-config" in argv:
        print(json.dumps(cfg, indent=2))
        return
    path = write_files(cfg)
    binary = shutil.which("prometheus") or "/usr/local/bin/prometheus"
    env = {k: v for k, v in os.environ.items() if k not in ("GRAFANA_RW_TOKEN", "OPENAI_API_KEY")}
    print(f"metrics agent: pushing ihds_* series to {cfg['remote_write'][0]['url'].split('?')[0]}", flush=True)
    os.execve(binary, [binary, "--agent", f"--config.file={path}", f"--storage.agent.path={AGENT_DIR / 'wal'}",
                       "--web.listen-address=127.0.0.1:9091", "--log.level=warn"], env)


if __name__ == "__main__":
    main(sys.argv[1:])
