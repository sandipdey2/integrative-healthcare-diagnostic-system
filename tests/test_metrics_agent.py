"""scripts/metrics_agent.py: the optional in-container Prometheus agent (Grafana Cloud push). Stdlib only."""
import importlib.util
import json
import os
import shutil
import stat
import subprocess

import pytest

from ihds.config import ROOT

spec = importlib.util.spec_from_file_location("metrics_agent", ROOT / "scripts" / "metrics_agent.py")
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)

CLOUD = {"GRAFANA_RW_URL": "https://prometheus-prod-01-example.grafana.net/api/prom/push",
         "GRAFANA_RW_USER": "123456", "GRAFANA_RW_TOKEN": "test-token-value"}


def test_agent_is_off_without_url():
    with pytest.raises(SystemExit, match="stays off"):
        agent.render({})
    start_all = (ROOT / "scripts" / "start_all.sh").read_text()
    assert 'if [ -n "${GRAFANA_RW_URL:-}" ]; then' in start_all


def test_grafana_cloud_config_uses_password_file_and_keeps_only_ihds_series(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "AGENT_DIR", tmp_path)
    cfg = agent.render(CLOUD)
    rw = cfg["remote_write"][0]
    assert rw["url"] == CLOUD["GRAFANA_RW_URL"]
    assert rw["basic_auth"]["username"] == "123456" and "password" not in rw["basic_auth"]
    assert rw["write_relabel_configs"][0] == {"source_labels": ["__name__"], "regex": "ihds_.*|up", "action": "keep"}
    assert rw["write_relabel_configs"][1]["regex"] == "ihds_.*_created"
    assert cfg["scrape_configs"][0]["static_configs"][0]["targets"] == ["127.0.0.1:8000"]
    path = agent.write_files(cfg, CLOUD)
    assert "test-token-value" not in path.read_text()
    token_file = tmp_path / "remote_write_token"
    assert token_file.read_text() == "test-token-value"
    assert stat.S_IMODE(os.stat(token_file).st_mode) == 0o600
    json.loads(path.read_text())


def test_self_hosted_config_has_no_auth_and_user_without_token_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "AGENT_DIR", tmp_path)
    cfg = agent.render({"GRAFANA_RW_URL": "http://ihds-prometheus/api/v1/write"})
    assert "basic_auth" not in cfg["remote_write"][0]
    with pytest.raises(SystemExit, match="GRAFANA_RW_TOKEN is empty"):
        agent.write_files(agent.render({**CLOUD, "GRAFANA_RW_TOKEN": ""}), {**CLOUD, "GRAFANA_RW_TOKEN": ""})


@pytest.mark.skipif(shutil.which("promtool") is None, reason="promtool not on PATH")
def test_rendered_config_passes_promtool(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "AGENT_DIR", tmp_path)
    path = agent.write_files(agent.render(CLOUD), CLOUD)
    out = subprocess.run(["promtool", "check", "config", "--agent", str(path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
