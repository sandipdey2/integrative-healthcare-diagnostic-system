"""Run the FastAPI service (api/main.py) inside the Streamlit process, for hosts that run only `streamlit run`
(Streamlit Community Cloud). The UI then uses the same REST route as Docker/Azure (`POST /assess`), and the API's
request metrics exist.

The API is served by uvicorn on a daemon thread, bound to 127.0.0.1 on a free port (never exposed). It reuses the
Streamlit process's already-built System (model, GAN, knowledge base, MCP stdio tool backend), so start-up costs
no second training run and no second MCP child. Gate: IHDS_EMBED_API (default on when IHDS_API_URL is unset;
scripts/start_all.sh sets it to 0 because the container runs the API itself).
"""
from __future__ import annotations

import os
import socket
import threading
import time
import urllib.request


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    if env.get("IHDS_API_URL"):
        return False
    return str(env.get("IHDS_EMBED_API", "1")).strip().lower() not in ("0", "false", "no", "off")


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


class EmbeddedAPI:
    def __init__(self, system, host: str = "127.0.0.1", port: int | None = None, timeout: float = 30.0):
        import uvicorn

        from api import main as api_main

        api_main.STATE["system"] = system  # share the app's System (lifespan's get_system() returns it)
        self.port = port or free_port(host)
        self.url = f"http://{host}:{self.port}"
        cfg = uvicorn.Config(api_main.app, host=host, port=self.port, log_level="warning", access_log=False,
                             lifespan="on")
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, name="ihds-embedded-api", daemon=True)
        t0 = time.time()
        self.thread.start()
        while True:
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if not self.thread.is_alive():
                raise RuntimeError("embedded API thread exited during start-up")
            if time.time() - t0 > timeout:
                self.stop()
                raise TimeoutError(f"embedded API /health not ready after {timeout:.0f}s")
            time.sleep(0.2)
        self.startup_s = round(time.time() - t0, 2)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)
