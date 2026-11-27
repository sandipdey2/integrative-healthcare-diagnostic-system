"""Serve the MLOps flow as a scheduled Prefect deployment on this machine.

    prefect server start              # terminal 1: local Prefect API + UI at http://127.0.0.1:4200
    python -m ihds.mlops.deploy       # terminal 2: registers the deployment and keeps running to fire the schedule

This uses Prefect 3 `flow.serve()`: the process registers a deployment with the API and runs each scheduled or
triggered flow run in a subprocess on this machine. There is no work pool, no worker, no Docker and no Prefect
Cloud. The schedule fires only while this process is running. Defaults come from the `deploy:` section of
params.yaml; the CLI flags override them.

    python -m ihds.mlops.deploy --run-now      # also trigger one run right after registering, then keep serving
    python -m ihds.mlops.deploy --dry-run      # print the deployment (name, schedule, parameters, tags) and exit
"""
from __future__ import annotations

import argparse
import json

from prefect.flows import EntrypointType
from prefect.schedules import Cron

from .common import load_params
from .flow import ihds_mlops_pipeline


def build_deployment(name: str | None = None, cron: str | None = None, timezone: str | None = None,
                     parameters: dict | None = None, tags: list[str] | None = None, paused: bool | None = None):
    """Build (but do not register or start) the RunnerDeployment. Missing arguments come from params.yaml."""
    cfg = load_params("deploy")
    return ihds_mlops_pipeline.to_deployment(
        name=name or cfg["name"],
        schedule=Cron(cron or cfg["cron"], timezone=timezone or cfg["timezone"]),
        parameters=parameters if parameters is not None else dict(cfg.get("parameters") or {}),
        tags=list(tags if tags is not None else cfg.get("tags") or []),
        description=cfg.get("description"),
        paused=cfg.get("paused", False) if paused is None else paused,
        # module path (not file path): ihds/mlops/flow.py uses relative imports
        entrypoint_type=EntrypointType.MODULE_PATH,
    )


def describe(dep) -> dict:
    sched = dep.schedules[0].schedule
    return {"deployment": f"{ihds_mlops_pipeline.name}/{dep.name}", "cron": sched.cron, "timezone": sched.timezone,
            "parameters": dep.parameters, "tags": list(dep.tags), "paused": dep.paused}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="deployment name (default: deploy.name in params.yaml)")
    ap.add_argument("--cron", help="cron expression (default: deploy.cron in params.yaml)")
    ap.add_argument("--timezone", help="IANA timezone for the cron (default: deploy.timezone in params.yaml)")
    ap.add_argument("--no-feast", action="store_true", help="pass with_feast=False to the flow")
    ap.add_argument("--paused", action="store_true", help="register the schedule paused (manual runs only)")
    ap.add_argument("--run-now", action="store_true", help="trigger one run right after registering, keep serving")
    ap.add_argument("--dry-run", action="store_true", help="print the deployment and exit; nothing is registered")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    params = {"with_feast": False} if args.no_feast else None
    dep = build_deployment(args.name, args.cron, args.timezone, params, paused=True if args.paused else None)
    info = describe(dep)
    print(json.dumps(info, indent=2))
    if args.dry_run:
        return 0
    if args.run_now:
        _trigger_in_background(info["deployment"])
    from prefect import serve
    serve(dep, pause_on_shutdown=False)  # blocks; Ctrl-C stops it (the registered schedule then stops firing runs)
    return 0


def _trigger_in_background(deployment_name: str, delay: float = 5.0) -> None:
    """Create one flow run for the deployment shortly after serve() has registered it."""
    import threading

    def go():
        import time

        from prefect.deployments import run_deployment
        time.sleep(delay)
        fr = run_deployment(deployment_name, timeout=0)
        print(f"triggered flow run {fr.id} ({fr.name})")

    threading.Thread(target=go, daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
