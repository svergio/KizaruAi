from __future__ import annotations

import os
from typing import Any

DEFAULT_URI = "http://127.0.0.1:5000"

_active = False
_run = None


def enabled() -> bool:
    if os.environ.get("KIZARU_MLFLOW", "1").strip() in ("0", "false", "no", "off"):
        return False
    return True


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI).rstrip("/")


def _client():
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    return mlflow


def start_run(
    experiment: str,
    run_name: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    run_id: str | None = None,
) -> str | None:
    global _active, _run
    if not enabled():
        return None
    try:
        mlflow = _client()
        mlflow.set_experiment(experiment)
        existing = run_id or os.environ.get("MLFLOW_RUN_ID") or None
        if existing:
            _run = mlflow.start_run(run_id=existing)
        else:
            _run = mlflow.start_run(run_name=run_name)
        _active = True
        if tags:
            for k, v in tags.items():
                if v is None:
                    continue
                mlflow.set_tag(str(k), str(v))
        if params:
            flat = {}
            for k, v in params.items():
                if v is None:
                    continue
                flat[str(k)] = str(v) if not isinstance(v, (int, float, bool)) else v
            mlflow.log_params(flat)
        return _run.info.run_id
    except Exception as exc:
        print(f"mlflow_warn start_run failed: {exc}", flush=True)
        _active = False
        _run = None
        return None


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    if not _active:
        return
    try:
        mlflow = _client()
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if not clean:
            return
        mlflow.log_metrics(clean, step=step)
    except Exception as exc:
        print(f"mlflow_warn log_metrics failed: {exc}", flush=True)


def set_tag(key: str, value: str) -> None:
    if not _active:
        return
    try:
        _client().set_tag(key, value)
    except Exception as exc:
        print(f"mlflow_warn set_tag failed: {exc}", flush=True)


def end_run(status: str = "FINISHED") -> None:
    global _active, _run
    if not _active:
        return
    try:
        _client().end_run(status=status)
    except Exception as exc:
        print(f"mlflow_warn end_run failed: {exc}", flush=True)
    _active = False
    _run = None


def active_run_id() -> str | None:
    if not _active or _run is None:
        return None
    try:
        return _run.info.run_id
    except Exception:
        return None
