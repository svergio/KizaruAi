import atexit
import json
import logging
import os
import threading
import time
from pathlib import Path

import torch

from core.bus.cancel_flag import GenerationCancelled, set_cancel_event
from core.bus.queue import CANCEL_POLL_SEC, LEASE_RENEW_SEC, JobStore
from core.logging_setup import configure_logging, log_generation_event

DEFAULT_DB = Path("data/bot.db")
POLL_SEC = 0.35
HEARTBEAT_SEC = 5.0

_current = {"job_id": None, "event": None}
_current_lock = threading.Lock()


def register_worker(store: JobStore, worker_kind: str):
    key = f"worker:{worker_kind}"
    store.set_meta(
        key,
        json.dumps({"pid": os.getpid(), "kind": worker_kind}, ensure_ascii=False),
    )

    def _unregister():
        raw = store.get_meta(key)
        if not raw:
            return
        try:
            meta = json.loads(raw)
        except json.JSONDecodeError:
            store.clear_meta(key)
            return
        if isinstance(meta, dict) and int(meta.get("pid") or 0) == os.getpid():
            store.clear_meta(key)

    atexit.register(_unregister)


def _lease_daemon(store: JobStore, worker_kind: str, stop: threading.Event):
    pid = os.getpid()
    key = f"worker:{worker_kind}"
    last_hb = 0.0
    while not stop.wait(min(LEASE_RENEW_SEC, CANCEL_POLL_SEC)):
        now = time.monotonic()
        if now - last_hb >= HEARTBEAT_SEC:
            store.set_meta(
                key,
                json.dumps({"pid": pid, "kind": worker_kind}, ensure_ascii=False),
            )
            last_hb = now
        with _current_lock:
            job_id = _current["job_id"]
            event = _current["event"]
        if job_id is not None and event is not None:
            status = store.job_status(int(job_id))
            if status == "cancelled":
                event.set()
                continue
        result = store.renew_running_lease(worker_kind, pid)
        if result == "timeout" and event is not None:
            event.set()


def _iter_runtime_models(runtime: dict):
    model = runtime.get("lstm_model")
    if model is not None and hasattr(model, "to"):
        yield model
    for key in ("gpt", "transformer", "deepseek", "bert_gen", "mood"):
        val = runtime.get(key)
        if val is None:
            continue
        if isinstance(val, (tuple, list)) and val:
            head = val[0]
            if hasattr(head, "to"):
                yield head
        elif hasattr(val, "to"):
            yield val


def _release_gpu(runtime: dict | None) -> None:
    """Move inference weights to CPU and free CUDA cache before terminal status."""
    runtime = runtime or {}
    hook = runtime.get("release_gpu")
    if callable(hook):
        hook()
        return
    for model in _iter_runtime_models(runtime):
        try:
            model.to("cpu")
        except Exception:
            pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _make_counter(runtime: dict):
    tok = runtime.get("hf_tokenizer")
    if tok is not None and hasattr(tok, "encode"):
        def hf_count(text: str) -> int:
            ids = tok.encode(str(text))
            return max(1, len(ids))

        return hf_count
    counter = runtime.get("count_tokens")
    if callable(counter):
        return lambda text: max(1, int(counter(str(text))))
    return lambda text: max(1, len(str(text)))


def _token_counts(payload: dict, result, runtime: dict | None) -> tuple[int, int]:
    runtime = runtime or {}
    start = str(payload.get("start") or payload.get("source") or payload.get("text") or "")
    counter = _make_counter(runtime)
    tin = int(counter(start))
    tout = 0
    if isinstance(result, dict):
        if result.get("type") == "message":
            tout = int(counter(str(result.get("content") or "")))
        elif result.get("type") == "chunks":
            tout = sum(int(counter(str(c))) for c in (result.get("chunks") or []))
    return tin, tout


def run_assigned_once(store: JobStore, worker_kind: str, processor, runtime: dict | None = None) -> bool:
    job = store.claim_assigned(worker_kind)
    if job is None:
        return False
    job_id = int(job["id"])
    pid = os.getpid()
    started = time.monotonic()
    cancel_event = threading.Event()
    with _current_lock:
        _current["job_id"] = job_id
        _current["event"] = cancel_event
    set_cancel_event(cancel_event)
    payload = None
    outcome = None
    try:
        try:
            payload = json.loads(job["payload"])
        except json.JSONDecodeError:
            outcome = ("bad_json", None)
        else:
            try:
                result = processor(payload)
                outcome = ("done", result)
            except GenerationCancelled:
                outcome = ("cancelled", None)
            except torch.cuda.OutOfMemoryError:
                outcome = ("oom", None)
            except Exception as exc:
                outcome = ("error", str(exc))
    finally:
        set_cancel_event(None)
        with _current_lock:
            _current["job_id"] = None
            _current["event"] = None
        # Terminal status only after weights leave GPU / cache cleared.
        _release_gpu(runtime)
        if outcome is None:
            return True
        kind, detail = outcome
        if kind == "bad_json":
            store.fail(job_id, "bad payload json", pid)
            _gen_log(store, job, worker_kind, started, "error", "bad payload json", None, {}, runtime)
        elif kind == "done":
            store.complete(job_id, detail, pid)
            _gen_log(store, job, worker_kind, started, "done", None, detail, payload, runtime)
        elif kind == "cancelled":
            status = store.job_status(job_id)
            if status == "running":
                store.fail(job_id, "cancelled", pid)
                status = "cancelled"
            err = None
            if status == "error":
                row = store.get_job(job_id) or {}
                err = row.get("error") or "timeout"
            elif status == "cancelled":
                err = "cancelled"
            _gen_log(
                store,
                job,
                worker_kind,
                started,
                status if status in ("cancelled", "error") else "cancelled",
                err,
                None,
                payload,
                runtime,
            )
        elif kind == "oom":
            store.fail(job_id, "OOM", pid)
            _gen_log(store, job, worker_kind, started, "error", "OOM", None, payload, runtime)
        else:
            store.fail(job_id, str(detail), pid)
            _gen_log(store, job, worker_kind, started, "error", str(detail), None, payload, runtime)
    return True


def _gen_log(store, job, worker_kind, started, status, error, result=None, payload=None, runtime=None):
    duration_ms = int((time.monotonic() - started) * 1000)
    kind = str(job.get("kind") or "")
    payload = payload or {}
    model = str(payload.get("model") or worker_kind)
    tin, tout = _token_counts(payload, result, runtime)
    checkpoint = str((runtime or {}).get("checkpoint") or worker_kind)
    user_id = store.user_id_for_job(int(job["id"]))
    log_generation_event(
        request_id=int(job["id"]),
        user_id=user_id,
        command=kind,
        model=model,
        checkpoint=checkpoint,
        duration_ms=duration_ms,
        tokens_in=tin,
        tokens_out=tout,
        status=status,
        error=error,
        worker_kind=worker_kind,
    )


def run_worker_loop(
    worker_kind: str,
    processor,
    db_path: str | Path,
    poll: float = POLL_SEC,
    runtime: dict | None = None,
):
    log = configure_logging("worker", worker_kind=worker_kind)
    store = JobStore(db_path, recover_running=True, worker_kind=worker_kind)
    register_worker(store, worker_kind)
    stop = threading.Event()
    threading.Thread(
        target=_lease_daemon,
        args=(store, worker_kind, stop),
        name=f"lease-{worker_kind}",
        daemon=True,
    ).start()
    log.info("worker %s ready db=%s pid=%s", worker_kind, db_path, os.getpid())
    try:
        while True:
            if run_assigned_once(store, worker_kind, processor, runtime=runtime):
                continue
            time.sleep(max(0.05, float(poll)))
    finally:
        stop.set()
