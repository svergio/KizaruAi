import base64
import json
import os
import sys
import time
from pathlib import Path

from core.bus.queue import (
    ActiveJobError,
    JobStore,
    PayloadTooLargeError,
    QueueFullError,
)
from core.bus.routing import resolve_worker_kind

DEFAULT_DB = Path("data/bot.db")
POLL_SEC = 0.25


def cli_user_id() -> str:
    return f"cli:{os.getpid()}"


def orchestrator_online(store: JobStore) -> bool:
    return store.get_meta("orchestrator") is not None


def require_broker(store: JobStore, worker_kind: str) -> None:
    if not orchestrator_online(store):
        raise SystemExit(
            "orchestrator не запущен (python orchestrator.py --db data/bot.db)"
        )
    if not store.worker_online(worker_kind):
        raise SystemExit(f"воркер {worker_kind} не запущен")


def poll_until_done_sync(
    store: JobStore, job_id: int, interval: float = POLL_SEC
) -> dict:
    while True:
        row = store.get_job(job_id)
        if row is None:
            raise SystemExit(f"job {job_id} not found")
        status = row["status"]
        if status == "done":
            if not row["result"]:
                raise SystemExit(f"job {job_id} finished without result")
            return json.loads(row["result"])
        if status == "error":
            raise SystemExit(row["error"] or f"job {job_id} failed")
        time.sleep(interval)


def enqueue_and_wait(
    store: JobStore,
    kind: str,
    payload: dict,
    *,
    user_id: str | None = None,
    interval: float = POLL_SEC,
) -> dict:
    worker_kind = resolve_worker_kind(kind, payload)
    require_broker(store, worker_kind)
    uid = user_id or cli_user_id()
    try:
        job_id = store.enqueue(uid, kind, payload)
    except ActiveJobError as exc:
        raise SystemExit(
            f"у пользователя {uid} уже есть активная заявка (job {exc.args[0]})"
        ) from exc
    except PayloadTooLargeError:
        raise SystemExit("payload слишком большой для очереди") from None
    except QueueFullError:
        raise SystemExit("очередь переполнена") from None
    print(f"job {job_id}", file=sys.stderr)
    return poll_until_done_sync(store, job_id, interval=interval)


def _print_text(text: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


def emit_job_result(result: dict, out_path: Path | None = None) -> None:
    kind = result.get("type")
    if kind == "message":
        _print_text(result["content"])
        return
    if kind == "chunks":
        _print_text("\n".join(result["chunks"]))
        return
    if kind == "file":
        raw = base64.b64decode(result["data_b64"])
        path = Path(out_path) if out_path else Path(result.get("filename") or "out.bin")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        print(str(path.resolve()))
        return
    raise SystemExit(f"unknown worker result type: {kind!r}")
