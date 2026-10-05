import atexit
import json
import os
import time
from pathlib import Path

from core.bus.queue import JobStore

DEFAULT_DB = Path("data/bot.db")
POLL_SEC = 0.35


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


def run_assigned_once(store: JobStore, worker_kind: str, processor) -> bool:
    job = store.claim_assigned(worker_kind)
    if job is None:
        return False
    job_id = int(job["id"])
    try:
        payload = json.loads(job["payload"])
    except json.JSONDecodeError:
        store.fail(job_id, "bad payload json")
        return True
    try:
        result = processor(payload)
        store.complete(job_id, result)
    except Exception as exc:
        store.fail(job_id, str(exc))
    return True


def run_worker_loop(worker_kind: str, processor, db_path: str | Path, poll: float = POLL_SEC):
    store = JobStore(db_path, recover_running=True, worker_kind=worker_kind)
    register_worker(store, worker_kind)
    print(f"worker {worker_kind} ready db={db_path} pid={os.getpid()}")
    while True:
        if run_assigned_once(store, worker_kind, processor):
            continue
        time.sleep(max(0.05, float(poll)))
