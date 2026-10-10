import argparse
import atexit
import json
import logging
import os
import time
from pathlib import Path

from core.bus.queue import JobStore
from core.logging_setup import configure_logging
from core.bus.routing import resolve_worker_kind

DEFAULT_DB = Path("data/bot.db")
POLL_SEC = 0.2
HEARTBEAT_SEC = 5.0
RETENTION_EVERY_SEC = 300.0


def parse_args():
    parser = argparse.ArgumentParser(description="Route queued jobs to worker inboxes (no generation)")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite bus (default: data/bot.db)")
    parser.add_argument("--poll", type=float, default=POLL_SEC, help="Idle sleep seconds")
    return parser.parse_args()


def main():
    configure_logging("orchestrator")
    args = parse_args()
    store = JobStore(args.db, recover_running=False)
    pid = os.getpid()

    def _heartbeat():
        store.set_meta(
            "orchestrator",
            json.dumps({"role": "router", "pid": pid}, ensure_ascii=False),
        )

    def _clear():
        raw = store.get_meta("orchestrator")
        if not raw:
            return
        try:
            meta = json.loads(raw)
        except json.JSONDecodeError:
            store.clear_meta("orchestrator")
            return
        if isinstance(meta, dict) and int(meta.get("pid") or 0) == pid:
            store.clear_meta("orchestrator")

    atexit.register(_clear)
    _heartbeat()
    log = logging.getLogger("kizaru.orchestrator")
    log.info("orchestrator ready db=%s", args.db)
    last_hb = 0.0
    last_retention = 0.0
    while True:
        now = time.monotonic()
        if now - last_hb >= HEARTBEAT_SEC:
            _heartbeat()
            last_hb = now
        if now - last_retention >= RETENTION_EVERY_SEC:
            try:
                store.retention_cleanup()
            except Exception:
                log.exception("retention_cleanup failed")
            last_retention = now
        store.reap_offline_assigned()
        if store.route_one(resolve_worker_kind):
            continue
        time.sleep(max(0.05, float(args.poll)))


if __name__ == "__main__":
    main()
