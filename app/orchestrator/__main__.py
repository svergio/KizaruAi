import argparse
import json
import time
from pathlib import Path

from core.bus.queue import JobStore
from core.bus.routing import resolve_worker_kind

DEFAULT_DB = Path("data/bot.db")
POLL_SEC = 0.2


def parse_args():
    parser = argparse.ArgumentParser(description="Route queued jobs to worker inboxes (no generation)")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite bus (default: data/bot.db)")
    parser.add_argument("--poll", type=float, default=POLL_SEC, help="Idle sleep seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    store = JobStore(args.db, recover_running=False)
    store.set_meta("orchestrator", json.dumps({"role": "router"}, ensure_ascii=False))
    print(f"orchestrator ready db={args.db}")
    while True:
        if store.route_one(resolve_worker_kind):
            continue
        time.sleep(max(0.05, float(args.poll)))


if __name__ == "__main__":
    main()
