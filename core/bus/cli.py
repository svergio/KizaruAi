import argparse
from pathlib import Path

from core.bus.worker_loop import DEFAULT_DB, POLL_SEC


def base_parser(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite bus")
    parser.add_argument("--poll", type=float, default=POLL_SEC, help="Idle poll seconds")
    return parser
