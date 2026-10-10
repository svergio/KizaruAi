import json
import logging
import os
import socket
import time
from pathlib import Path

import pytest

from core.bus.queue import JobStore
from core.bus.routing import resolve_worker_kind
from core.bus.worker_loop import run_assigned_once
from core.logging_setup import (
    JsonLineFormatter,
    LokiHandler,
    configure_logging,
    log_generation_event,
    reset_logging_for_tests,
)


@pytest.fixture(autouse=True)
def _restore_root_handlers():
    root = logging.getLogger()
    before = list(root.handlers)
    before_flag = getattr(configure_logging, "_configured", False)
    yield
    reset_logging_for_tests()
    root.handlers.clear()
    for h in before:
        root.addHandler(h)
    configure_logging._configured = before_flag


def test_json_formatter_fields():
    fmt = JsonLineFormatter("test-svc")
    record = logging.LogRecord(
        name="kizaru.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.job_id = 7
    data = json.loads(fmt.format(record))
    assert data["service"] == "test-svc" and data["job_id"] == 7 and data["message"] == "hello"


def test_configure_logging_json_lines(capsys):
    reset_logging_for_tests()
    configure_logging("web")
    logging.getLogger("kizaru.web").info("panel ok")
    err = capsys.readouterr().err.strip().splitlines()
    assert err
    assert json.loads(err[-1])["service"] == "web"


def test_generation_json_line_no_prompt(monkeypatch, job_store, capsys):
    seen: list[str] = []
    monkeypatch.setattr(
        "core.logging_setup._GEN_LOGGER.info",
        lambda msg: seen.append(msg),
    )
    job_id = job_store.enqueue("u1", "text", {"model": "gpt", "start": "SECRET_PROMPT_123"})
    job_store.route_one(resolve_worker_kind)
    runtime = {
        "checkpoint": "gpt.pt",
        "count_tokens": lambda text: max(1, len(str(text).split())),
    }

    def processor(payload):
        assert "SECRET_PROMPT_123" in payload["start"]
        return {"type": "message", "content": "ok one two"}

    assert run_assigned_once(job_store, "text_gpt", processor, runtime=runtime)
    assert len(seen) == 1
    data = json.loads(seen[0])
    assert data["worker_kind"] == "text_gpt"
    assert data["checkpoint"] == "gpt.pt"
    assert data["tokens_in"] >= 1 and data["tokens_out"] >= 1
    assert "SECRET_PROMPT_123" not in seen[0]
    assert "prompt" not in data


def test_no_loki_without_url(monkeypatch):
    monkeypatch.delenv("LOKI_URL", raising=False)
    reset_logging_for_tests()
    called = []

    class BoomSocket(socket.socket):
        def connect(self, address):
            called.append(address)
            raise AssertionError("network connect without LOKI_URL")

    monkeypatch.setattr(socket, "socket", BoomSocket)
    configure_logging("worker", worker_kind="mood")
    logging.getLogger("kizaru.worker").info("hello")
    assert called == []


def test_loki_down_fast_and_bounded(monkeypatch):
    monkeypatch.setenv("LOKI_URL", "http://127.0.0.1:9/loki")
    reset_logging_for_tests()
    t0 = time.monotonic()
    configure_logging("web")
    log = logging.getLogger("kizaru.web")
    for _ in range(300):
        log.info("flood")
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0
    loki = next(h for h in logging.getLogger().handlers if isinstance(h, LokiHandler))
    assert loki._queue.qsize() <= loki.max_queue


def test_no_print_in_worker_sources():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "core" / "bus" / "worker_loop.py",
        root / "app" / "orchestrator" / "__main__.py",
    ]
    paths.extend(sorted((root / "app" / "workers").glob("*.py")))
    for path in paths:
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "print(" not in text, f"print() found in {path}"


def test_discord_no_duplicate_ready_print():
    text = (Path(__file__).resolve().parents[1] / "app" / "discord" / "__main__.py").read_text(
        encoding="utf-8"
    )
    assert "print(" not in text
    assert "basicConfig" not in text
