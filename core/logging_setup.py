import json
import logging
import os
import queue
import threading
from datetime import datetime, timezone

LOKI_QUEUE_MAX = 256


class JsonLineFormatter(logging.Formatter):
    def __init__(self, service: str):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("worker_kind", "job_id"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info and record.exc_info[1]:
            payload["error"] = str(record.exc_info[1])
        return json.dumps(payload, ensure_ascii=False)


class LokiHandler(logging.Handler):
    def __init__(self, url: str, max_queue: int = LOKI_QUEUE_MAX):
        super().__init__()
        self.url = url.rstrip("/")
        self.max_queue = max_queue
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="loki-sender", daemon=True)
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._queue.put_nowait(msg)
        except queue.Full:
            pass
        except Exception:
            pass

    def _worker(self) -> None:
        import urllib.error
        import urllib.request

        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            req = urllib.request.Request(
                self.url,
                data=item.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=0.8):
                    pass
            except (urllib.error.URLError, TimeoutError, OSError):
                pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        super().close()


def configure_logging(service: str, *, worker_kind: str | None = None) -> logging.Logger:
    root = logging.getLogger()
    if getattr(configure_logging, "_configured", False):
        log = logging.getLogger(f"kizaru.{service}")
        return log
    root.handlers.clear()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLineFormatter(service))
    root.addHandler(handler)
    loki_url = (os.environ.get("LOKI_URL") or "").strip()
    if loki_url:
        loki = LokiHandler(loki_url)
        loki.setFormatter(JsonLineFormatter(service))
        root.addHandler(loki)
    configure_logging._configured = True
    if worker_kind:
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.worker_kind = worker_kind
            return record

        logging.setLogRecordFactory(record_factory)
    return logging.getLogger(f"kizaru.{service}")


def reset_logging_for_tests() -> None:
    configure_logging._configured = False
    root = logging.getLogger()
    for handler in list(root.handlers):
        try:
            handler.close()
        except Exception:
            pass
    root.handlers.clear()


_GEN_LOGGER = logging.getLogger("kizaru.generation")


def _ensure_generation_logger() -> None:
    if _GEN_LOGGER.handlers:
        return
    _GEN_LOGGER.propagate = False
    _GEN_LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    _GEN_LOGGER.addHandler(handler)


def log_generation_event(
    *,
    request_id: int,
    user_id: int | None,
    command: str,
    model: str,
    checkpoint: str,
    duration_ms: int,
    tokens_in: int,
    tokens_out: int,
    status: str,
    error: str | None = None,
    worker_kind: str | None = None,
) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "user_id": user_id,
        "command": command,
        "model": model,
        "checkpoint": checkpoint,
        "duration_ms": duration_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "status": status,
        "error": error,
    }
    if worker_kind:
        payload["worker_kind"] = worker_kind
    _ensure_generation_logger()
    _GEN_LOGGER.info(json.dumps(payload, ensure_ascii=False))
