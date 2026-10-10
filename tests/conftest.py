"""Общие фикстуры KizaruAi.

Правила набора:
* никакой внешней сети: любой connect() не на loopback падает;
* никакой видеокарты: torch.cuda.is_available() -> False;
* MLflow выключен по умолчанию (KIZARU_MLFLOW=0), для проверок есть заглушки
  fake_mlflow / broken_mlflow;
* LLM-сервер (llama-swap, OpenAI-совместимый API) подменяется fake_llm_server;
* все файлы пишутся только в tmp_path.
"""
from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.dont_write_bytecode = True  # не мусорим __pycache__ в проекте

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


# --------------------------------------------------------------------------- сеть


@pytest.fixture(autouse=True)
def _no_external_network(monkeypatch):
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _check(address):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in LOCAL_HOSTS:
            raise RuntimeError(f"тесты не ходят в сеть: попытка подключения к {address!r}")

    def guarded_connect(self, address):
        _check(address)
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        _check(address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


# ------------------------------------------------------------------ CPU и окружение


@pytest.fixture(autouse=True)
def _cpu_only_and_clean_env(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("KIZARU_MLFLOW", "0")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)

    from training import mlflow_track

    monkeypatch.setattr(mlflow_track, "_active", False)
    monkeypatch.setattr(mlflow_track, "_run", None)
    yield


# ---------------------------------------------------------------------- MLflow


class FakeMlflow(types.ModuleType):
    """Заглушка модуля mlflow: пишет все вызовы, по желанию падает."""

    def __init__(self, fail_on=()):
        super().__init__("mlflow")
        self.fail_on = set(fail_on)
        self.calls: list[tuple] = []
        self.params: dict = {}
        self.tags: dict = {}
        self.metrics: list[tuple[dict, int | None]] = []
        self.ended: list[str] = []
        self.experiments: list[str] = []
        self.tracking_uris: list[str] = []
        self.started: list[dict] = []

    def _hit(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        if name in self.fail_on or "*" in self.fail_on:
            raise ConnectionError(f"MLflow недоступен (заглушка, {name})")

    def set_tracking_uri(self, uri):
        self.tracking_uris.append(uri)
        self.calls.append(("set_tracking_uri", (uri,), {}))

    def set_experiment(self, name):
        self._hit("set_experiment", name)
        self.experiments.append(name)

    def start_run(self, run_id=None, run_name=None):
        self._hit("start_run", run_id=run_id, run_name=run_name)
        self.started.append({"run_id": run_id, "run_name": run_name})
        rid = run_id or "fake-run-1"
        return types.SimpleNamespace(info=types.SimpleNamespace(run_id=rid))

    def set_tag(self, key, value):
        self._hit("set_tag", key, value)
        self.tags[key] = value

    def log_params(self, params):
        self._hit("log_params", params)
        self.params.update(params)

    def log_metrics(self, metrics, step=None):
        self._hit("log_metrics", metrics, step=step)
        self.metrics.append((dict(metrics), step))

    def end_run(self, status="FINISHED"):
        self._hit("end_run", status=status)
        self.ended.append(status)

    def metric_names(self) -> set[str]:
        return {k for m, _ in self.metrics for k in m}


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("KIZARU_MLFLOW", "1")
    return fake


@pytest.fixture
def broken_mlflow(monkeypatch):
    """MLflow-сервер на 127.0.0.1:5000 не поднят: любой вызов падает."""
    fake = FakeMlflow(fail_on={"*"})
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("KIZARU_MLFLOW", "1")
    return fake


# ------------------------------------------------------- LLM-сервер (llama-swap)


class _FakeLlmState:
    def __init__(self):
        self.requests: list[dict] = []
        self.reply = "Я на районе, ночь не спит\nДеньги в кармане, рифма звенит"
        self.model = "qwen-fake"


def _make_llm_handler(state: _FakeLlmState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": state.model, "object": "model"}]})
                return
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._send(400, {"error": {"message": "bad json"}})
                return
            state.requests.append({"path": self.path, "body": body})
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": {"message": "not found"}})
                return
            self._send(
                200,
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model") or state.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": state.reply},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

    return Handler


@pytest.fixture
def fake_llm_server():
    """Поддельный OpenAI-совместимый сервер (как llama-swap) на 127.0.0.1:<свободный порт>."""
    state = _FakeLlmState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_llm_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    state.base_url = f"http://{host}:{port}/v1"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


# ------------------------------------------------------------------- помощники


def load_module_from_path(path: Path, name: str | None = None):
    """Импорт скрипта по пути (для training/scripts и var/, где нет __init__.py)."""
    path = Path(path)
    mod_name = name or f"_kz_test_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


def subprocess_env(**extra) -> dict:
    env = dict(os.environ)
    env.pop("DISCORD_TOKEN", None)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["KIZARU_MLFLOW"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run_py(args, cwd=None, timeout=180, **env_extra) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd or ROOT),
        env=subprocess_env(**env_extra),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture
def job_store(tmp_path):
    from core.bus.queue import JobStore

    return JobStore(tmp_path / "bot.db")


@pytest.fixture
def tiny_corpus(tmp_path) -> Path:
    lines = [
        "Я на районе, ночь опять не спит",
        "Деньги в кармане, сердце не болит",
        "Мы идём вперёд, и город нам кричит",
        "Каждый новый день как старый хит",
        "Кайф и огонь, а музыка звучит",
        "Тихо на душе, никто не говорит",
    ]
    text = "\n".join(lines * 8) + "\n"
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def small_png_bytes() -> bytes:
    from io import BytesIO

    import numpy as np
    from PIL import Image

    y, x = np.mgrid[0:64, 0:64]
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[..., 0] = (x * 4).clip(0, 255)
    arr[..., 1] = (y * 4).clip(0, 255)
    arr[16:48, 16:48, :] = 255
    buf = BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def animated_gif_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image

    frames = [Image.new("RGB", (16, 16), (0, 0, 0)), Image.new("RGB", (16, 16), (255, 255, 255))]
    buf = BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=50, loop=0)
    return buf.getvalue()
