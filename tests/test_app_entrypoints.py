"""Деплой: app.orchestrator, app.workers.*, app.cli.*, app.discord, start_all.ps1.
Проверяем импорт, --help и понятные ошибки без нужного окружения. Ничего не запускается надолго."""
import importlib
import re

import pytest

from conftest import ROOT, run_py

APP_MODULES = [
    "app.orchestrator.__main__",
    "app.web.__main__",
    "app.discord.__main__",
    "app.cli.generate",
    "app.cli.image",
    "app.workers.lstm",
    "app.workers.lstm_word",
    "app.workers.transformer",
    "app.workers.gpt",
    "app.workers.deepseek",
    "app.workers.bert_gen",
    "app.workers.mood",
    "app.workers.still",
    "app.workers.video",
]


@pytest.mark.parametrize("name", APP_MODULES)
def test_app_module_imports(name):
    module = importlib.import_module(name)
    assert callable(getattr(module, "main", None)), f"{name}: нет main()"


HELP_TARGETS = [
    ["-m", "app.orchestrator"],
    ["-m", "app.web"],
    ["-m", "app.discord"],
    ["-m", "app.cli.generate"],
    ["-m", "app.cli.image"],
    ["-m", "app.workers.lstm"],
    ["-m", "app.workers.mood"],
    ["-m", "app.workers.still"],
]


@pytest.mark.integration
@pytest.mark.parametrize("args", HELP_TARGETS, ids=[a[1] for a in HELP_TARGETS])
def test_help_exits_zero(args):
    proc = run_py([*args, "--help"], timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "usage" in proc.stdout.lower()


@pytest.mark.integration
@pytest.mark.parametrize("worker", ["lstm", "lstm_word", "gpt", "transformer", "deepseek", "bert_gen", "mood"])
def test_worker_without_model_fails_clearly(worker, tmp_path):
    missing = tmp_path / "missing.pt"
    proc = run_py(["-m", f"app.workers.{worker}", str(missing), "--db", str(tmp_path / "bot.db")], timeout=120)
    assert proc.returncode != 0
    assert "not found" in proc.stderr.lower() or "missing.pt" in proc.stderr, proc.stderr[-2000:]


@pytest.mark.integration
def test_discord_without_token_exits(tmp_path):
    proc = run_py(["-m", "app.discord", "--db", str(tmp_path / "bot.db")], timeout=120)
    assert proc.returncode == 1
    assert "DISCORD_TOKEN" in proc.stderr


@pytest.mark.integration
def test_cli_generate_without_orchestrator(tmp_path):
    proc = run_py(["-m", "app.cli.generate", "--arch", "gpt", "--db", str(tmp_path / "bot.db")], timeout=120)
    assert proc.returncode != 0
    assert "orchestrator" in proc.stderr


def test_start_all_ps1_static():
    text = (ROOT / "start_all.ps1").read_text(encoding="utf-8")
    assert "Get-Command python" in text
    assert "-m app.orchestrator" in text
    for module in ("app.workers.lstm", "app.workers.gpt", "app.workers.mood", "app.workers.still", "app.discord"):
        assert f"-m {module}" in text, module
    assert not re.search(r"(?i)[a-z]:[\\/]+users[\\/]", text)
    # Токен читается из .env, но не печатается.
    assert "Write-Host $token" not in text
