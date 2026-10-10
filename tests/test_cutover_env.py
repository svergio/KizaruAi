"""var/deepseek_cutover_15k_rl.py: без KIZARU_SFT_MLFLOW_RUN выходит с кодом 2 до chdir, PID-файла и MLflow."""
import os
import subprocess
import sys
import time
import types

import pytest

from conftest import ROOT, load_module_from_path

SCRIPT = ROOT / "var" / "deepseek_cutover_15k_rl.py"
ENV = "KIZARU_SFT_MLFLOW_RUN"


class _Touched(Exception):
    pass


class _RecordingModule(types.ModuleType):
    """Фейковый mlflow: любое обращение к атрибуту записывается."""

    def __init__(self, name, log):
        super().__init__(name)
        self._log = log

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        self._log.append(f"{self.__name__}.{item}")
        raise _Touched(f"mlflow touched: {self.__name__}.{item}")


@pytest.fixture
def guarded(monkeypatch):
    """Запрещаем всё, что может запустить процессы, обучение или ожидание."""
    hits = []

    def forbid(name):
        def _f(*a, **k):
            hits.append(name)
            raise _Touched(f"{name} called: {a[:1]}")
        return _f

    for name in ("run", "Popen", "check_output", "check_call", "call"):
        monkeypatch.setattr(subprocess, name, forbid(f"subprocess.{name}"))
    monkeypatch.setattr(os, "chdir", forbid("os.chdir"))
    monkeypatch.setattr(os, "kill", forbid("os.kill"))
    monkeypatch.setattr(time, "sleep", forbid("time.sleep"))

    mlflow_log = []
    ml = _RecordingModule("mlflow", mlflow_log)
    tracking = _RecordingModule("mlflow.tracking", mlflow_log)
    monkeypatch.setitem(sys.modules, "mlflow", ml)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking)
    return types.SimpleNamespace(hits=hits, mlflow=mlflow_log)


def _load_fresh(monkeypatch):
    name = "_kz_test_cutover_env_fresh"
    monkeypatch.delitem(sys.modules, name, raising=False)
    mod = load_module_from_path(SCRIPT, name)
    monkeypatch.setitem(sys.modules, name, mod)  # убрать после теста
    return mod


@pytest.mark.parametrize("value", [None, ""])
def test_cutover_without_sft_run_exits_2(value, monkeypatch, tmp_path, capsys, guarded):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    cwd_before = os.getcwd()

    mod = _load_fresh(monkeypatch)
    assert mod.MLFLOW_RUN_ID == ""                 # читается при импорте

    # PID-файлы и логи — в tmp_path, на случай регрессии реальный var/ не трогаем
    real_pid = mod.WATCHER_PID
    real_existed = real_pid.exists()
    real_mtime = real_pid.stat().st_mtime_ns if real_existed else None
    for attr in ("WATCHER_PID", "WATCHER_LOG", "PID_FILE", "WAITER_PID", "STATUS", "RL_STATUS",
                 "CUTOVER", "MLFLOW_META", "RL_LOG", "RL_ERR"):
        monkeypatch.setattr(mod, attr, tmp_path / getattr(mod, attr).name)

    try:
        code = mod.main()
    except SystemExit as e:
        code = e.code
    cap = capsys.readouterr()

    assert code == 2
    assert ENV in cap.err
    assert os.getcwd() == cwd_before
    assert not (tmp_path / real_pid.name).exists()
    assert list(tmp_path.iterdir()) == []          # вообще ничего не записано
    assert real_pid.exists() == real_existed
    if real_existed:
        assert real_pid.stat().st_mtime_ns == real_mtime
    assert guarded.mlflow == []                    # fake mlflow не тронут
    assert guarded.hits == []                      # ни процессов, ни chdir, ни ожидания


def test_cutover_reads_env_at_import(monkeypatch, guarded):
    monkeypatch.setenv(ENV, "abc123")
    mod = _load_fresh(monkeypatch)
    assert mod.MLFLOW_RUN_ID == "abc123"
    assert guarded.hits == [] and guarded.mlflow == []
