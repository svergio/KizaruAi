"""training/mlflow_track.py: MLflow необязателен. Без сервера обучение не падает, с заглушкой метрики пишутся."""
import socket
import time

import pytest

from training import mlflow_track


def test_enabled_flag(monkeypatch):
    for value in ("0", "false", "no", "off", " off "):
        monkeypatch.setenv("KIZARU_MLFLOW", value)
        assert mlflow_track.enabled() is False
    monkeypatch.setenv("KIZARU_MLFLOW", "1")
    assert mlflow_track.enabled() is True
    monkeypatch.delenv("KIZARU_MLFLOW")
    assert mlflow_track.enabled() is True


def test_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    assert mlflow_track.tracking_uri() == mlflow_track.DEFAULT_URI == "http://127.0.0.1:5000"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001/")
    assert mlflow_track.tracking_uri() == "http://127.0.0.1:5001"


def test_disabled_does_nothing(fake_mlflow, monkeypatch):
    monkeypatch.setenv("KIZARU_MLFLOW", "0")
    assert mlflow_track.start_run("exp", run_name="r") is None
    mlflow_track.log_metrics({"loss": 1.0})
    mlflow_track.set_tag("a", "b")
    mlflow_track.end_run()
    assert fake_mlflow.calls == []
    assert mlflow_track.active_run_id() is None


def test_server_down_does_not_break_training(broken_mlflow, capsys):
    assert mlflow_track.start_run("exp", run_name="r", params={"lr": 0.1}) is None
    mlflow_track.log_metrics({"loss": 1.0}, step=1)
    mlflow_track.set_tag("gate", "x")
    mlflow_track.end_run(status="FAILED")
    assert "mlflow_warn start_run failed" in capsys.readouterr().out
    assert mlflow_track.active_run_id() is None


def test_full_run_with_fake_server(fake_mlflow):
    run_id = mlflow_track.start_run(
        "kizaru-lm",
        run_name="gpt_char",
        params={"lr": 0.001, "steps": 5, "out": None, "path": object.__name__, "flag": True},
        tags={"project": "kizaruai", "skip": None},
    )
    assert run_id == "fake-run-1" == mlflow_track.active_run_id()
    assert fake_mlflow.experiments == ["kizaru-lm"]
    assert fake_mlflow.tracking_uris[-1] == mlflow_track.tracking_uri()
    assert fake_mlflow.params == {"lr": 0.001, "steps": 5, "path": "object", "flag": True}
    assert fake_mlflow.tags == {"project": "kizaruai"}
    mlflow_track.log_metrics({"loss": 2, "skip": None}, step=3)
    mlflow_track.log_metrics({"only_none": None})
    mlflow_track.set_tag("gate", "pass")
    assert fake_mlflow.metrics == [({"loss": 2.0}, 3)]
    assert fake_mlflow.tags["gate"] == "pass"
    mlflow_track.end_run(status="KILLED")
    assert fake_mlflow.ended == ["KILLED"]
    assert mlflow_track.active_run_id() is None
    mlflow_track.log_metrics({"loss": 1.0})
    assert len(fake_mlflow.metrics) == 1


def test_resume_run_from_env(fake_mlflow, monkeypatch):
    monkeypatch.setenv("MLFLOW_RUN_ID", "abc123")
    assert mlflow_track.start_run("exp") == "abc123"
    assert fake_mlflow.started[-1]["run_id"] == "abc123"
    mlflow_track.end_run()


def test_log_metrics_failure_is_swallowed(fake_mlflow, capsys):
    mlflow_track.start_run("exp")
    fake_mlflow.fail_on = {"log_metrics", "set_tag", "end_run"}
    mlflow_track.log_metrics({"loss": 1.0})
    mlflow_track.set_tag("a", "b")
    mlflow_track.end_run()
    out = capsys.readouterr().out
    assert "log_metrics failed" in out and "set_tag failed" in out and "end_run failed" in out
    assert mlflow_track.active_run_id() is None


@pytest.mark.integration
def test_real_mlflow_unreachable_server(monkeypatch, capsys):
    pytest.importorskip("mlflow")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    monkeypatch.setenv("KIZARU_MLFLOW", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
    started = time.monotonic()
    assert mlflow_track.start_run("kizaru-test", run_name="unreachable") is None
    assert time.monotonic() - started < 60
    mlflow_track.log_metrics({"loss": 1.0})
    mlflow_track.end_run()
    assert "mlflow_warn start_run failed" in capsys.readouterr().out
