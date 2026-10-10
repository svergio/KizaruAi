"""app.web (админка Flask): test_client, НЕ порт 8765. Процессы, GPU и чекпоинты подменены."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

import app.web.queries as queries
from app.web import create_app
from core.bus.routing import resolve_worker_kind


def _web_ns():
    return SimpleNamespace(
        ROOT=queries.ROOT,
        DEFAULT_HOST=queries.DEFAULT_HOST,
        DEFAULT_PORT=queries.DEFAULT_PORT,
        create_app=create_app,
        _collect_dashboard=queries._collect_dashboard,
        _iter_python_processes=queries._iter_python_processes,
        _system_stats=queries._system_stats,
        _gpu_stats=queries._gpu_stats,
        _collect_model_scores=queries._collect_model_scores,
        _format_bytes=queries._format_bytes,
        _cmdline_has_module=queries._cmdline_has_module,
        _loose_fields=queries._loose_fields,
        _who_parts=queries._who_parts,
        _fmt_ts=queries._fmt_ts,
        _freshness=queries._freshness,
        _utc_now_iso=queries._utc_now_iso,
        _now_utc=queries._now_utc,
        main=__import__("app.web.__main__", fromlist=["main"]).main,
    )


@pytest.fixture
def isolated_web(monkeypatch):
    monkeypatch.setattr(
        queries, "_iter_python_processes", lambda: iter([(4242, "python -m app.orchestrator --db x", 1024)])
    )
    monkeypatch.setattr(queries, "_system_stats", lambda: (12.5, 1024**3, 4 * 1024**3))
    monkeypatch.setattr(queries, "_gpu_stats", lambda: None)
    monkeypatch.setattr(
        queries,
        "_collect_model_scores",
        lambda: [
            {
                "label": "GPT",
                "path": "checkpoints/gpt.pt",
                "present": False,
                "mean_loss": None,
                "train_axis_acc": None,
                "extra": "",
            }
        ],
    )
    return _web_ns()


@pytest.fixture
def filled_db(job_store, tmp_path):
    done = job_store.enqueue("123456", "text", {"model": "gpt", "start": "Я"})
    job_store.route_one(resolve_worker_kind)
    job_store.claim_assigned("text_gpt")
    job_store.complete(
        done, {"type": "message", "content": "Сгенерированный текст"}, os.getpid()
    )
    failed = job_store.enqueue("cli:1", "mood", {"source": "грустно"})
    job_store.route_one(resolve_worker_kind)
    job_store.claim_assigned("mood")
    job_store.fail(failed, "mood not loaded", os.getpid())
    job_store.enqueue("777", "pic", {"mode_key": "2", "data_b64": "AAAA", "filename": "a.png"})
    job_store.set_meta(
        "orchestrator",
        json.dumps({"role": "router", "pid": os.getpid()}),
    )
    job_store.sync_guilds([(1, "Тестовый сервер", 5)])
    job_store.insert_bot_stats(
        guilds=1, members_total=5, members_online=None, latency_ms=12.0
    )
    return tmp_path / "bot.db"


@pytest.fixture
def admin_client(isolated_web, filled_db):
    app = create_app(filled_db)
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def admin_server(isolated_web, monkeypatch):
    """Compatibility fixture: returns a callable that yields a Flask test client bound to db."""

    def start(db_path):
        app = create_app(db_path)
        app.config["TESTING"] = True
        return app.test_client()

    yield start


def _request(client, method, path):
    resp = client.open(path, method=method)
    return resp.status_code, dict(resp.headers), resp.data


@pytest.mark.integration
def test_dashboard_renders_with_jobs(admin_server, filled_db):
    client = admin_server(filled_db)
    status, headers, body = _request(client, "GET", "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert headers.get("Cache-Control") == "no-store"
    html = body.decode("utf-8")
    assert "<html" in html.lower()
    assert "KizaruAi" in html
    assert "Главная" in html
    assert 'data-page="home"' in html
    jobs = client.get("/api/jobs").get_json()
    assert any("mood not loaded" in str(j.get("error") or "") for j in jobs["jobs"])
    home = client.get("/api/home").get_json()
    assert home["db_ok"] is True
    assert home["kpi"]["guilds"] == 1
    workers = client.get("/api/workers").get_json()
    assert "workers" in workers


@pytest.mark.integration
def test_index_html_alias_and_head(admin_server, filled_db):
    client = admin_server(filled_db)
    assert _request(client, "GET", "/index.html")[0] == 200
    status, _h, body = _request(client, "HEAD", "/")
    assert status == 200
    assert body == b""


@pytest.mark.integration
def test_unknown_path_404(admin_server, filled_db):
    client = admin_server(filled_db)
    assert _request(client, "GET", "/admin/secret")[0] == 404
    assert _request(client, "HEAD", "/x")[0] == 404


@pytest.mark.integration
def test_dashboard_without_db(admin_server, tmp_path):
    client = admin_server(tmp_path / "missing.db")
    status, _h, body = _request(client, "GET", "/")
    assert status == 200 and body
    html = body.decode("utf-8")
    assert "без БД" in html


def test_collect_dashboard_data(isolated_web, filled_db):
    data = queries._collect_dashboard(filled_db)
    assert data["db_ok"] is True
    assert data["by_status"]["done"] == 1 and data["by_status"]["error"] == 1 and data["by_status"]["queued"] == 1
    assert data["history_total"] >= 2
    assert any(s["key"] == "orchestrator" and s["running"] for s in data["services"])
    assert data["running_count"] == 1
    assert any(g.get("name") == "Тестовый сервер" for g in data["guilds"])
    app = create_app(filled_db)
    app.config["TESTING"] = True
    html = app.test_client().get("/").data.decode("utf-8")
    assert "Главная" in html
    assert 'id="cluster"' in html
    home = queries.api_home(filled_db)
    assert home["kpi"]["guilds"] == 1


def test_collect_dashboard_empty_db(isolated_web, job_store, tmp_path):
    data = queries._collect_dashboard(tmp_path / "bot.db")
    assert data["db_ok"] is True and data["active_jobs"] == [] and data["history_jobs"] == []


def test_helpers():
    assert queries._format_bytes(0) == "—"
    assert queries._format_bytes(512) == "512 B"
    assert queries._format_bytes(1536) == "1.5 KiB"
    assert queries._cmdline_has_module("python -m app.web --port 1", "app.web")
    assert not queries._cmdline_has_module("python -m app.webx", "app.web")
    assert queries._loose_fields('{"model": "gpt"}') == {"model": "gpt"}
    assert queries._loose_fields('{"model": "gpt", "data_b64": "AAA') == {"model": "gpt"}
    assert queries._who_parts("cli:12") == ("discord", "cli:12")
    assert queries._who_parts("cli") == ("cli", "cli")
    assert queries._who_parts(None) == ("—", "")
    assert queries._fmt_ts(None) == "—"
    assert queries._freshness(None, None) == "старое"


def test_main_refuses_non_local_bind(monkeypatch):
    import app.web.__main__ as web_main

    monkeypatch.setattr(sys, "argv", ["app.web", "--host", "0.0.0.0", "--port", "0"])
    with pytest.raises(SystemExit) as exc:
        web_main.main()
    assert exc.value.code == 1


def test_waitress_default(monkeypatch, tmp_path):
    import app.web.__main__ as web_main

    called = {}

    def fake_serve(app, host, port, threads=8):
        called["host"] = host
        called["port"] = port
        called["threads"] = threads

    class FakeWaitress:
        serve = staticmethod(fake_serve)

    monkeypatch.setitem(sys.modules, "waitress", FakeWaitress)
    monkeypatch.setattr(web_main, "create_app", lambda db: object())
    db = tmp_path / "bot.db"
    db.write_bytes(b"")
    web_main.main(["--host", "127.0.0.1", "--port", "8765", "--db", str(db)])
    assert called == {"host": "127.0.0.1", "port": 8765, "threads": 8}


def test_default_port_is_8765():
    assert queries.DEFAULT_PORT == 8765 and queries.DEFAULT_HOST == "127.0.0.1"
