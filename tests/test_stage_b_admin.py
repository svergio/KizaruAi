import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

import app.web.queries as queries
from app.web.contracts import STATUS
from app.web import create_app
from app.web.queries import api_user_stats
from core.bus.queue import STATUS_BADGE_CLASSES, TERMINAL_STATUSES, WAIT_STATUSES
from core.bus.routing import resolve_worker_kind

pytest_plugins = ("test_admin_web",)

SNAP = Path(__file__).parent / "snapshots" / "admin_dashboard.html"


def test_api_endpoints_200(admin_server, filled_db):
    client = admin_server(filled_db)

    def get(path):
        resp = client.get(path)
        return resp.status_code, resp.get_json()

    assert get("/api/queue")[0] == 200
    assert get("/api/jobs")[0] == 200
    assert get("/api/stats")[0] == 200
    assert get("/api/users")[0] == 200
    users = get("/api/users")[1]
    assert isinstance(users, dict) and "users" in users
    assert isinstance(users["users"], list)
    code, stats = get("/api/users/123456")
    assert code == 200 and stats["total"] >= 1
    assert isinstance(stats, dict)
    for key in ("id", "total", "by_status"):
        assert key in stats


def test_api_user_unknown_404(admin_server, filled_db):
    client = admin_server(filled_db)
    assert client.get("/api/users/no-such-user-xyz").status_code == 404


def test_web_and_api_share_user_stats(filled_db):
    from core.bus.queue import JobStore

    direct = api_user_stats(filled_db, "123456")
    store_stats = JobStore(filled_db, recover_running=False).user_stats_for_discord("123456")
    assert direct == store_stats


def test_admin_no_external_assets(admin_server, filled_db):
    client = admin_server(filled_db)
    html = client.get("/").data.decode("utf-8")
    css = client.get("/static/admin.css").data.decode("utf-8")
    blob = html + "\n" + css
    assert not re.search(r"""(?:src|href|url\()\s*['"]?https?://""", blob, re.I)


def test_status_badge_palette_single_source():
    css = Path(queries.ROOT / "app" / "web" / "static" / "admin.css").read_text(encoding="utf-8")
    statuses = set(WAIT_STATUSES) | set(TERMINAL_STATUSES)
    assert set(STATUS_BADGE_CLASSES) == statuses
    assert set(STATUS) == statuses
    for status, cls in STATUS_BADGE_CLASSES.items():
        assert f".{cls}" in css
        assert STATUS[status]["css"] == cls
    contracts = Path(queries.ROOT / "app" / "web" / "contracts.py").read_text(encoding="utf-8")
    views = Path(queries.ROOT / "app" / "web" / "views.py").read_text(encoding="utf-8")
    assert "STATUS" in contracts
    assert "#d8f0d8" not in views


def test_static_admin_css_served(admin_server, filled_db):
    client = admin_server(filled_db)
    resp = client.get("/static/admin.css")
    assert resp.status_code == 200
    assert "text/css" in (resp.headers.get("Content-Type") or "")
    assert b".status-queued" in resp.data


def test_db_error_page_no_traceback(admin_server, filled_db, monkeypatch, caplog):
    import logging
    import sqlite3

    def boom(_db):
        raise sqlite3.Error("disk I/O error")

    monkeypatch.setattr(queries, "api_home", boom)
    client = admin_server(filled_db)
    with caplog.at_level(logging.ERROR, logger="kizaru.web"):
        resp = client.get("/api/home")
        body = resp.get_data(as_text=True)
        assert resp.status_code == 500
        assert "Traceback" not in body
        assert "disk I/O" not in body
        assert "error" in body.lower()
        html = client.get("/")
        html_body = html.get_data(as_text=True)
        assert html.status_code in (500, 503)
        assert "Ошибка базы данных" in html_body
        assert "Traceback" not in html_body
    assert any("admin sqlite error" in r.message for r in caplog.records)


def test_bind_localhost_only():
    assert queries.DEFAULT_HOST == "127.0.0.1"


def test_dashboard_snapshot_stable(isolated_web, filled_db, monkeypatch):
    from core.bus.queue import JobStore

    fixed = datetime(2099, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(queries, "_now_utc", lambda: fixed)
    monkeypatch.setattr(os, "getpid", lambda: 4242)
    store = JobStore(filled_db, recover_running=False)
    store.set_meta("orchestrator", json.dumps({"role": "router", "pid": 4242}))
    with store._connect() as conn:
        conn.execute(
            "UPDATE worker_meta SET updated_at = ? WHERE key = 'orchestrator'",
            ("2099-01-01T00:00:00+00:00",),
        )
        conn.execute(
            """
            UPDATE jobs SET
              created_at = '2099-01-01T00:00:00+00:00',
              started_at = CASE WHEN started_at IS NULL THEN NULL ELSE '2099-01-01T00:00:01+00:00' END,
              finished_at = CASE WHEN finished_at IS NULL THEN NULL ELSE '2099-01-01T00:00:02+00:00' END,
              worker_pid = CASE WHEN status = 'running' THEN 4242 ELSE worker_pid END
            """
        )
        conn.execute(
            "UPDATE guilds SET updated_at = '2099-01-01T00:00:00+00:00'"
        )
        conn.execute(
            "UPDATE users SET first_seen = '2099-01-01T00:00:00+00:00', last_seen = '2099-01-01T00:00:00+00:00'"
        )
        conn.commit()
    app = create_app(filled_db)
    app.config["TESTING"] = True
    html = app.test_client().get("/").data.decode("utf-8")
    html = html.replace(str(filled_db), "data/bot.db")
    html = html.replace(str(filled_db).replace("\\", "/"), "data/bot.db")
    html = html.replace(filled_db.as_posix(), "data/bot.db")
    html = re.sub(r"[A-Za-z]:[^\"'<>\s]*bot\.db", "data/bot.db", html)
    # Strip volatile CSRF / script noise not part of the shell contract.
    html = re.sub(r"\s+", " ", html).strip()
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        SNAP.parent.mkdir(parents=True, exist_ok=True)
        SNAP.write_text(html, encoding="utf-8")
    assert SNAP.is_file(), "snapshot missing; set UPDATE_SNAPSHOTS=1 to create"
    expect = re.sub(r"\s+", " ", SNAP.read_text(encoding="utf-8")).strip()
    assert expect == html
