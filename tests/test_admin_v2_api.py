"""Contract tests for Admin v2 JSON API and UI assets."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import app.web.queries as queries
from app.web import create_app
from app.web.contracts import (
    ENDPOINTS,
    EXAMPLE_HOME,
    EXAMPLE_LOGS,
    EXAMPLE_MODELS,
    EXAMPLE_SERVICES,
    EXAMPLE_TIMESERIES,
    EXAMPLE_WORKERS,
    KIND,
    PAGE_SIZE,
    STATUS,
    TIMESERIES_METRICS,
    TIMESERIES_RANGES,
    WORKER_KIND,
)
from app.web.formatters import fmt_dur, fmt_msk, mask_discord_id, trunc_ellipsis
from core.bus.queue import ActiveJobError, JobStore

pytest_plugins = ("test_admin_web",)

VENDOR_SHA256 = "f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539"

STATUS_SPEC = {
    "queued": ("#e5e7eb", "#374151"),
    "assigned": ("#dbeafe", "#1e40af"),
    "running": ("#cffafe", "#155e75"),
    "done": ("#dcfce7", "#166534"),
    "error": ("#fee2e2", "#991b1b"),
    "cancelled": ("#e5e7eb", "#6b7280"),
}

ROOT_TOKENS = {
    "--bg": "#f5f6f8",
    "--panel": "#ffffff",
    "--border": "#d9dde3",
    "--row-alt": "#fafbfc",
    "--row-hover": "#eef4fb",
    "--text": "#1f2328",
    "--muted": "#6b7280",
    "--link": "#1a6fb5",
    "--header": "#2b3138",
    "--header-text": "#e6e8eb",
    "--tab-active": "#ffffff",
}

ALLOWED_UI_SYMBOLS = set("●▲▼▸›‹")


@pytest.fixture
def client(filled_db, isolated_web):
    app = create_app(filled_db)
    app.config["TESTING"] = True
    return app.test_client()


def queries_root_vendor():
    return Path(queries.ROOT) / "app" / "web" / "static" / "vendor" / "d3.v7.min.js"


def _web_root() -> Path:
    return Path(queries.ROOT) / "app" / "web"


def test_contracts_dicts_complete():
    assert set(STATUS) == {"queued", "assigned", "running", "done", "error", "cancelled"}
    assert set(KIND) == {"text", "mood", "pic"}
    assert "text_gpt" in WORKER_KIND and "pic_video" in WORKER_KIND
    assert PAGE_SIZE == 50
    assert "GET /api/home" in ENDPOINTS
    assert "GET /api/overview" not in ENDPOINTS


def test_status_tokens_match_spec():
    for key, (bg, fg) in STATUS_SPEC.items():
        assert STATUS[key]["bg"] == bg
        assert STATUS[key]["fg"] == fg
    css = (_web_root() / "static" / "admin.css").read_text(encoding="utf-8")
    root = re.search(r":root\s*\{([^}]+)\}", css)
    assert root
    block = root.group(1)
    for key, (bg, fg) in STATUS_SPEC.items():
        assert f"--status-{key}-bg: {bg}" in block
        assert f"--status-{key}-fg: {fg}" in block
    for name, val in ROOT_TOKENS.items():
        assert f"{name}: {val}" in block
    assert "--warn:" in block
    assert "--warn-bg:" in block
    assert "--chart-series" not in css


def test_admin_css_tokens():
    root = _web_root()
    css = (root / "static" / "admin.css").read_text(encoding="utf-8")
    root_m = re.search(r":root\s*\{([^}]+)\}", css)
    assert root_m
    root_block = root_m.group(1)
    outside = css[root_m.end() :]
    hex_re = re.compile(r"#[0-9a-fA-F]{3,8}\b")
    rgb_re = re.compile(r"\brgba?\s*\(", re.I)
    assert not hex_re.search(outside), "hex outside :root forbidden"
    assert not rgb_re.search(outside), "rgb outside :root forbidden"
    for name, val in ROOT_TOKENS.items():
        assert f"{name}: {val}" in root_block
    # templates + app.js: no raw hex/rgb (vendor skipped)
    js = (root / "static" / "js" / "app.js").read_text(encoding="utf-8")
    charts = "\n".join(p.read_text(encoding="utf-8") for p in (root / "static" / "js" / "charts").glob("*.js"))
    templates = "\n".join(p.read_text(encoding="utf-8") for p in (root / "templates").glob("*.html"))
    for blob, label in ((js, "app.js"), (charts, "charts"), (templates, "templates")):
        assert not hex_re.search(blob), f"hex in {label}"
        assert not rgb_re.search(blob), f"rgb in {label}"


def test_admin_css_forbidden():
    root = _web_root()
    css = (root / "static" / "admin.css").read_text(encoding="utf-8")
    for needle in ("box-shadow", "linear-gradient", "radial-gradient", "backdrop-filter", "@import", "@font-face"):
        assert needle not in css
    assert "url(http" not in css.lower()
    for m in re.finditer(r"border-radius\s*:\s*([^;]+)", css):
        val = m.group(1).strip()
        if val in ("0", "0px", "none"):
            continue
        nums = re.findall(r"(\d+(?:\.\d+)?)\s*px", val)
        for n in nums:
            assert float(n) <= 3.0, val
    blob = css + "\n" + "\n".join(
        p.read_text(encoding="utf-8") for p in (root / "templates").glob("*.html")
    )
    for cls in (
        "container-fluid",
        "btn-primary",
        "grid-cols-",
        "bg-gray-",
        "text-sm",
        "rounded-lg",
        "shadow-md",
    ):
        assert cls not in blob
    assert not re.search(r'\bclass="[^"]*\bflex\b', blob)
    assert not re.search(r"\bclass=['\"]flex['\"]", blob)


def test_formatters():
    assert fmt_dur(0.054) == "54 мс"
    assert fmt_dur(2.1) == "2,1 с"
    assert fmt_dur(192) == "3 мин 12 с"
    assert fmt_dur(3900) == "1 ч 05 мин"
    from datetime import datetime, timezone

    # 2026-10-09 21:08:53 UTC → 10 окт, 00:08:53 MSK
    assert (
        fmt_msk(
            "2026-10-09T21:08:53+00:00",
            now=datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc),
        )
        == "10 окт, 00:08:53"
    )
    # same calendar day in MSK → time only
    assert (
        fmt_msk(
            "2026-10-10T10:15:30+00:00",
            now=datetime(2026, 10, 10, 12, 0, 0, tzinfo=timezone.utc),
        )
        == "13:15:30"
    )
    assert mask_discord_id("123456789") == "1234…89"
    assert mask_discord_id("12345678") == "12345678"
    assert mask_discord_id("cli") == "cli"
    assert trunc_ellipsis("abcdef", 4) == "abc…"
    assert trunc_ellipsis("ab", 10) == "ab"


def test_filters_in_url(client):
    r = client.get("/jobs?status=error&page=2")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert re.search(
        r'class="chip active"[^>]*data-status="error"|data-status="error"[^>]*class="[^"]*\bactive\b',
        html,
    )
    active_chips = re.findall(r'<a class="chip active"[^>]*data-status="([^"]*)"', html)
    if not active_chips:
        active_chips = re.findall(
            r'<a class="chip[^"]*\bactive\b"[^>]*data-status="([^"]*)"', html
        )
    assert active_chips == ["error"]
    r_all = client.get("/jobs")
    html_all = r_all.get_data(as_text=True)
    assert re.search(r'class="chip active"[^>]*data-status=""', html_all) or re.search(
        r'data-status=""[^>]*class="[^"]*\bactive\b', html_all
    )
    assert 'id="jobs-page-num">2<' in html or ">2</span>" in html
    assert "filter_page" not in html
    assert 'value="error"' in html


def test_pages_no_emoji():
    root = _web_root()
    blobs = []
    for p in list((root / "templates").glob("*.html")) + [
        root / "static" / "js" / "app.js",
        root / "static" / "admin.css",
    ]:
        blobs.append(p.read_text(encoding="utf-8"))
    for p in (root / "static" / "js" / "charts").glob("*.js"):
        blobs.append(p.read_text(encoding="utf-8"))
    text = "\n".join(blobs)
    emoji = re.findall(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]", text)
    assert emoji == []
    # Any non-ASCII symbol-like chars outside allowed set → fail if in Private Use / misc symbols broadly
    for ch in set(re.findall(r"[^\x00-\x7F]", text)):
        if ch.isalpha() or ch.isdigit() or ch in "«»—–…„“”‚‘’°×÷±":
            continue
        if ch in ALLOWED_UI_SYMBOLS:
            continue
        # Cyrillic and punctuation OK
        if "\u0400" <= ch <= "\u04FF" or ch in "ёЁ№":
            continue
        if ch in "\u00a0\u202f\u2009":
            continue
        # allow CSS content quotes etc.
        if ch in "\"'":
            continue
        pytest.fail(f"disallowed symbol {ch!r} U+{ord(ch):04X}")


def test_vendor_pinned():
    vendor_dir = _web_root() / "static" / "vendor"
    files = [p for p in vendor_dir.iterdir() if p.is_file()]
    assert len(files) == 1
    assert files[0].name == "d3.v7.min.js"
    digest = hashlib.sha256(files[0].read_bytes()).hexdigest()
    assert digest == VENDOR_SHA256
    lic = (Path(queries.ROOT) / "docs" / "licensing.md").read_text(encoding="utf-8")
    assert VENDOR_SHA256 in lic


def _shape_match(example, actual, path=""):
    if isinstance(example, dict):
        assert isinstance(actual, dict), path
        for k, v in example.items():
            assert k in actual, f"missing {path}.{k}"
            _shape_match(v, actual[k], f"{path}.{k}" if path else k)
    elif isinstance(example, list):
        assert isinstance(actual, list), path
        if example and actual:
            _shape_match(example[0], actual[0], f"{path}[]")
    else:
        assert type(example) is type(actual) or (
            example is None or actual is None
        ) or isinstance(actual, type(example)), path


def test_api_shapes_match_contracts(client):
    h = client.get("/api/home").get_json()
    _shape_match(EXAMPLE_HOME, h)
    svc = client.get("/api/services").get_json()
    _shape_match(EXAMPLE_SERVICES, svc)
    w = client.get("/api/workers").get_json()
    _shape_match(EXAMPLE_WORKERS, w)
    m = client.get("/api/models").get_json()
    assert "models" in m
    if m["models"]:
        ex = EXAMPLE_MODELS["models"][0]
        for k in ex:
            assert k in m["models"][0]
    ts = client.get("/api/stats/timeseries?metric=jobs_created&range=24h").get_json()
    _shape_match(EXAMPLE_TIMESERIES, ts)
    logs = client.get("/api/logs").get_json()
    _shape_match(EXAMPLE_LOGS, logs)


def test_timeseries_per_point_and_user_stats_once(filled_db, monkeypatch):
    from datetime import datetime, timezone

    store = JobStore(filled_db, recover_running=False)
    with store._connect() as conn:
        conn.execute("DELETE FROM jobs")
        uid = conn.execute("SELECT id FROM users LIMIT 1").fetchone()[0]
        # two jobs in hour 10, one in hour 11 (UTC), window ends 2099-01-01T12:00
        for ts in (
            "2099-01-01T10:10:00+00:00",
            "2099-01-01T10:50:00+00:00",
            "2099-01-01T11:15:00+00:00",
        ):
            conn.execute(
                "INSERT INTO jobs (user_id, kind, payload, status, created_at) VALUES (?,?,?,?,?)",
                (uid, "text", '{"model":"lstm","start":"x"}', "done", ts),
            )
        conn.commit()
    fixed = datetime(2099, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(queries, "_now_utc", lambda: fixed)
    from app.web.queries import stats as stats_mod

    monkeypatch.setattr(stats_mod, "_now_utc", lambda: fixed)
    body = queries.api_timeseries(filled_db, metric="jobs_created", range_key="24h")
    assert body["metric"] == "jobs_created"
    assert body["bucket_sec"] == 3600
    assert len(body["points"]) == 24
    by_t = {p["t"]: p["v"] for p in body["points"]}
    # start = 2099-01-01T12:00 - 24h = 2099-01-00? → 2098-12-31T12:00
    assert sum(by_t.values()) == 3
    hour10 = [p for p in body["points"] if p["t"].startswith("2099-01-01T10:")]
    hour11 = [p for p in body["points"] if p["t"].startswith("2099-01-01T11:")]
    assert hour10 and hour10[0]["v"] == 2
    assert hour11 and hour11[0]["v"] == 1
    for p in body["points"]:
        assert set(p.keys()) == {"t", "v"}
        assert isinstance(p["v"], int)

    mock = MagicMock(wraps=store.user_stats_for_discord)
    monkeypatch.setattr(JobStore, "user_stats_for_discord", mock)
    queries.api_user_stats(filled_db, "123456")
    mock.assert_called_once()


def test_api_models_no_torch_load(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    bare = ckpt_dir / "orphan.pt"
    bare.write_bytes(b"not-a-real-checkpoint")
    meta_pt = ckpt_dir / "with_meta.pt"
    meta_pt.write_bytes(b"x")
    (ckpt_dir / "with_meta.pt.meta.json").write_text(
        json.dumps(
            {
                "name": "with_meta.pt",
                "tokenizer": "char",
                "vocab_size": 10,
                "step": 3,
                "val_ru_rap": 1.5,
                "bpc": 2.0,
                "date": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    from app.web.queries import models as models_mod

    monkeypatch.setattr(models_mod, "ROOT", tmp_path)
    torch_load = MagicMock(side_effect=AssertionError("torch.load must not be called"))
    monkeypatch.setitem(__import__("sys").modules, "torch", MagicMock(load=torch_load))
    body = models_mod.api_models()
    names = {m["name"]: m for m in body["models"]}
    assert names["orphan.pt"]["missing_metadata"] is True
    assert names["orphan.pt"]["note"] == "нет метаданных"
    assert names["with_meta.pt"]["step"] == 3
    assert names["with_meta.pt"]["val_ru_rap"] == 1.5
    torch_load.assert_not_called()


def test_app_web_no_torch_weights_only_false():
    root = _web_root()
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        assert "weights_only=False" not in text
        assert "import torch" not in text
        assert "from torch" not in text


def test_api_home_shape(client):
    r = client.get("/api/home")
    assert r.status_code == 200
    data = r.get_json()
    for key in ("generated_at", "db_ok", "cluster", "kpi", "bars", "audience", "storage", "recent_jobs"):
        assert key in data
    assert data["thresholds"]["warn"] == 70
    assert data["thresholds"]["error"] == 90
    ids = [n["id"] for n in data["cluster"]["nodes"]]
    assert ids == ["discord", "orchestrator", "queue", "workers", "models"]


def test_api_queue_and_jobs_filters(client):
    assert client.get("/api/queue").status_code == 200
    r = client.get("/api/jobs?status=done&page=1")
    assert r.status_code == 200
    data = r.get_json()
    assert data["page"] == 1 and data["page_size"] == PAGE_SIZE
    assert all(j["status"] == "done" for j in data["jobs"])


def test_api_job_detail_404(client):
    assert client.get("/api/jobs/999999").status_code == 404


def test_api_cancel_csrf_required(client, filled_db, job_store):
    jid = job_store.enqueue("999", "text", {"model": "lstm", "start": "Я"})
    r = client.post(f"/api/jobs/{jid}/cancel")
    assert r.status_code == 400
    assert r.get_json()["error"] == "csrf"


def test_api_cancel_ok(client, job_store):
    jid = job_store.enqueue("888", "text", {"model": "lstm", "start": "Я"})
    csrf = client.get("/api/csrf").get_json()["csrf_token"]
    r = client.post(
        f"/api/jobs/{jid}/cancel",
        headers={"X-CSRF-Token": csrf},
    )
    assert r.status_code == 200
    assert r.get_json()["status"] == "cancelled"


def test_api_users_hide_cli(client):
    r = client.get("/api/users?hide_cli=1")
    assert r.status_code == 200
    body = r.get_json()
    users = body["users"]
    assert all("display_name" not in u and "avatar_url" not in u for u in users)
    assert all("id_mask" in u and "key" in u for u in users)
    assert all(u["id"] == u["id_mask"] for u in users)
    assert all("…" in u["id"] or len(u["id"]) <= 8 or u["id"] == "cli" for u in users)
    assert "summary" in body and "users_total" in body["summary"]
    assert "audience" in body and "strip" in body["audience"]
    assert "activity" in body and "by_recency" in body["activity"]
    assert "jobs_sparkline_7d" in body


def test_api_workers_models_logs_timeseries(client):
    assert client.get("/api/services").status_code == 200
    assert client.get("/api/workers").status_code == 200
    assert client.get("/api/models").status_code == 200
    assert client.get("/api/logs").status_code == 200
    for metric in TIMESERIES_METRICS:
        r = client.get(f"/api/stats/timeseries?metric={metric}&range=24h")
        assert r.status_code == 200
        body = r.get_json()
        assert body["metric"] == metric
        assert "points" in body
    assert set(TIMESERIES_RANGES) >= {"1h", "24h", "7d", "30d"}


def test_pages_200(client):
    for path in ("/", "/jobs", "/services", "/system", "/users", "/workers", "/logs"):
        assert client.get(path).status_code == 200
    assert client.get("/queue", follow_redirects=False).status_code == 302
    assert client.get("/queue").headers.get("Location", "").endswith("/jobs")
    assert client.get("/models", follow_redirects=False).status_code == 302
    assert client.get("/models").headers.get("Location", "").endswith("/workers")
    assert client.get("/jobs/1").status_code == 200
    assert client.get("/users/123456").status_code == 200


def test_jobs_page_queue_strip(client):
    r = client.get("/jobs")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'id="queue-pipeline"' in html
    assert 'id="pipe-clients"' in html
    assert 'data-page="jobs"' in html
    assert "Активная очередь" in html
    assert 'href="/queue"' not in html
    assert 'data-page="overview"' not in html
    assert 'data-page="home"' not in html


def test_services_page_containers(client):
    r = client.get("/services")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'data-page="services"' in html
    assert 'id="tbl-services"' in html
    assert "Сервисы" in html
    assert 'id="services-deploy-cmd"' in html
    assert "runbook.md" in html


def test_system_page_containers(client):
    r = client.get("/system")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'data-page="system"' in html
    assert 'id="system-bars"' in html
    assert 'id="system-storage"' in html
    assert "Система" in html


def test_html_db_error_no_traceback(client, monkeypatch, caplog):
    import logging
    import sqlite3

    def boom(_db):
        raise sqlite3.Error("disk I/O error")

    monkeypatch.setattr(queries, "api_home", boom)
    with caplog.at_level(logging.ERROR, logger="kizaru.web"):
        resp = client.get("/")
        body = resp.get_data(as_text=True)
        assert resp.status_code in (500, 503)
        assert "Ошибка базы данных" in body
        assert "Traceback" not in body
        assert "disk I/O" not in body
        assert "test_admin" not in body
        assert ".py" not in body or "Ошибка" in body
    assert any("admin sqlite error" in r.message for r in caplog.records)


def test_debug_false_and_vendor_present():
    app = create_app()
    assert app.config.get("DEBUG") is False
    vendor = queries_root_vendor()
    assert vendor.is_file() and vendor.stat().st_size > 1000


def test_cli_user_merged(job_store):
    a = job_store.enqueue("cli:11", "text", {"model": "lstm", "start": "а"})
    b = job_store.enqueue("cli:22", "text", {"model": "lstm", "start": "б"})
    assert a != b
    assert job_store.user_id_for_discord("cli") is not None
    assert job_store.user_id_for_discord("cli:11") == job_store.user_id_for_discord("cli")


def test_cli_migration_idempotent(tmp_path):
    db = tmp_path / "mig.db"
    store = JobStore(db, recover_running=False)
    with store._connect() as conn:
        now = "2026-01-01T00:00:00+00:00"
        conn.execute(
            "INSERT INTO users (discord_id, first_seen, last_seen) VALUES ('cli:old', ?, ?)",
            (now, now),
        )
        old_id = int(conn.execute("SELECT id FROM users WHERE discord_id='cli:old'").fetchone()[0])
        conn.execute(
            "INSERT INTO jobs (user_id, kind, payload, status, created_at) VALUES (?,?,?,?,?)",
            (old_id, "text", '{"model":"lstm","start":"x"}', "done", now),
        )
        conn.commit()
        before_jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        before_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    store2 = JobStore(db, recover_running=False)
    with store2._connect() as conn:
        after_jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        after_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        cli_jobs = conn.execute(
            "SELECT COUNT(*) FROM jobs j JOIN users u ON u.id=j.user_id WHERE u.discord_id='cli'"
        ).fetchone()[0]
        leftover = conn.execute(
            "SELECT COUNT(*) FROM users WHERE discord_id LIKE 'cli:%'"
        ).fetchone()[0]
    assert before_jobs == after_jobs
    assert cli_jobs >= 1
    assert leftover == 0
    JobStore(db, recover_running=False)  # second run
    with JobStore(db, recover_running=False)._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == after_jobs
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == after_users
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM users WHERE discord_id LIKE 'cli:%'"
            ).fetchone()[0]
            == 0
        )


def test_display_name_avatar_migration_idempotent(tmp_path):
    """Old users schema without profile columns → migrate twice; profile via /ai me."""
    import asyncio
    import sqlite3
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    import app.discord.__main__ as discord_main

    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            discord_id TEXT NOT NULL UNIQUE,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL
        );
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            error TEXT,
            result TEXT,
            worker_kind TEXT,
            worker_pid INTEGER,
            lease_until TEXT,
            attempts INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE worker_meta (
            key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE guilds (
            discord_id TEXT PRIMARY KEY, name TEXT NOT NULL,
            member_count INTEGER, updated_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()
    JobStore(db, recover_running=False)
    JobStore(db, recover_running=False)
    with sqlite3.connect(str(db)) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(users)")}
    assert "display_name" in cols and "avatar_url" in cols

    bot = discord_main.build_bot(db_path=db)
    interaction = MagicMock()
    interaction.user = SimpleNamespace(
        id=42,
        name="raw",
        display_name="Bob",
        display_avatar=SimpleNamespace(url="https://cdn.discordapp.com/avatars/42/a.png"),
    )
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    for cmd in bot.tree.get_commands():
        if cmd.name != "ai":
            continue
        for sub in cmd.commands:
            if sub.name == "me":
                asyncio.run(sub.callback(interaction))
                break
    profile = queries.api_user_profile(db, "42")
    assert profile["display_name"] == "Bob"
    assert profile["avatar_url"].startswith("https://cdn.discordapp.com/")
    detail = queries.api_user_detail(db, "42")
    assert detail["display_name"] is None
    assert detail["avatar_url"] is None
    assert "…" in detail["id"] or detail["id"] == "42"


def test_bot_updates_name_and_avatar(job_store):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    import app.discord.__main__ as discord_main

    bot = discord_main.build_bot(db_path=job_store.db_path)
    interaction = MagicMock()
    interaction.user = SimpleNamespace(
        id=4242,
        name="raw",
        display_name="Nice Name",
        display_avatar=SimpleNamespace(url="https://cdn.discordapp.com/avatars/4242/a.png"),
    )
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    for cmd in bot.tree.get_commands():
        if cmd.name != "ai":
            continue
        for sub in cmd.commands:
            if sub.name == "me":
                asyncio.run(sub.callback(interaction))
                break
    profile = queries.api_user_profile(job_store.db_path, "4242")
    assert profile["display_name"] == "Nice Name"
    assert profile["id"] == "4242"
    interaction.response.send_message.assert_awaited()


def test_cli_unlimited_active_discord_limited(job_store):
    job_store.enqueue("cli", "text", {"model": "lstm", "start": "а"})
    job_store.enqueue("cli", "text", {"model": "lstm", "start": "б"})
    job_store.enqueue("cli", "mood", {"source": "x"})
    job_store.enqueue("555", "text", {"model": "lstm", "start": "я"})
    with pytest.raises(ActiveJobError):
        job_store.enqueue("555", "text", {"model": "lstm", "start": "ты"})


def test_jobs_page_sql_budget(filled_db, monkeypatch):
    store_path = filled_db
    conn = sqlite3.connect(str(store_path))
    try:
        uid = conn.execute("SELECT id FROM users WHERE discord_id='cli'").fetchone()
        if uid is None:
            conn.execute(
                "INSERT INTO users (discord_id, first_seen, last_seen) VALUES ('cli','t','t')"
            )
            conn.commit()
            uid = conn.execute("SELECT id FROM users WHERE discord_id='cli'").fetchone()
        cli_id = int(uid[0])
        conn.executemany(
            "INSERT INTO jobs (user_id, kind, payload, status, created_at) VALUES (?,?,?,?,?)",
            [
                (cli_id, "text", '{"model":"lstm","start":"x"}', "done", "2026-01-01T00:00:00+00:00")
                for _ in range(200)
            ],
        )
        conn.commit()
    finally:
        conn.close()

    counts = {"n": 0}
    real_ro = queries._connect_ro

    class _CountingConn:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *a, **k):
            counts["n"] += 1
            return self._inner.execute(sql, *a, **k)

        def close(self):
            return self._inner.close()

    def counting_ro(db_path):
        return _CountingConn(real_ro(db_path))

    monkeypatch.setattr(queries, "_connect_ro", counting_ro)
    body = queries.api_jobs_page(store_path, page=1)
    assert body["total"] >= 200
    assert len(body["jobs"]) == PAGE_SIZE
    assert counts["n"] <= 3


def test_pages_have_chart_containers(client):
    home = client.get("/").get_data(as_text=True)
    assert 'data-page="home"' in home
    assert 'id="cluster"' in home
    assert 'id="chart-home-spark"' in home
    assert 'id="chart-home-heatmap"' in home
    assert "Аудитория бота" in home
    assert 'id="home-bars"' in home
    assert "donut" not in home
    assert "Главная" in home
    jobs = client.get("/jobs").get_data(as_text=True)
    assert 'id="chart-jobs-gantt"' in jobs
    assert 'id="queue-pipeline"' in jobs
    system = client.get("/system").get_data(as_text=True)
    assert 'id="system-bars"' in system
    users_page = client.get("/users").get_data(as_text=True)
    assert 'data-page="users"' in users_page
    assert 'id="users-kpi"' in users_page
    assert 'id="chart-users-audience"' in users_page
    assert 'id="chart-users-jobs-spark"' in users_page
    assert 'id="users-activity-recency"' in users_page
    assert "Аудитория" in users_page
    assert 'id="tbl-users"' in users_page
    assert "<th>id</th>" not in users_page
    assert "<th>Имя</th>" not in users_page
    user = client.get("/users/123456").get_data(as_text=True)
    assert 'id="chart-user-spark"' in user
    workers = client.get("/workers").get_data(as_text=True)
    assert 'id="chart-models-loss"' in workers
    assert 'id="tbl-runtime"' in workers
    assert 'data-page="workers"' in workers
    assert 'href="/models"' not in workers


def test_ckpt_meta_null_val_keys(tmp_path):
    import torch
    from training import ckpt_meta

    path = tmp_path / "bare.pt"
    torch.save({"char_to_idx": {"a": 0}, "step": 1}, path)
    meta = ckpt_meta.extract_and_write(path)
    assert "val_ru_rap" in meta and meta["val_ru_rap"] is None
    assert "bpc" in meta and meta["bpc"] is None
    side = json.loads(ckpt_meta.sidecar_path(path).read_text(encoding="utf-8"))
    assert "val_ru_rap" in side and side["val_ru_rap"] is None
    assert "bpc" in side and side["bpc"] is None


def test_user_profile_csrf_and_delete(client, job_store):
    job_store.update_user_profile(
        "999888777666",
        display_name="Zed",
        avatar_url="https://cdn.discordapp.com/avatars/1/z.png",
    )
    pub = client.get("/api/users/999888777666").get_json()
    assert pub["display_name"] is None
    assert "…" in pub["id"]
    r = client.post("/api/users/999888777666/profile")
    assert r.status_code == 400
    csrf = client.get("/api/csrf").get_json()["csrf_token"]
    r = client.post(
        "/api/users/999888777666/profile",
        headers={"X-CSRF-Token": csrf},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["id"] == "999888777666"
    assert body["display_name"] == "Zed"
    r = client.post(
        "/api/users/999888777666/delete",
        headers={"X-CSRF-Token": csrf},
    )
    assert r.status_code == 200
    assert r.get_json()["deleted"] is True
    r2 = client.post(
        "/api/users/999888777666/delete",
        headers={"X-CSRF-Token": csrf},
    )
    assert r2.get_json()["deleted"] is False


def test_bot_stats_and_retention(tmp_path):
    from datetime import datetime, timedelta, timezone

    db = tmp_path / "ret.db"
    store = JobStore(db, recover_running=False)
    store.insert_bot_stats(guilds=2, members_total=10, members_online=None, latency_ms=12.5)
    old = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO bot_stats (ts, guilds, members_total, members_online, latency_ms) "
            "VALUES (?, 1, 1, NULL, 1.0)",
            (old,),
        )
        uid = conn.execute("SELECT id FROM users WHERE discord_id='cli'").fetchone()[0]
        ancient = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        conn.execute(
            "INSERT INTO jobs (user_id, kind, payload, status, created_at, finished_at) "
            "VALUES (?,?,?,?,?,?)",
            (uid, "text", "{}", "done", ancient, ancient),
        )
        conn.commit()
    out = store.retention_cleanup()
    assert out["bot_stats_deleted"] >= 1
    assert out["jobs_deleted"] >= 1
    with store._connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM bot_stats").fetchone()[0]
        assert n >= 1
        meta = conn.execute(
            "SELECT value FROM worker_meta WHERE key='retention_cleanup'"
        ).fetchone()
        assert meta is not None


def test_bind_and_debug_defaults():
    assert queries.DEFAULT_HOST == "127.0.0.1"
    assert queries.DEFAULT_PORT == 8765
    app = create_app()
    assert app.debug is False
    assert app.config["DEBUG"] is False


def test_dot_is_bullet_symbol():
    css = (_web_root() / "static" / "admin.css").read_text(encoding="utf-8")
    assert 'content: "●"' in css or "content: '●'" in css
    assert not re.search(r"\.dot\s*\{[^}]*border-radius", css, re.S)
