"""core/bus/queue.py: очередь заданий на SQLite (временная база)."""
import asyncio
import json
import os
import sqlite3

import pytest

from core.bus import queue as q
from core.bus.queue import (
    ActiveJobError,
    JobStore,
    PayloadTooLargeError,
    QueueFullError,
    estimate_wait_seconds,
    format_eta_ru,
    poll_until_done,
)
from core.bus.routing import resolve_worker_kind

TEXT_GPT = {"model": "gpt", "start": "Я"}


def test_estimate_wait_seconds():
    assert estimate_wait_seconds(0, 10) == 10
    assert estimate_wait_seconds(2, 10) == 30
    assert estimate_wait_seconds(-5, 0) == 1


@pytest.mark.parametrize(
    "sec, expected",
    [(30, "~30 сек"), (0, "~1 сек"), (120, "~2 мин"), (3600, "~1 ч"), (3900, "~1 ч 5 мин")],
)
def test_format_eta_ru(sec, expected):
    assert format_eta_ru(sec) == expected


def test_full_job_lifecycle(job_store):
    job_id = job_store.enqueue("u1", "text", TEXT_GPT)
    assert job_store.get_job(job_id)["status"] == "queued"
    assert job_store.queue_position(job_id) == 1
    assert job_store.route_one(resolve_worker_kind) is True
    row = job_store.get_job(job_id)
    assert row["status"] == "assigned" and row["worker_kind"] == "text_gpt"
    claimed = job_store.claim_assigned("text_gpt")
    assert claimed["id"] == job_id and json.loads(claimed["payload"]) == TEXT_GPT
    assert job_store.get_job(job_id)["status"] == "running"
    job_store.complete(job_id, {"type": "message", "content": "привет"}, os.getpid())
    row = job_store.get_job(job_id)
    assert row["status"] == "done"
    assert json.loads(row["result"]) == {"type": "message", "content": "привет"}
    assert asyncio.run(poll_until_done(job_store, job_id, interval=0.01))["content"] == "привет"


def test_fail_job(job_store):
    job_id = job_store.enqueue("u1", "mood", {"source": "x"})
    job_store.route_one(resolve_worker_kind)
    job_store.claim_assigned("mood")
    job_store.fail(job_id, "сломалось", os.getpid())
    row = job_store.get_job(job_id)
    assert row["status"] == "error" and row["error"] == "сломалось"
    with pytest.raises(RuntimeError, match="сломалось"):
        asyncio.run(poll_until_done(job_store, job_id, interval=0.01))


def test_enqueue_unknown_kind(job_store):
    with pytest.raises(ValueError):
        job_store.enqueue("u1", "video", {})


def test_one_active_job_per_user(job_store):
    job_store.enqueue("u1", "text", TEXT_GPT)
    with pytest.raises(ActiveJobError):
        job_store.enqueue("u1", "mood", {"source": "x"})
    job_store.enqueue("u2", "mood", {"source": "x"})


def test_payload_too_large(job_store, monkeypatch):
    monkeypatch.setattr(q, "MAX_PAYLOAD_BYTES", 10)
    with pytest.raises(PayloadTooLargeError):
        job_store.enqueue("u1", "text", {"model": "gpt", "start": "очень длинный старт"})


def test_queue_full(job_store, monkeypatch):
    monkeypatch.setattr(q, "MAX_QUEUED_JOBS", 2)
    job_store.enqueue("u1", "mood", {"source": "x"})
    job_store.enqueue("u2", "mood", {"source": "x"})
    with pytest.raises(QueueFullError):
        job_store.enqueue("u3", "mood", {"source": "x"})


def test_route_one_empty_and_errors(job_store):
    assert job_store.route_one(resolve_worker_kind) is False
    bad = job_store.enqueue("u1", "text", {"model": "gpt"})

    def boom(kind, payload):
        raise RuntimeError("routing failed")

    assert job_store.route_one(boom) is True
    assert job_store.get_job(bad)["status"] == "error"
    other = job_store.enqueue("u2", "mood", {"source": "x"})
    assert job_store.route_one(lambda k, p: "nonexistent") is True
    assert "bad worker_kind" in job_store.get_job(other)["error"]


def test_claim_assigned_one_running_per_worker(job_store):
    with pytest.raises(ValueError):
        job_store.claim_assigned("nope")
    a = job_store.enqueue("u1", "text", TEXT_GPT)
    b = job_store.enqueue("u2", "text", TEXT_GPT)
    job_store.route_one(resolve_worker_kind)
    job_store.route_one(resolve_worker_kind)
    assert job_store.claim_assigned("text_gpt")["id"] == a
    assert job_store.claim_assigned("text_gpt") is None  # воркер занят
    job_store.complete(a, {"type": "message", "content": "ok"}, os.getpid())
    assert job_store.claim_assigned("text_gpt")["id"] == b


def test_recover_interrupted_reassigns_running(tmp_path, monkeypatch):
    db = tmp_path / "bot.db"
    store = JobStore(db)
    job_id = store.enqueue("u1", "text", TEXT_GPT)
    store.route_one(resolve_worker_kind)
    store.claim_assigned("text_gpt")
    with store._connect() as conn:
        conn.execute(
            "UPDATE jobs SET worker_pid = 0, lease_until = '2000-01-01T00:00:00+00:00' WHERE id = ?",
            (job_id,),
        )
        conn.commit()
    monkeypatch.setattr(q, "_pid_alive", lambda pid: False)
    JobStore(db, recover_running=True, worker_kind="text_gpt")
    assert store.get_job(job_id)["status"] == "assigned"


def test_estimate_queue_wait_counts_same_worker(job_store):
    a = job_store.enqueue("u1", "text", TEXT_GPT)
    b = job_store.enqueue("u2", "text", TEXT_GPT)
    job_store.enqueue("u3", "mood", {"source": "x"})
    info = job_store.estimate_queue_wait(b, "text_gpt")
    assert info["ahead"] == 1 and info["position"] == 2
    assert info["avg_sec"] == q.DEFAULT_DURATION_SEC["text_gpt"] and info["avg_from_history"] is False
    assert info["eta_sec"] == 2 * q.DEFAULT_DURATION_SEC["text_gpt"]
    assert job_store.jobs_ahead_for_worker(a, "text_gpt") == 0


def test_meta_and_worker_online(job_store):
    assert job_store.get_meta("x") is None
    job_store.set_meta("x", "1")
    job_store.set_meta("x", "2")
    assert job_store.get_meta("x") == "2"
    job_store.clear_meta("x")
    assert job_store.get_meta("x") is None

    assert job_store.get_worker_models() is None
    job_store.set_meta("worker:text_gpt", json.dumps({"pid": os.getpid()}))
    assert job_store.worker_online("text_gpt") is True
    models = job_store.get_worker_models()
    assert models["gpt"] is True and models["lstm"] is False

    job_store.set_meta("worker:mood", json.dumps({"pid": 0}))
    assert job_store.worker_online("mood") is False
    assert job_store.get_meta("worker:mood") is None  # мёртвая запись чистится

    job_store.set_meta("worker:pic_still", "not json")
    assert job_store.worker_online("pic_still") is False


def test_sync_guilds(job_store, tmp_path):
    job_store.sync_guilds([(1, "Сервер", 10), ("2", "x" * 600, None)])
    job_store.sync_guilds([(1, "Переименован", 11)])
    conn = sqlite3.connect(tmp_path / "bot.db")
    rows = dict(conn.execute("SELECT discord_id, name FROM guilds").fetchall())
    conn.close()
    assert rows["1"] == "Переименован" and len(rows["2"]) == 500


def test_legacy_db_with_check_constraint_is_migrated(tmp_path):
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, discord_id TEXT NOT NULL UNIQUE,
                            first_seen TEXT NOT NULL, last_seen TEXT NOT NULL);
        CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id),
                           kind TEXT NOT NULL, payload TEXT NOT NULL,
                           status TEXT NOT NULL CHECK (status IN ('queued','running','done','error')),
                           created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, error TEXT);
        INSERT INTO users (discord_id, first_seen, last_seen) VALUES ('old', 't', 't');
        INSERT INTO jobs (user_id, kind, payload, status, created_at) VALUES (1, 'mood', '{"source":"x"}', 'done', 't');
        """
    )
    conn.commit()
    conn.close()
    store = JobStore(db)
    assert store.get_job(1)["status"] == "done"
    new_id = store.enqueue("u1", "text", TEXT_GPT)
    assert new_id == 2
    store.route_one(resolve_worker_kind)
    assert store.get_job(new_id)["status"] == "assigned"
