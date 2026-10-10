import asyncio
import json
import multiprocessing
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from core.bus import broker, queue as q
from core.bus.cancel_flag import GenerationCancelled, check_cancelled, set_cancel_event
from core.bus.queue import JobStore
from core.bus.routing import resolve_worker_kind
from core.bus.worker_loop import run_assigned_once
from core.text import sample as sample_mod


def _enqueue_process(db_path, user, output):
    try:
        output.put(("ok", JobStore(db_path).enqueue(user, "mood", {"source": "x"})))
    except Exception as exc:
        output.put((type(exc).__name__, str(exc)))


def _claim_drain_process(db_path, output):
    store = JobStore(db_path)
    got = []
    idle = 0
    while idle < 80:
        row = store.claim_assigned("mood")
        if row is None:
            idle += 1
            time.sleep(0.01)
            continue
        idle = 0
        jid = int(row["id"])
        got.append(jid)
        store.complete(jid, {"type": "message", "content": "x"}, os.getpid())
    output.put(got)


def test_claim_atomic_multiprocess(tmp_path):
    db = tmp_path / "bot.db"
    store = JobStore(db)
    ids = []
    for i in range(40):
        ids.append(store.enqueue(f"u{i}", "mood", {"source": "x"}))
        store.route_one(resolve_worker_kind)
    output = multiprocessing.Queue()
    workers = [
        multiprocessing.Process(target=_claim_drain_process, args=(str(db), output))
        for _ in range(4)
    ]
    for process in workers:
        process.start()
    for process in workers:
        process.join(60)
    claimed = []
    for _ in workers:
        claimed.extend(output.get(timeout=5))
    assert sorted(claimed) == sorted(ids)
    assert len(claimed) == len(set(claimed)) == 40


def test_enqueue_same_user_multiprocess(tmp_path):
    db = tmp_path / "bot.db"
    JobStore(db)
    output = multiprocessing.Queue()
    workers = [
        multiprocessing.Process(target=_enqueue_process, args=(str(db), "u", output))
        for _ in range(2)
    ]
    for process in workers:
        process.start()
    for process in workers:
        process.join(10)
    results = [output.get(timeout=2)[0] for _ in workers]
    assert sorted(results) == ["ActiveJobError", "ok"]


def test_busy_db_no_lock_error(tmp_path):
    db = tmp_path / "bot.db"
    JobStore(db)
    holder_done = threading.Event()

    def hold():
        conn = sqlite3.connect(db, timeout=30)
        conn.execute("BEGIN IMMEDIATE")
        time.sleep(1.5)
        conn.execute("COMMIT")
        conn.close()
        holder_done.set()

    threading.Thread(target=hold, daemon=True).start()
    time.sleep(0.2)
    store = JobStore(db)
    job_id = store.enqueue("u-busy", "mood", {"source": "x"})
    store.route_one(resolve_worker_kind)
    claimed = store.claim_assigned("mood")
    assert claimed is not None and int(claimed["id"]) == job_id
    holder_done.wait(5)


def test_drop_stale_job_checks_migration(tmp_path):
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, discord_id TEXT UNIQUE,
                            first_seen TEXT, last_seen TEXT);
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, kind TEXT, payload TEXT,
            status TEXT CHECK(status IN ('queued','running','done','error')), created_at TEXT,
            started_at TEXT, finished_at TEXT, error TEXT, result TEXT, worker_kind TEXT,
            worker_pid INTEGER, lease_until TEXT, attempts INTEGER DEFAULT 0
        );
        INSERT INTO users VALUES (1, 'u', 't', 't');
        INSERT INTO jobs VALUES (1, 1, 'mood', '{}', 'done', 't', NULL, 't', NULL, '{}',
                                 'mood', 12, 't', 2);
        """
    )
    conn.commit()
    conn.close()
    store = JobStore(db)
    row = store.get_job(1)
    assert row["status"] == "done" and row["attempts"] == 2 and row["worker_pid"] == 12


def test_claim_sets_lease_pid_attempts(job_store, monkeypatch):
    monkeypatch.setattr(q, "lease_seconds", lambda _kind: 180)
    job_id = job_store.enqueue("u", "mood", {"source": "x"})
    job_store.route_one(resolve_worker_kind)
    job_store.claim_assigned("mood")
    row = job_store.get_job(job_id)
    assert row["worker_pid"] == os.getpid()
    assert row["lease_until"] and row["attempts"] == 1


def test_running_lease_expires(tmp_path, monkeypatch):
    db = tmp_path / "bot.db"
    store = JobStore(db)
    job_id = store.enqueue("u", "mood", {"source": "x"})
    store.route_one(resolve_worker_kind)
    store.claim_assigned("mood")
    past = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    with store._connect() as conn:
        conn.execute(
            "UPDATE jobs SET lease_until = ?, worker_pid = 0 WHERE id = ?",
            (past, job_id),
        )
        conn.commit()
    monkeypatch.setattr(q, "_pid_alive", lambda pid: False)
    store.recover_interrupted("mood")
    assert store.get_job(job_id)["status"] == "assigned"


def test_lease_renewed_while_main_blocked(job_store, monkeypatch):
    """Lease daemon inside run_assigned_once renews while processor is blocked."""
    from core.bus import worker_loop as wl

    monkeypatch.setattr(q, "LEASE_RENEW_SEC", 0.05)
    monkeypatch.setattr(q, "CANCEL_POLL_SEC", 0.05)
    monkeypatch.setattr(wl, "LEASE_RENEW_SEC", 0.05)
    monkeypatch.setattr(wl, "CANCEL_POLL_SEC", 0.05)
    monkeypatch.setattr(q, "lease_seconds", lambda _kind: 0.25)
    monkeypatch.setattr(q, "max_runtime_seconds", lambda _kind: 3600)
    job_id = job_store.enqueue("u", "mood", {"source": "x"})
    job_store.route_one(resolve_worker_kind)
    stop = threading.Event()
    threading.Thread(
        target=wl._lease_daemon, args=(job_store, "mood", stop), daemon=True
    ).start()
    seen = {}

    def processor(payload):
        before = job_store.get_job(job_id)["lease_until"]
        seen["before"] = before
        time.sleep(0.7)
        after = job_store.get_job(job_id)["lease_until"]
        seen["after"] = after
        assert after != before
        assert job_store.get_job(job_id)["status"] == "running"
        assert job_store.claim_assigned("mood") is None
        return {"type": "message", "content": "ok"}

    assert run_assigned_once(job_store, "mood", processor)
    stop.set()
    assert seen["after"] != seen["before"]
    assert job_store.get_job(job_id)["status"] == "done"


def test_recover_only_dead_or_expired_and_scoped(tmp_path, monkeypatch):
    db = tmp_path / "bot.db"
    store = JobStore(db)
    mood_id = store.enqueue("u1", "mood", {"source": "x"})
    store.route_one(resolve_worker_kind)
    gpt_id = store.enqueue("u2", "text", {"model": "gpt", "start": "x"})
    store.route_one(resolve_worker_kind)
    store.claim_assigned("mood")
    store.claim_assigned("text_gpt")
    live_pid = os.getpid()
    with store._connect() as conn:
        conn.execute("UPDATE jobs SET worker_pid = ? WHERE id = ?", (live_pid, mood_id))
        conn.execute("UPDATE jobs SET worker_pid = 0 WHERE id = ?", (gpt_id,))
        conn.commit()

    def alive(pid):
        return int(pid) == live_pid

    monkeypatch.setattr(q, "_pid_alive", alive)
    JobStore(db, recover_running=True, worker_kind="mood")
    assert store.get_job(mood_id)["status"] == "running"
    JobStore(db, recover_running=True, worker_kind="text_gpt")
    assert store.get_job(gpt_id)["status"] == "assigned"


def test_second_worker_start_does_not_steal_running(tmp_path, monkeypatch):
    db = tmp_path / "bot.db"
    store = JobStore(db)
    job_id = store.enqueue("u", "mood", {"source": "x"})
    store.route_one(resolve_worker_kind)
    store.claim_assigned("mood")
    owner = os.getpid()
    monkeypatch.setattr(q, "_pid_alive", lambda pid: int(pid) == owner)
    JobStore(db, recover_running=True, worker_kind="mood")
    assert store.get_job(job_id)["status"] == "running"
    assert store.get_job(job_id)["worker_pid"] == owner


def test_poison_job_max_attempts(job_store, monkeypatch):
    monkeypatch.setattr(q, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(q, "_pid_alive", lambda pid: False)
    job_id = job_store.enqueue("u", "mood", {"source": "x"})
    for _ in range(3):
        job_store.route_one(resolve_worker_kind)
        assert job_store.claim_assigned("mood") is not None
        job_store.recover_interrupted("mood")
    row = job_store.get_job(job_id)
    assert row["status"] == "error"
    assert "max attempts" in (row["error"] or "")


def test_complete_requires_running_owner(job_store):
    job_id = job_store.enqueue("u", "mood", {"source": "x"})
    job_store.route_one(resolve_worker_kind)
    job_store.claim_assigned("mood")
    assert job_store.complete(job_id, {"type": "message", "content": "x"}, 999999) == 0
    assert job_store.get_job(job_id)["status"] == "running"
    assert job_store.fail(job_id, "nope", 999999) == 0
    assert job_store.get_job(job_id)["status"] == "running"
    with pytest.raises(TypeError):
        job_store.complete(job_id, {"type": "message", "content": "x"})  # type: ignore[misc]
    assert job_store.complete(job_id, {"type": "message", "content": "ok"}, os.getpid()) == 1
    assert job_store.get_job(job_id)["status"] == "done"


def test_late_fail_after_reclaim_does_not_change_done(tmp_path, monkeypatch):
    """A leased, lease expired, B completed done; late fail from A is ignored."""
    db = tmp_path / "bot.db"
    store = JobStore(db)
    job_id = store.enqueue("u", "mood", {"source": "x"})
    store.route_one(resolve_worker_kind)
    store.claim_assigned("mood")
    pid_a = 111111
    past = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    with store._connect() as conn:
        conn.execute(
            "UPDATE jobs SET worker_pid = ?, lease_until = ? WHERE id = ?",
            (pid_a, past, job_id),
        )
        conn.commit()
    monkeypatch.setattr(q, "_pid_alive", lambda pid: False)
    store.recover_interrupted("mood")
    assert store.get_job(job_id)["status"] == "assigned"
    store.claim_assigned("mood")
    assert store.complete(job_id, {"type": "message", "content": "b"}, os.getpid()) == 1
    assert store.get_job(job_id)["status"] == "done"
    assert store.fail(job_id, "late from A", pid_a) == 0
    assert store.get_job(job_id)["status"] == "done"
    assert store.complete(job_id, {"type": "message", "content": "late"}, pid_a) == 0
    assert store.get_job(job_id)["status"] == "done"


def test_queue_limit_counts_assigned(job_store, monkeypatch):
    monkeypatch.setattr(q, "MAX_QUEUED_JOBS", 2)
    a = job_store.enqueue("u1", "mood", {"source": "a"})
    b = job_store.enqueue("u2", "mood", {"source": "b"})
    assert job_store.route_one(resolve_worker_kind)
    assert job_store.route_one(resolve_worker_kind)
    assert job_store.get_job(a)["status"] == "assigned"
    assert job_store.get_job(b)["status"] == "assigned"
    with pytest.raises(q.QueueFullError):
        job_store.enqueue("u3", "mood", {"source": "c"})


def test_orchestrator_online_requires_live_pid(job_store, monkeypatch):
    assert broker.orchestrator_online(job_store) is False
    job_store.set_meta("orchestrator", json.dumps({"role": "router", "pid": os.getpid()}))
    monkeypatch.setattr(q, "_pid_alive", lambda pid: int(pid) == os.getpid())
    assert broker.orchestrator_online(job_store) is True
    past = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    with job_store._connect() as conn:
        conn.execute(
            "UPDATE worker_meta SET updated_at = ? WHERE key = 'orchestrator'",
            (past,),
        )
        conn.commit()
    assert broker.orchestrator_online(job_store) is False


def test_assigned_for_offline_worker_expires_or_rejected(job_store, monkeypatch):
    job_id = job_store.enqueue("u", "mood", {"source": "x"})
    job_store.route_one(resolve_worker_kind)
    monkeypatch.setattr(JobStore, "worker_online", lambda self, kind: False)
    assert job_store.reap_offline_assigned() >= 1
    assert job_store.get_job(job_id)["status"] == "error"
    assert "offline" in (job_store.get_job(job_id)["error"] or "")


def test_worker_oom_marks_error_and_survives(job_store, monkeypatch):
    import torch

    first = job_store.enqueue("u1", "mood", {"source": "a"})
    second = job_store.enqueue("u2", "mood", {"source": "b"})
    job_store.route_one(resolve_worker_kind)
    job_store.route_one(resolve_worker_kind)
    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    state = {"n": 0}

    def processor(payload):
        state["n"] += 1
        if state["n"] == 1:
            raise torch.cuda.OutOfMemoryError("no memory")
        return {"type": "message", "content": payload["source"]}

    assert run_assigned_once(job_store, "mood", processor)
    assert job_store.get_job(first)["error"] == "OOM"
    assert emptied
    assert run_assigned_once(job_store, "mood", processor)
    assert job_store.get_job(second)["status"] == "done"


def test_poll_until_done_sync_timeout(job_store):
    job_id = job_store.enqueue("u", "mood", {"source": "x"})
    with pytest.raises((TimeoutError, SystemExit)):
        broker.poll_until_done_sync(job_store, job_id, interval=0.01, timeout=0.05)


def test_cancel_rules_and_flag_stops_sampling(job_store, monkeypatch):
    job_id = job_store.enqueue("owner", "mood", {"source": "x"})
    assert job_store.cancel(job_id, "other") is False
    assert job_store.get_job(job_id)["status"] == "queued"
    assert job_store.cancel(job_id, "owner") is True
    assert job_store.get_job(job_id)["status"] == "cancelled"
    assert job_store.cancel(job_id, "owner") is False

    job2 = job_store.enqueue("owner2", "mood", {"source": "y"})
    job_store.route_one(resolve_worker_kind)
    job_store.claim_assigned("mood")
    event = threading.Event()
    set_cancel_event(event)
    event.set()
    with pytest.raises(GenerationCancelled):
        check_cancelled()
    set_cancel_event(None)

    calls = {"db": 0}
    real_get = job_store.get_job

    def tracked(jid):
        calls["db"] += 1
        return real_get(jid)

    monkeypatch.setattr(job_store, "get_job", tracked)
    from core.text.rnn import TextRNN

    alphabet = "abc "
    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    model = TextRNN(len(alphabet), 8, 8)
    event = threading.Event()
    set_cancel_event(event)
    steps = {"n": 0}
    real_check = check_cancelled

    def counting_check():
        steps["n"] += 1
        if steps["n"] >= 3:
            event.set()
        real_check()

    monkeypatch.setattr("core.bus.cancel_flag.check_cancelled", counting_check)
    with pytest.raises(GenerationCancelled):
        sample_mod.evaluate(
            model,
            char_to_idx,
            idx_to_char,
            start_text="a",
            prediction_len=20,
            temp=1.0,
        )
    assert steps["n"] <= 4
    assert calls["db"] == 0
    set_cancel_event(None)

    job_store.cancel(job2, "owner2")
    assert job_store.get_job(job2)["status"] == "cancelled"


def test_max_runtime_timeout(job_store, monkeypatch):
    monkeypatch.setattr(q, "max_runtime_seconds", lambda _kind: 0)
    job_id = job_store.enqueue("u", "mood", {"source": "x"})
    job_store.route_one(resolve_worker_kind)
    job_store.claim_assigned("mood")
    past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    with job_store._connect() as conn:
        conn.execute("UPDATE jobs SET started_at = ? WHERE id = ?", (past, job_id))
        conn.commit()
    assert job_store.renew_running_lease("mood", os.getpid()) == "timeout"
    assert job_store.get_job(job_id)["status"] == "error"
    assert job_store.get_job(job_id)["error"] == "timeout"


def test_late_complete_fail_do_not_overwrite_cancel(job_store):
    job_id = job_store.enqueue("u", "mood", {"source": "x"})
    job_store.route_one(resolve_worker_kind)
    job_store.claim_assigned("mood")
    assert job_store.cancel(job_id, "u")
    assert job_store.complete(job_id, {"type": "message"}, os.getpid()) == 0
    assert job_store.fail(job_id, "late", os.getpid()) == 0
    assert job_store.get_job(job_id)["status"] == "cancelled"


def test_poll_timeout_and_cancelled(job_store):
    job_id = job_store.enqueue("u", "mood", {"source": "x"})
    with pytest.raises(TimeoutError):
        asyncio.run(q.poll_until_done(job_store, job_id, interval=0.01, timeout=0.02))
    job_store.cancel(job_id, "u")
    with pytest.raises(RuntimeError, match="cancel"):
        asyncio.run(q.poll_until_done(job_store, job_id, interval=0.01, timeout=1))
