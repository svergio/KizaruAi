"""One CUDA job across all GPU_KINDS (no real GPU)."""
import multiprocessing
import os
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
import torch

from core.bus import queue as q
from core.bus.queue import GPU_KINDS, JobStore
from core.bus.routing import WORKER_KINDS, resolve_worker_kind
from core.bus.worker_loop import register_worker, run_assigned_once


def _registered_worker_kinds() -> set[str]:
    from app.workers import bert_gen, deepseek, gpt, lstm, lstm_word, mood, still, transformer, video

    return {
        lstm.KIND,
        lstm_word.KIND,
        transformer.KIND,
        gpt.KIND,
        deepseek.KIND,
        bert_gen.KIND,
        mood.KIND,
        still.KIND,
        video.KIND,
    }


def _gpu_holder_process(db_path, hold_sec, output):
    store = JobStore(db_path)
    job = store.claim_assigned("mood")
    output.put(("mood", None if job is None else int(job["id"])))
    if job is None:
        return
    time.sleep(hold_sec)
    store.complete(int(job["id"]), {"type": "message", "content": "ok"}, os.getpid())
    output.put(("mood_done", int(job["id"])))


def _gpu_waiter_process(db_path, output):
    store = JobStore(db_path)
    first = store.claim_assigned("text_gpt")
    output.put(("gpt_while_busy", None if first is None else int(first["id"])))
    deadline = time.time() + 8.0
    claimed = None
    while time.time() < deadline:
        claimed = store.claim_assigned("text_gpt")
        if claimed is not None:
            break
        time.sleep(0.05)
    output.put(("gpt_after", None if claimed is None else int(claimed["id"])))
    if claimed is not None:
        store.complete(int(claimed["id"]), {"type": "message", "content": "g"}, os.getpid())


def test_gpu_kinds_match_registered_cuda_workers():
    registered = _registered_worker_kinds()
    assert set(WORKER_KINDS) == registered
    assert set(GPU_KINDS) == registered
    assert "pic_still" in GPU_KINDS and "pic_video" in GPU_KINDS


def test_gpu_exclusive_claim_multiprocess(tmp_path):
    db = tmp_path / "bot.db"
    store = JobStore(db)
    mood_id = store.enqueue("u-mood", "mood", {"source": "x"})
    store.route_one(resolve_worker_kind)
    gpt_id = store.enqueue("u-gpt", "text", {"model": "gpt", "start": "a"})
    store.route_one(resolve_worker_kind)
    assert store.get_job(mood_id)["worker_kind"] == "mood"
    assert store.get_job(gpt_id)["worker_kind"] == "text_gpt"

    output = multiprocessing.Queue()
    holder = multiprocessing.Process(
        target=_gpu_holder_process, args=(str(db), 0.6, output)
    )
    waiter = multiprocessing.Process(target=_gpu_waiter_process, args=(str(db), output))
    holder.start()
    time.sleep(0.15)
    waiter.start()
    holder.join(15)
    waiter.join(15)
    assert holder.exitcode == 0 and waiter.exitcode == 0
    events = {}
    for _ in range(4):
        key, val = output.get(timeout=5)
        events[key] = val
    assert events["mood"] == mood_id
    assert events["gpt_while_busy"] is None
    assert events["gpt_after"] == gpt_id
    assert events["mood_done"] == mood_id
    assert store.get_job(mood_id)["status"] == "done"
    assert store.get_job(gpt_id)["status"] == "done"


def test_unload_before_complete_and_fail(job_store, monkeypatch):
    order = []
    model = MagicMock()

    def to_cpu(device):
        order.append(("to", str(device)))
        return model

    model.to.side_effect = to_cpu
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: order.append("empty_cache")
    )

    def tracking_complete(job_id, result, worker_pid):
        order.append("complete")
        return JobStore.complete(job_store, job_id, result, worker_pid)

    def tracking_fail(job_id, message, worker_pid):
        order.append("fail")
        return JobStore.fail(job_store, job_id, message, worker_pid)

    monkeypatch.setattr(job_store, "complete", tracking_complete)
    monkeypatch.setattr(job_store, "fail", tracking_fail)

    job_ok = job_store.enqueue("u1", "mood", {"source": "ok"})
    job_store.route_one(resolve_worker_kind)
    runtime = {"lstm_model": model, "checkpoint": "stub"}
    assert run_assigned_once(
        job_store,
        "mood",
        lambda p: {"type": "message", "content": p["source"]},
        runtime=runtime,
    )
    assert order[:3] == [("to", "cpu"), "empty_cache", "complete"]
    assert job_store.get_job(job_ok)["status"] == "done"

    order.clear()
    job_err = job_store.enqueue("u2", "mood", {"source": "boom"})
    job_store.route_one(resolve_worker_kind)

    def boom(_payload):
        raise RuntimeError("nope")

    assert run_assigned_once(job_store, "mood", boom, runtime=runtime)
    assert order[:3] == [("to", "cpu"), "empty_cache", "fail"]
    assert job_store.get_job(job_err)["status"] == "error"


def test_gpu_wait_assigned_not_reaped_while_video_runs(job_store, monkeypatch):
    monkeypatch.setattr(q, "_pid_alive", lambda pid: int(pid) == os.getpid())
    register_worker(job_store, "pic_video")
    register_worker(job_store, "text_gpt")

    video_id = job_store.enqueue(
        "u-vid",
        "pic",
        {
            "mode_key": "1",
            "animated": True,
            "as_file": True,
            "data_b64": "AA==",
            "filename": "a.gif",
        },
    )
    job_store.route_one(resolve_worker_kind)
    assert job_store.get_job(video_id)["worker_kind"] == "pic_video"
    assert job_store.claim_assigned("pic_video") is not None

    started = (datetime.now(timezone.utc) - timedelta(minutes=25)).isoformat()
    lease = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    with job_store._connect() as conn:
        conn.execute(
            "UPDATE jobs SET started_at = ?, lease_until = ? WHERE id = ?",
            (started, lease, video_id),
        )
        conn.commit()

    gpt_id = job_store.enqueue("u-gpt2", "text", {"model": "gpt", "start": "a"})
    job_store.route_one(resolve_worker_kind)
    assert job_store.get_job(gpt_id)["status"] == "assigned"
    assert job_store.claim_assigned("text_gpt") is None

    assert job_store.reap_offline_assigned() == 0
    assert job_store.get_job(gpt_id)["status"] == "assigned"

    assert job_store.complete(
        video_id, {"type": "message", "content": "v"}, os.getpid()
    ) == 1
    claimed = job_store.claim_assigned("text_gpt")
    assert claimed is not None and int(claimed["id"]) == gpt_id
