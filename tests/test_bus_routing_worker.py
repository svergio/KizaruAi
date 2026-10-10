"""core/bus/routing.py, worker_loop.py, broker.py."""
import base64
import json
import os
import sqlite3

import pytest

from core.bus import broker, routing
from core.bus.routing import (
    build_pic_payload,
    build_text_payload,
    pic_mode_name_to_key,
    resolve_worker_kind,
)
from core.bus.worker_loop import register_worker, run_assigned_once


def test_build_text_payload():
    assert build_text_payload("gpt", "Я", temp=0.5) == {"model": "gpt", "start": "Я", "temp": 0.5}
    assert build_text_payload("lstm") == {"model": "lstm", "start": "Я"}
    with pytest.raises(ValueError):
        build_text_payload("gpt-5")


@pytest.mark.parametrize(
    "model, worker",
    [
        ("lstm", "text_lstm"),
        ("lstm-word", "text_lstm_word"),
        ("transformer", "text_transformer"),
        ("gpt", "text_gpt"),
        ("deepseek", "text_deepseek"),
        ("bert-gen", "text_bert_gen"),
    ],
)
def test_resolve_text_workers(model, worker):
    assert resolve_worker_kind("text", {"model": model}) == worker
    assert worker in routing.WORKER_KINDS


def test_resolve_misc_kinds():
    assert resolve_worker_kind("mood", {}) == "mood"
    with pytest.raises(ValueError):
        resolve_worker_kind("text", {"model": "nope"})
    with pytest.raises(ValueError):
        resolve_worker_kind("video", {})
    with pytest.raises(ValueError):
        resolve_worker_kind("pic", {"mode_key": "9"})


def test_resolve_pic_still_and_video(small_png_bytes, animated_gif_bytes):
    png = base64.b64encode(small_png_bytes).decode()
    gif = base64.b64encode(animated_gif_bytes).decode()
    assert resolve_worker_kind("pic", {"mode_key": "2", "data_b64": png}) == "pic_still"
    assert resolve_worker_kind("pic", {"mode_key": "3", "data_b64": png, "filename": "a.png"}) == "pic_still"
    assert resolve_worker_kind("pic", {"mode_key": "3", "data_b64": gif, "filename": "a.gif"}) == "pic_video"
    assert resolve_worker_kind("pic", {"mode_key": "1", "animated": True, "as_file": True}) == "pic_video"
    assert resolve_worker_kind("pic", {"mode_key": "2", "animated": True, "as_file": True}) == "pic_still"


def test_build_pic_payload(small_png_bytes):
    simple = build_pic_payload("2", small_png_bytes, "a.png")
    assert simple["mode_key"] == "2" and "data_b64" in simple
    assert simple["dark"] and simple["light"]
    render = build_pic_payload("3", small_png_bytes, "a.png", scale=0, cells=40)
    assert render["as_file"] is True and render["scale"] == 1 and render["cells"] == 40
    assert {"tone", "structure", "details"} <= set(render["line"])
    details = build_pic_payload("4", None, "a.png", source_path="x.png")
    assert details["source_path"] == "x.png" and details["line"]["details"] is True
    with pytest.raises(ValueError):
        build_pic_payload("9", small_png_bytes, "a.png")
    with pytest.raises(ValueError):
        build_pic_payload("2", None, "a.png")


def test_pic_mode_name_to_key():
    assert pic_mode_name_to_key("line") == "1"
    assert pic_mode_name_to_key("simple") == "2"
    assert pic_mode_name_to_key("render") == "3"
    assert pic_mode_name_to_key("details") == "4"


# --------------------------------------------------------------- worker_loop


def _assigned_job(store, user="u1"):
    job_id = store.enqueue(user, "mood", {"source": "кайф"})
    store.route_one(resolve_worker_kind)
    return job_id


def test_run_assigned_once_success(job_store):
    assert run_assigned_once(job_store, "mood", lambda p: {}) is False
    job_id = _assigned_job(job_store)
    assert run_assigned_once(job_store, "mood", lambda p: {"type": "message", "content": p["source"]})
    row = job_store.get_job(job_id)
    assert row["status"] == "done" and json.loads(row["result"])["content"] == "кайф"


def test_run_assigned_once_processor_error(job_store):
    job_id = _assigned_job(job_store)

    def boom(payload):
        raise RuntimeError("модель упала")

    assert run_assigned_once(job_store, "mood", boom)
    row = job_store.get_job(job_id)
    assert row["status"] == "error" and row["error"] == "модель упала"


def test_run_assigned_once_bad_payload(job_store, tmp_path):
    job_id = _assigned_job(job_store)
    conn = sqlite3.connect(tmp_path / "bot.db")
    conn.execute("UPDATE jobs SET payload = 'not json' WHERE id = ?", (job_id,))
    conn.commit()
    conn.close()
    assert run_assigned_once(job_store, "mood", lambda p: {})
    assert job_store.get_job(job_id)["error"] == "bad payload json"


def test_register_worker_writes_pid(job_store):
    register_worker(job_store, "mood")
    meta = json.loads(job_store.get_meta("worker:mood"))
    assert meta == {"pid": os.getpid(), "kind": "mood"}
    assert job_store.worker_online("mood")


# -------------------------------------------------------------------- broker


def test_require_broker(job_store):
    with pytest.raises(SystemExit, match="orchestrator"):
        broker.require_broker(job_store, "mood")
    job_store.set_meta(
        "orchestrator",
        json.dumps({"role": "router", "pid": os.getpid()}),
    )
    with pytest.raises(SystemExit, match="mood"):
        broker.require_broker(job_store, "mood")
    register_worker(job_store, "mood")
    broker.require_broker(job_store, "mood")


def test_poll_until_done_sync(job_store):
    ok = _assigned_job(job_store, "u1")
    job_store.claim_assigned("mood")
    job_store.complete(ok, {"type": "message", "content": "x"}, os.getpid())
    assert broker.poll_until_done_sync(job_store, ok, interval=0.01)["content"] == "x"
    bad = _assigned_job(job_store, "u2")
    job_store.claim_assigned("mood")
    job_store.fail(bad, "ошибка воркера", os.getpid())
    with pytest.raises(SystemExit, match="ошибка воркера"):
        broker.poll_until_done_sync(job_store, bad, interval=0.01)
    with pytest.raises(SystemExit):
        broker.poll_until_done_sync(job_store, 999, interval=0.01)


def test_cli_user_id():
    assert broker.cli_user_id() == "cli"
