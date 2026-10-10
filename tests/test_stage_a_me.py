import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.discord import __main__ as discord_main
from core.bus.queue import JobStore
from core.bus.routing import resolve_worker_kind
from core.bus.user_stats import aggregate_user_job_rows


def _seed_done(job_store, user: str, kind: str, payload: dict, started: str, finished: str):
    job_id = job_store.enqueue(user, kind, payload)
    job_store.route_one(resolve_worker_kind)
    job_store.claim_assigned(resolve_worker_kind(kind, payload))
    with job_store._connect() as conn:
        conn.execute(
            """
            UPDATE jobs SET started_at = ?, finished_at = ?, status = 'done',
                   result = '{"type":"message","content":"x"}'
            WHERE id = ?
            """,
            (started, finished, job_id),
        )
        conn.commit()
    return job_id


def test_user_stats_scoped_to_user(job_store):
    _seed_done(
        job_store,
        "u1",
        "mood",
        {"source": "a"},
        "2020-01-01T00:00:00+00:00",
        "2020-01-01T00:00:10+00:00",
    )
    job_store.enqueue("u2", "mood", {"source": "b"})
    uid1 = job_store.user_id_for_discord("u1")
    uid2 = job_store.user_id_for_discord("u2")
    s1 = job_store.user_stats(uid1)
    s2 = job_store.user_stats(uid2)
    assert s1["total"] == 1 and s2["total"] == 1
    assert s1["done"] == 1 and s2["done"] == 0


def test_p95_nearest_rank():
    rows = []
    for i in range(20):
        rows.append(
            {
                "kind": "mood",
                "status": "done",
                "created_at": "t",
                "started_at": "2020-01-01T00:00:00+00:00",
                "finished_at": f"2020-01-01T00:00:{i + 1:02d}+00:00",
            }
        )
    stats = aggregate_user_job_rows(rows)
    assert stats["p95_duration_sec"] == 19.0
    assert stats["avg_duration_sec"] == sum(range(1, 21)) / 20


def test_empty_user_stub(job_store):
    stats = job_store.user_stats_for_discord("never-seen")
    assert stats["total"] == 0
    assert stats["done"] == 0
    from core.bus.user_stats import format_me_embed_lines

    lines = format_me_embed_lines(stats, "Name")
    assert "заявок пока нет" in lines[0][1]


def test_jobs_user_kind_index_idempotent(tmp_path):
    db = tmp_path / "bot.db"
    JobStore(db)
    JobStore(db)
    conn = __import__("sqlite3").connect(db)
    names = {r[1] for r in conn.execute("PRAGMA index_list(jobs)").fetchall()}
    conn.close()
    assert "idx_jobs_user_kind" in names


def test_ai_me_uses_user_stats(job_store, monkeypatch):
    _seed_done(
        job_store,
        "99",
        "mood",
        {"source": "a"},
        "2020-01-01T00:00:00+00:00",
        "2020-01-01T00:00:05+00:00",
    )
    bot = discord_main.build_bot(db_path=job_store.db_path)
    store = bot.job_store
    captured = []
    orig = store.user_stats_for_discord

    def _track(discord_id):
        captured.append(discord_id)
        return orig(discord_id)

    monkeypatch.setattr(store, "user_stats_for_discord", _track)

    interaction = MagicMock()
    interaction.user = SimpleNamespace(
        id=99,
        name="tester",
        display_name="Tester",
        display_avatar=SimpleNamespace(url="http://127.0.0.1/avatar.png"),
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

    assert captured == ["99"]
    stats = store.user_stats_for_discord("99")
    assert stats["total"] == 1
    interaction.response.send_message.assert_awaited()
