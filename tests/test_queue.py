import asyncio
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from core.bus.queue import (
    MAX_PAYLOAD_BYTES,
    MAX_QUEUED_JOBS,
    ActiveJobError,
    JobStore,
    PayloadTooLargeError,
    QueueFullError,
    poll_until_done,
)
from core.bus.routing import resolve_worker_kind
def _orchestrator_loop(store: JobStore, stop: threading.Event):
    while not stop.is_set():
        if store.route_one(resolve_worker_kind):
            continue
        time.sleep(0.02)


def _gen_worker_loop(store: JobStore, worker_kind: str, processor, stop: threading.Event, order: list, peak: list):
    running = 0
    while not stop.is_set():
        job = store.claim_assigned(worker_kind)
        if job is None:
            time.sleep(0.02)
            continue
        job_id = int(job["id"])
        running += 1
        peak[0] = max(peak[0], running)
        try:
            payload = json.loads(job["payload"])
            order.append((worker_kind, payload.get("n")))
            result = processor(payload)
            store.complete(job_id, result)
        except Exception as exc:
            store.fail(job_id, str(exc))
        finally:
            running -= 1


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "bot.db"
        self.bot_store = JobStore(self.db)
        self.worker_store = JobStore(self.db)
        self.process_order = []
        self.peak = [0]
        self._stop = threading.Event()

        def processor(payload):
            return {"type": "message", "content": str(payload.get("n"))}

        self._orch = threading.Thread(
            target=_orchestrator_loop,
            args=(self.worker_store, self._stop),
            daemon=True,
        )
        self._worker = threading.Thread(
            target=_gen_worker_loop,
            args=(
                self.worker_store,
                "text_lstm",
                processor,
                self._stop,
                self.process_order,
                self.peak,
            ),
            daemon=True,
        )
        self._orch.start()
        self._worker.start()

    async def asyncTearDown(self):
        self._stop.set()
        self._orch.join(timeout=3)
        self._worker.join(timeout=3)
        self.tmp.cleanup()

    async def test_bot_worker_process_isolation(self):
        job_id = await asyncio.to_thread(
            self.bot_store.enqueue, "u1", "text", {"model": "lstm", "n": 7}
        )
        row = await poll_until_done(self.bot_store, job_id, interval=0.05)
        self.assertEqual(row["content"], "7")
        finished = self.worker_store.get_job(job_id)
        self.assertEqual(finished["status"], "done")
        self.assertEqual(finished["worker_kind"], "text_lstm")

    async def test_fifo_enqueue_order(self):
        ids = []
        for i in range(3):
            jid = await asyncio.to_thread(
                self.bot_store.enqueue,
                f"user-{i}",
                "text",
                {"model": "lstm", "n": i},
            )
            ids.append(jid)
        for jid in ids:
            await poll_until_done(self.bot_store, jid, interval=0.05)
        self.assertEqual([x[1] for x in self.process_order], [0, 1, 2])

    async def test_one_job_per_worker_kind(self):
        self._stop.set()
        self._orch.join(timeout=3)
        self._worker.join(timeout=3)
        store = JobStore(self.db)
        stop = threading.Event()
        order = []
        peak = [0]

        def proc(payload):
            time.sleep(0.1)
            return {"type": "message", "content": "ok"}

        orch = threading.Thread(target=_orchestrator_loop, args=(store, stop), daemon=True)
        worker = threading.Thread(
            target=_gen_worker_loop,
            args=(store, "text_lstm", proc, stop, order, peak),
            daemon=True,
        )
        orch.start()
        worker.start()
        j1 = store.enqueue("a", "text", {"model": "lstm", "n": 1})
        j2 = store.enqueue("b", "text", {"model": "lstm", "n": 2})
        await poll_until_done(store, j1, interval=0.05)
        await poll_until_done(store, j2, interval=0.05)
        stop.set()
        orch.join(timeout=3)
        worker.join(timeout=3)
        self.assertEqual(peak[0], 1)

    async def test_active_job_per_user(self):
        j1 = await asyncio.to_thread(
            self.bot_store.enqueue, "same", "text", {"model": "lstm", "n": 1}
        )
        with self.assertRaises(ActiveJobError):
            self.bot_store.enqueue("same", "mood", {"n": 2})
        result = await poll_until_done(self.bot_store, j1, interval=0.05)
        self.assertEqual(result["content"], "1")

    async def test_pic_routes_to_pic_still_worker(self):
        self._stop.set()
        self._orch.join(timeout=3)
        self._worker.join(timeout=3)
        store = JobStore(Path(self.tmp.name) / "pic.db")
        stop = threading.Event()
        seen = []

        def pic_proc(payload):
            seen.append(payload.get("mode_key"))
            return {"type": "message", "content": "pic"}

        orch = threading.Thread(target=_orchestrator_loop, args=(store, stop), daemon=True)
        worker = threading.Thread(
            target=_gen_worker_loop,
            args=(store, "pic_still", pic_proc, stop, [], [0]),
            daemon=True,
        )
        orch.start()
        worker.start()
        jid = store.enqueue("p", "pic", {"mode_key": "1", "data_b64": "eA==", "filename": "x.png"})
        await poll_until_done(store, jid, interval=0.05)
        row = store.get_job(jid)
        self.assertEqual(row["worker_kind"], "pic_still")
        stop.set()
        orch.join(timeout=3)
        worker.join(timeout=3)

    def test_enqueue_rejects_oversized_payload(self):
        store = JobStore(Path(self.tmp.name) / "big.db")
        blob = "x" * (MAX_PAYLOAD_BYTES + 1)
        with self.assertRaises(PayloadTooLargeError):
            store.enqueue("u-big", "text", {"data": blob})

    def test_enqueue_rejects_when_queue_full(self):
        store = JobStore(Path(self.tmp.name) / "full.db")
        for i in range(MAX_QUEUED_JOBS):
            store.enqueue(f"user-{i}", "text", {"model": "lstm", "n": i})
        with self.assertRaises(QueueFullError):
            store.enqueue("user-overflow", "text", {"model": "lstm", "n": 99})

    def test_worker_claims_only_its_kind(self):
        store = JobStore(Path(self.tmp.name) / "kind.db")
        j_text = store.enqueue("a", "text", {"model": "lstm"})
        j_mood = store.enqueue("b", "mood", {"source": "x"})
        self.assertTrue(store.route_one(resolve_worker_kind))
        self.assertTrue(store.route_one(resolve_worker_kind))
        self.assertEqual(store.get_job(j_mood)["worker_kind"], "mood")
        claimed = store.claim_assigned("text_lstm")
        self.assertEqual(int(claimed["id"]), j_text)
        mood_claim = store.claim_assigned("mood")
        self.assertEqual(int(mood_claim["id"]), j_mood)
        self.assertIsNone(store.claim_assigned("text_lstm"))

    def test_legacy_status_check_accepts_assigned(self):
        path = Path(self.tmp.name) / "legacy.db"
        conn = sqlite3.connect(path)
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
                user_id INTEGER NOT NULL REFERENCES users(id),
                kind TEXT NOT NULL CHECK(kind IN ('text', 'mood', 'pic')),
                payload TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued', 'running', 'done', 'error')),
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                error TEXT
            );
            CREATE TABLE worker_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE guilds (
                discord_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                member_count INTEGER,
                updated_at TEXT NOT NULL
            );
            INSERT INTO users (discord_id, first_seen, last_seen)
            VALUES ('legacy-user', 't', 't');
            INSERT INTO jobs (user_id, kind, payload, status, created_at)
            VALUES (1, 'text', '{"model":"lstm","start":"Я"}', 'queued', 't');
            """
        )
        conn.commit()
        conn.close()
        store = JobStore(path)
        self.assertTrue(store.route_one(resolve_worker_kind))
        row = store.get_job(1)
        self.assertEqual(row["status"], "assigned")
        self.assertEqual(row["worker_kind"], "text_lstm")


if __name__ == "__main__":
    unittest.main()
