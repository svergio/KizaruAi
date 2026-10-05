import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from core.bus.routing import BOT_KINDS, WORKER_KINDS

MAX_PAYLOAD_BYTES = 12 * 1024 * 1024
MAX_QUEUED_JOBS = 64
MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024

WAIT_STATUSES = ("queued", "assigned", "running")
ACTIVE_STATUSES = ("queued", "assigned", "running")
TERMINAL_STATUSES = ("done", "error")


class ActiveJobError(Exception):
    pass


class PayloadTooLargeError(Exception):
    pass


class QueueFullError(Exception):
    pass


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, db_path: Path | str, recover_running: bool = False, worker_kind: str | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._recover_worker_kind = worker_kind
        self._init_db()
        if recover_running:
            self.recover_interrupted()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        discord_id TEXT NOT NULL UNIQUE,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS jobs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL REFERENCES users(id),
                        kind TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT,
                        error TEXT
                    );
                    CREATE TABLE IF NOT EXISTS worker_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS guilds (
                        discord_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        member_count INTEGER,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_jobs_status_created
                        ON jobs(status, created_at, id);
                    CREATE INDEX IF NOT EXISTS idx_jobs_user_status
                        ON jobs(user_id, status);
                    """
                )
                cols = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
                }
                if "result" not in cols:
                    conn.execute("ALTER TABLE jobs ADD COLUMN result TEXT")
                if "worker_kind" not in cols:
                    conn.execute("ALTER TABLE jobs ADD COLUMN worker_kind TEXT")
                self._drop_stale_job_checks(conn)
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_jobs_status_created
                        ON jobs(status, created_at, id)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_jobs_user_status
                        ON jobs(user_id, status)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_jobs_worker_assigned
                        ON jobs(worker_kind, status, created_at, id)
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def _drop_stale_job_checks(self, conn):
        # Старая bot.db запрещала status='assigned'. CHECK в SQLite не меняется ALTER-ом.
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
        ).fetchone()
        sql = (row[0] or "") if row else ""
        if "CHECK" not in sql.upper():
            return
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            CREATE TABLE jobs_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                error TEXT,
                result TEXT,
                worker_kind TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO jobs_new (
                id, user_id, kind, payload, status, created_at,
                started_at, finished_at, error, result, worker_kind
            )
            SELECT
                id, user_id, kind, payload, status, created_at,
                started_at, finished_at, error, result, worker_kind
            FROM jobs
            """
        )
        conn.execute("DROP TABLE jobs")
        conn.execute("ALTER TABLE jobs_new RENAME TO jobs")
        max_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM jobs").fetchone()[0]
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('jobs', 'jobs_new')")
        conn.execute(
            "INSERT INTO sqlite_sequence(name, seq) VALUES ('jobs', ?)",
            (max_id,),
        )
        conn.execute("PRAGMA foreign_keys=ON")

    def recover_interrupted(self):
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                if self._recover_worker_kind:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'assigned', started_at = NULL, error = NULL
                        WHERE status = 'running' AND worker_kind = ?
                        """,
                        (self._recover_worker_kind,),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'assigned', started_at = NULL
                        WHERE status = 'running' AND worker_kind IS NOT NULL
                        """
                    )
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'error', finished_at = ?, error = 'interrupted'
                        WHERE status = 'running' AND worker_kind IS NULL
                        """,
                        (now,),
                    )
                conn.commit()
            finally:
                conn.close()

    def set_meta(self, key: str, value: str):
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO worker_meta (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                    """,
                    (key, value, now),
                )
                conn.commit()
            finally:
                conn.close()

    def sync_guilds(self, entries: list[tuple[int | str, str, int | None]]):
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                for guild_id, name, member_count in entries:
                    gid = str(guild_id)
                    conn.execute(
                        """
                        INSERT INTO guilds (discord_id, name, member_count, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(discord_id) DO UPDATE SET
                            name = excluded.name,
                            member_count = excluded.member_count,
                            updated_at = excluded.updated_at
                        """,
                        (gid, name[:500], member_count, now),
                    )
                conn.commit()
            finally:
                conn.close()

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT value FROM worker_meta WHERE key = ?", (key,)
                ).fetchone()
                return None if row is None else str(row["value"])
            finally:
                conn.close()

    def worker_online(self, worker_kind: str) -> bool:
        return self.get_meta(f"worker:{worker_kind}") is not None

    def get_worker_models(self) -> dict | None:
        mapping = {
            "lstm": "text_lstm",
            "lstm-word": "text_lstm_word",
            "transformer": "text_transformer",
            "gpt": "text_gpt",
            "deepseek": "text_deepseek",
            "bert-gen": "text_bert_gen",
            "mood": "mood",
        }
        seen = any(self.worker_online(wk) for wk in WORKER_KINDS)
        if not seen:
            return None
        return {key: self.worker_online(wk) for key, wk in mapping.items()}

    def _upsert_user_locked(self, conn, discord_id: str) -> int:
        now = _utc_now()
        row = conn.execute(
            "SELECT id FROM users WHERE discord_id = ?", (discord_id,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE users SET last_seen = ? WHERE id = ?", (now, row["id"])
            )
            return int(row["id"])
        cur = conn.execute(
            "INSERT INTO users (discord_id, first_seen, last_seen) VALUES (?, ?, ?)",
            (discord_id, now, now),
        )
        return int(cur.lastrowid)

    def enqueue(self, discord_id: str, kind: str, payload: dict) -> int:
        if kind not in BOT_KINDS:
            raise ValueError(f"unknown kind: {kind}")
        payload_json = json.dumps(payload, ensure_ascii=False)
        if len(payload_json.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise PayloadTooLargeError(len(payload_json.encode("utf-8")))
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                queued = conn.execute(
                    "SELECT COUNT(*) AS c FROM jobs WHERE status = 'queued'"
                ).fetchone()
                if int(queued["c"]) >= MAX_QUEUED_JOBS:
                    conn.execute("ROLLBACK")
                    raise QueueFullError(int(queued["c"]))
                user_id = self._upsert_user_locked(conn, discord_id)
                active = conn.execute(
                    """
                    SELECT id FROM jobs
                    WHERE user_id = ? AND status IN ('queued', 'assigned', 'running')
                    LIMIT 1
                    """,
                    (user_id,),
                ).fetchone()
                if active:
                    conn.execute("ROLLBACK")
                    raise ActiveJobError(int(active["id"]))
                now = _utc_now()
                cur = conn.execute(
                    """
                    INSERT INTO jobs (user_id, kind, worker_kind, payload, status, created_at)
                    VALUES (?, ?, NULL, ?, 'queued', ?)
                    """,
                    (user_id, kind, payload_json, now),
                )
                job_id = int(cur.lastrowid)
                conn.commit()
            finally:
                conn.close()
        return job_id

    def get_job(self, job_id: int) -> dict | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT id, kind, worker_kind, payload, status, result, error
                    FROM jobs WHERE id = ?
                    """,
                    (job_id,),
                ).fetchone()
                if row is None:
                    return None
                return dict(row)
            finally:
                conn.close()

    def queue_position(self, job_id: int) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT status, created_at FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                if row is None:
                    return 0
                if row["status"] == "running":
                    return 1
                if row["status"] not in ("queued", "assigned"):
                    return 0
                created_at = row["created_at"]
                ahead = conn.execute(
                    """
                    SELECT COUNT(*) AS c FROM jobs
                    WHERE status IN ('queued', 'assigned', 'running')
                      AND (created_at < ? OR (created_at = ? AND id < ?))
                    """,
                    (created_at, created_at, job_id),
                ).fetchone()
                return int(ahead["c"]) + 1
            finally:
                conn.close()

    def route_one(self, resolve_fn) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """
                    SELECT id, kind, payload FROM jobs
                    WHERE status = 'queued'
                    ORDER BY created_at ASC, id ASC
                    LIMIT 1
                    """
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    return False
                job_id = int(row["id"])
                try:
                    payload = json.loads(row["payload"])
                    worker_kind = resolve_fn(row["kind"], payload)
                except Exception as exc:
                    now = _utc_now()
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'error', finished_at = ?, error = ?
                        WHERE id = ?
                        """,
                        (now, str(exc)[:2000], job_id),
                    )
                    conn.commit()
                    return True
                if worker_kind not in WORKER_KINDS:
                    now = _utc_now()
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'error', finished_at = ?, error = ?
                        WHERE id = ?
                        """,
                        (now, f"bad worker_kind: {worker_kind}", job_id),
                    )
                    conn.commit()
                    return True
                conn.execute(
                    """
                    UPDATE jobs SET status = 'assigned', worker_kind = ?
                    WHERE id = ? AND status = 'queued'
                    """,
                    (worker_kind, job_id),
                )
                conn.commit()
                return True
            finally:
                conn.close()

    def claim_assigned(self, worker_kind: str) -> dict | None:
        if worker_kind not in WORKER_KINDS:
            raise ValueError(f"unknown worker_kind: {worker_kind}")
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                busy = conn.execute(
                    """
                    SELECT 1 FROM jobs
                    WHERE status = 'running' AND worker_kind = ?
                    LIMIT 1
                    """,
                    (worker_kind,),
                ).fetchone()
                if busy:
                    conn.execute("ROLLBACK")
                    return None
                row = conn.execute(
                    """
                    SELECT id, kind, worker_kind, payload FROM jobs
                    WHERE status = 'assigned' AND worker_kind = ?
                    ORDER BY created_at ASC, id ASC
                    LIMIT 1
                    """,
                    (worker_kind,),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    return None
                now = _utc_now()
                conn.execute(
                    """
                    UPDATE jobs SET status = 'running', started_at = ?
                    WHERE id = ? AND status = 'assigned'
                    """,
                    (now, row["id"]),
                )
                conn.commit()
                return dict(row)
            finally:
                conn.close()

    def complete(self, job_id: int, result: dict):
        now = _utc_now()
        blob = json.dumps(result, ensure_ascii=False)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'done', finished_at = ?, result = ?, error = NULL
                    WHERE id = ?
                    """,
                    (now, blob, job_id),
                )
                conn.commit()
            finally:
                conn.close()

    def fail(self, job_id: int, message: str):
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'error', finished_at = ?, error = ?, result = NULL
                    WHERE id = ?
                    """,
                    (now, message[:2000], job_id),
                )
                conn.commit()
            finally:
                conn.close()


async def poll_until_done(store: JobStore, job_id: int, interval: float = 0.25) -> dict:
    while True:
        row = await asyncio.to_thread(store.get_job, job_id)
        if row is None:
            raise RuntimeError("job not found")
        status = row["status"]
        if status == "done":
            if not row["result"]:
                raise RuntimeError("job done without result")
            return json.loads(row["result"])
        if status == "error":
            raise RuntimeError(row["error"] or "job failed")
        await asyncio.sleep(interval)
