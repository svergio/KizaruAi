import asyncio
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.bus.routing import BOT_KINDS, WORKER_KINDS, resolve_worker_kind
from core.bus.user_stats import aggregate_user_job_rows

MAX_PAYLOAD_BYTES = 12 * 1024 * 1024
MAX_QUEUED_JOBS = 64
MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024
MAX_ATTEMPTS = 3
COLD_LOAD_SEC = 120
MIN_LEASE_SEC = 180
LEASE_RENEW_SEC = 30
ORCHESTRATOR_FRESH_SEC = 15
WORKER_META_FRESH_SEC = 15
RETENTION_DAYS = 90
BOT_STATS_KEEP_DAYS = 30
ACTIVE_SEEN_WINDOW_SEC = 900
BAR_WARN_PCT = 70
BAR_ERROR_PCT = 90
DEFAULT_POLL_TIMEOUT = 1800
# Kinds that may touch CUDA (text/mood via cuda_for_job; pic_* via pick_device).
# At most one running job across GPU_KINDS system-wide (expired lease frees the slot).
GPU_KINDS = frozenset(WORKER_KINDS)
# Wall-clock cap per running job; after this lease is not renewed and job -> error "timeout".
MAX_RUNTIME_SEC = {
    "text_lstm": 300,
    "text_lstm_word": 300,
    "text_transformer": 360,
    "text_gpt": 420,
    "text_deepseek": 480,
    "text_bert_gen": 600,
    "mood": 60,
    "pic_still": 180,
    # Video is long-running (decode/encode batches); keep well above text caps.
    "pic_video": 1800,
}
FALLBACK_MAX_RUNTIME_SEC = 600
CANCEL_POLL_SEC = 1.0

WAIT_STATUSES = ("queued", "assigned", "running")
ACTIVE_STATUSES = ("queued", "assigned", "running")
TERMINAL_STATUSES = ("done", "error", "cancelled")
STATUS_BADGE_CLASSES = {
    "queued": "status-queued",
    "assigned": "status-assigned",
    "running": "status-running",
    "done": "status-done",
    "error": "status-error",
    "cancelled": "status-cancelled",
}

# Coarse fallbacks when bot.db has no recent done timings for a worker_kind.
DEFAULT_DURATION_SEC = {
    "text_lstm": 20,
    "text_lstm_word": 25,
    "text_transformer": 30,
    "text_gpt": 40,
    "text_deepseek": 45,
    "text_bert_gen": 60,
    "mood": 5,
    "pic_still": 15,
    "pic_video": 90,
}
FALLBACK_DURATION_SEC = 25
AVG_DURATION_SAMPLE = 20


def estimate_wait_seconds(ahead: int, avg_sec: float) -> int:
    """ETA until this job finishes: jobs ahead (incl. running) + own job."""
    ahead = max(0, int(ahead))
    avg = max(1.0, float(avg_sec))
    return max(1, int(round((ahead + 1) * avg)))


def format_eta_ru(sec: int) -> str:
    sec = max(1, int(sec))
    if sec < 60:
        return f"~{sec} сек"
    minutes = max(1, int(round(sec / 60.0)))
    if minutes < 60:
        return f"~{minutes} мин"
    hours = minutes // 60
    rem = minutes % 60
    if rem:
        return f"~{hours} ч {rem} мин"
    return f"~{hours} ч"


class ActiveJobError(Exception):
    pass


class PayloadTooLargeError(Exception):
    pass


class QueueFullError(Exception):
    pass


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _utc_now_dt():
    return datetime.now(timezone.utc)


def _lease_until_iso(worker_kind: str) -> str:
    sec = lease_seconds(worker_kind)
    return (_utc_now_dt() + timedelta(seconds=sec)).isoformat()


def lease_seconds(worker_kind: str) -> int:
    return max(
        MIN_LEASE_SEC,
        int(DEFAULT_DURATION_SEC.get(worker_kind, FALLBACK_DURATION_SEC)) + COLD_LOAD_SEC,
    )


def max_runtime_seconds(worker_kind: str) -> int:
    return int(MAX_RUNTIME_SEC.get(worker_kind, FALLBACK_MAX_RUNTIME_SEC))


def _pid_alive(pid) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetLastError(0)
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        # ACCESS_DENIED (5): process exists but we cannot open it
        return kernel32.GetLastError() == 5
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


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
                if "worker_pid" not in cols:
                    conn.execute("ALTER TABLE jobs ADD COLUMN worker_pid INTEGER")
                if "lease_until" not in cols:
                    conn.execute("ALTER TABLE jobs ADD COLUMN lease_until TEXT")
                if "attempts" not in cols:
                    conn.execute("ALTER TABLE jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
                self._drop_stale_job_checks(conn)
                self._migrate_users_profile(conn)
                self._migrate_cli_users(conn)
                self._migrate_bot_stats(conn)
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
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_jobs_user_kind ON jobs(user_id, kind)"
                )
                conn.commit()
            finally:
                conn.close()

    def _migrate_users_profile(self, conn):
        cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "display_name" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN display_name TEXT")
        if "avatar_url" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN avatar_url TEXT")

    def _migrate_cli_users(self, conn):
        now = _utc_now()
        cli_row = conn.execute(
            "SELECT id FROM users WHERE discord_id = 'cli'"
        ).fetchone()
        if cli_row is None:
            conn.execute(
                "INSERT INTO users (discord_id, first_seen, last_seen, display_name) "
                "VALUES ('cli', ?, ?, 'cli')",
                (now, now),
            )
            cli_id = int(
                conn.execute(
                    "SELECT id FROM users WHERE discord_id = 'cli'"
                ).fetchone()["id"]
            )
        else:
            cli_id = int(cli_row["id"])
        old = conn.execute(
            "SELECT id FROM users WHERE discord_id LIKE 'cli:%'"
        ).fetchall()
        if not old:
            return
        old_ids = [int(r["id"]) for r in old]
        placeholders = ",".join("?" * len(old_ids))
        conn.execute(
            f"UPDATE jobs SET user_id = ? WHERE user_id IN ({placeholders})",
            (cli_id, *old_ids),
        )
        conn.execute(
            f"DELETE FROM users WHERE id IN ({placeholders})",
            old_ids,
        )

    def _migrate_bot_stats(self, conn):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_stats (
                ts TEXT NOT NULL,
                guilds INTEGER,
                members_total INTEGER,
                members_online INTEGER,
                latency_ms REAL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bot_stats_ts ON bot_stats(ts)"
        )

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
                worker_kind TEXT,
                worker_pid INTEGER,
                lease_until TEXT,
                attempts INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            INSERT INTO jobs_new (
                id, user_id, kind, payload, status, created_at, started_at,
                finished_at, error, result, worker_kind, worker_pid, lease_until, attempts
            )
            SELECT id, user_id, kind, payload, status, created_at, started_at,
                   finished_at, error, result, worker_kind, worker_pid, lease_until,
                   COALESCE(attempts, 0)
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

    def recover_interrupted(self, worker_kind: str | None = None):
        scope = worker_kind if worker_kind is not None else self._recover_worker_kind
        now = _utc_now()
        now_dt = _utc_now_dt()
        with self._lock:
            conn = self._connect()
            try:
                if scope:
                    rows = conn.execute(
                        """
                        SELECT id, worker_pid, lease_until, attempts FROM jobs
                        WHERE status = 'running' AND worker_kind = ?
                        """,
                        (scope,),
                    ).fetchall()
                    for row in rows:
                        job_id = int(row["id"])
                        attempts = int(row["attempts"] or 0)
                        lease = row["lease_until"]
                        expired = False
                        if lease:
                            try:
                                expired = datetime.fromisoformat(str(lease)) <= now_dt
                            except (TypeError, ValueError):
                                expired = True
                        dead = not _pid_alive(row["worker_pid"])
                        if not dead and not expired:
                            continue
                        if attempts >= MAX_ATTEMPTS:
                            conn.execute(
                                """
                                UPDATE jobs
                                SET status = 'error', finished_at = ?, error = ?,
                                    worker_pid = NULL, lease_until = NULL
                                WHERE id = ?
                                """,
                                (now, "max attempts", job_id),
                            )
                        else:
                            conn.execute(
                                """
                                UPDATE jobs
                                SET status = 'assigned', started_at = NULL,
                                    worker_pid = NULL, lease_until = NULL, error = NULL
                                WHERE id = ?
                                """,
                                (job_id,),
                            )
                else:
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

    def get_own_running(self, worker_kind: str, worker_pid: int) -> dict | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT id, kind, worker_kind, status, started_at, worker_pid, lease_until, attempts
                    FROM jobs
                    WHERE status = 'running' AND worker_kind = ? AND worker_pid = ?
                    ORDER BY id ASC LIMIT 1
                    """,
                    (worker_kind, int(worker_pid)),
                ).fetchone()
                return None if row is None else dict(row)
            finally:
                conn.close()

    def renew_running_lease(self, worker_kind: str, worker_pid: int) -> str:
        """Renew lease or mark timeout. Returns 'ok' | 'timeout' | 'idle'."""
        row = self.get_own_running(worker_kind, worker_pid)
        if row is None:
            return "idle"
        started = row.get("started_at")
        if started:
            try:
                started_dt = datetime.fromisoformat(str(started))
                age = (_utc_now_dt() - started_dt).total_seconds()
            except (TypeError, ValueError):
                age = 0.0
            if age > max_runtime_seconds(worker_kind):
                self.fail(int(row["id"]), "timeout", int(worker_pid))
                return "timeout"
        lease_until = _lease_until_iso(worker_kind)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE jobs SET lease_until = ?
                    WHERE status = 'running' AND worker_kind = ? AND worker_pid = ?
                    """,
                    (lease_until, worker_kind, int(worker_pid)),
                )
                conn.commit()
            finally:
                conn.close()
        return "ok"

    def reap_offline_assigned(self) -> int:
        # Assigned jobs waiting on the global GPU slot stay assigned while their
        # worker process is alive; only offline kinds are marked error.
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, worker_kind FROM jobs
                    WHERE status = 'assigned' AND worker_kind IS NOT NULL
                    """
                ).fetchall()
            finally:
                conn.close()
        stale_ids = []
        for row in rows:
            wk = row["worker_kind"]
            if wk and self.worker_online(str(wk)):
                continue
            stale_ids.append(int(row["id"]))
        if not stale_ids:
            return 0
        changed = 0
        with self._lock:
            conn = self._connect()
            try:
                for job_id in stale_ids:
                    cur = conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'error', finished_at = ?, error = 'worker offline'
                        WHERE id = ? AND status = 'assigned'
                        """,
                        (now, job_id),
                    )
                    changed += cur.rowcount
                conn.commit()
            finally:
                conn.close()
        return changed

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

    def insert_bot_stats(
        self,
        *,
        guilds: int,
        members_total: int | None,
        members_online: int | None,
        latency_ms: float | None,
        ts: str | None = None,
    ) -> None:
        stamp = ts or _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                self._migrate_bot_stats(conn)
                conn.execute(
                    """
                    INSERT INTO bot_stats (ts, guilds, members_total, members_online, latency_ms)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (stamp, guilds, members_total, members_online, latency_ms),
                )
                conn.commit()
            finally:
                conn.close()

    def retention_cleanup(self) -> dict:
        """Drop old jobs/users noise and bot_stats; counters go to worker_meta."""
        now_dt = _utc_now_dt()
        jobs_cut = (now_dt - timedelta(days=RETENTION_DAYS)).isoformat()
        stats_cut = (now_dt - timedelta(days=BOT_STATS_KEEP_DAYS)).isoformat()
        deleted_jobs = 0
        deleted_stats = 0
        with self._lock:
            conn = self._connect()
            try:
                self._migrate_bot_stats(conn)
                cur = conn.execute(
                    """
                    DELETE FROM jobs
                    WHERE status IN ('done', 'error', 'cancelled')
                      AND COALESCE(finished_at, created_at) < ?
                    """,
                    (jobs_cut,),
                )
                deleted_jobs = int(cur.rowcount or 0)
                cur = conn.execute("DELETE FROM bot_stats WHERE ts < ?", (stats_cut,))
                deleted_stats = int(cur.rowcount or 0)
                payload = json.dumps(
                    {
                        "jobs_deleted": deleted_jobs,
                        "bot_stats_deleted": deleted_stats,
                        "ran_at": _utc_now(),
                        "retention_days": RETENTION_DAYS,
                        "bot_stats_keep_days": BOT_STATS_KEEP_DAYS,
                    },
                    ensure_ascii=False,
                )
                conn.execute(
                    """
                    INSERT INTO worker_meta (key, value, updated_at)
                    VALUES ('retention_cleanup', ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value, updated_at = excluded.updated_at
                    """,
                    (payload, _utc_now()),
                )
                conn.commit()
            finally:
                conn.close()
        return {"jobs_deleted": deleted_jobs, "bot_stats_deleted": deleted_stats}

    def delete_user_by_discord_id(self, discord_id: str) -> bool:
        """Delete user and their jobs in one transaction. Idempotent."""
        discord_id = self._normalize_discord_id(discord_id)
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id FROM users WHERE discord_id = ?", (discord_id,)
                ).fetchone()
                if row is None:
                    return False
                uid = int(row["id"])
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM jobs WHERE user_id = ?", (uid,))
                conn.execute("DELETE FROM users WHERE id = ?", (uid,))
                conn.commit()
                return True
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def delete_user_by_key(self, user_key: int) -> tuple[bool, str | None]:
        """Delete by users.id. Returns (changed, discord_id_or_None)."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id, discord_id FROM users WHERE id = ?", (int(user_key),)
                ).fetchone()
                if row is None:
                    return False, None
                uid = int(row["id"])
                did = str(row["discord_id"])
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM jobs WHERE user_id = ?", (uid,))
                conn.execute("DELETE FROM users WHERE id = ?", (uid,))
                conn.commit()
                return True, did
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
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

    def clear_meta(self, key: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM worker_meta WHERE key = ?", (key,))
                conn.commit()
            finally:
                conn.close()

    def worker_online(self, worker_kind: str) -> bool:
        key = f"worker:{worker_kind}"
        raw = self.get_meta(key)
        if raw is None:
            return False
        try:
            meta = json.loads(raw)
        except json.JSONDecodeError:
            self.clear_meta(key)
            return False
        pid = meta.get("pid") if isinstance(meta, dict) else None
        if not _pid_alive(pid):
            self.clear_meta(key)
            return False
        return True

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

    @staticmethod
    def _normalize_discord_id(discord_id: str) -> str:
        text = str(discord_id)
        if text.startswith("cli:"):
            return "cli"
        return text

    def _upsert_user_locked(self, conn, discord_id: str) -> int:
        discord_id = self._normalize_discord_id(discord_id)
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

    def update_user_profile(
        self,
        discord_id: str,
        *,
        display_name: str | None = None,
        avatar_url: str | None = None,
    ) -> None:
        discord_id = self._normalize_discord_id(discord_id)
        with self._lock:
            conn = self._connect()
            try:
                uid = self._upsert_user_locked(conn, discord_id)
                if display_name is not None:
                    conn.execute(
                        "UPDATE users SET display_name = ? WHERE id = ?",
                        (display_name, uid),
                    )
                if avatar_url is not None:
                    conn.execute(
                        "UPDATE users SET avatar_url = ? WHERE id = ?",
                        (avatar_url, uid),
                    )
                conn.commit()
            finally:
                conn.close()

    def cancel_job(self, job_id: int) -> bool:
        """Admin cancel: any owner, active statuses only."""
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'cancelled', finished_at = ?, error = 'cancelled',
                        worker_pid = NULL, lease_until = NULL
                    WHERE id = ? AND status IN ('queued', 'assigned', 'running')
                    """,
                    (now, int(job_id)),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def enqueue(self, discord_id: str, kind: str, payload: dict) -> int:
        if kind not in BOT_KINDS:
            raise ValueError(f"unknown kind: {kind}")
        discord_id = self._normalize_discord_id(discord_id)
        payload_json = json.dumps(payload, ensure_ascii=False)
        if len(payload_json.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise PayloadTooLargeError(len(payload_json.encode("utf-8")))
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                queued = conn.execute(
                    """
                    SELECT COUNT(*) AS c FROM jobs
                    WHERE status IN ('queued', 'assigned')
                    """
                ).fetchone()
                if int(queued["c"]) >= MAX_QUEUED_JOBS:
                    conn.execute("ROLLBACK")
                    raise QueueFullError(int(queued["c"]))
                user_id = self._upsert_user_locked(conn, discord_id)
                # Shared CLI user may enqueue many jobs; Discord users stay 1-active.
                if discord_id != "cli":
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
                    SELECT id, kind, worker_kind, payload, status, result, error,
                           worker_pid, lease_until, attempts
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

    def _resolve_row_worker_kind(self, kind: str, worker_kind: str | None, payload_raw: str) -> str | None:
        if worker_kind:
            return str(worker_kind)
        try:
            payload = json.loads(payload_raw)
            return resolve_worker_kind(kind, payload)
        except Exception:
            return None

    def avg_job_duration_sec(self, worker_kind: str, limit: int = AVG_DURATION_SAMPLE) -> float | None:
        limit = max(1, int(limit))
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT started_at, finished_at FROM jobs
                    WHERE status = 'done'
                      AND worker_kind = ?
                      AND started_at IS NOT NULL
                      AND finished_at IS NOT NULL
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (worker_kind, limit),
                ).fetchall()
            finally:
                conn.close()
        durations = []
        for row in rows:
            try:
                started = datetime.fromisoformat(str(row["started_at"]))
                finished = datetime.fromisoformat(str(row["finished_at"]))
                sec = (finished - started).total_seconds()
            except (TypeError, ValueError):
                continue
            if 0.2 <= sec <= 3600:
                durations.append(sec)
        if not durations:
            return None
        return float(sum(durations) / len(durations))

    def jobs_ahead_for_worker(self, job_id: int, worker_kind: str) -> int:
        """Active jobs for the same worker_kind that are ahead of job_id (FIFO)."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT status, created_at FROM jobs WHERE id = ?
                    """,
                    (job_id,),
                ).fetchone()
                if row is None or row["status"] not in ACTIVE_STATUSES:
                    return 0
                created_at = row["created_at"]
                rows = conn.execute(
                    """
                    SELECT id, kind, worker_kind, payload FROM jobs
                    WHERE status IN ('queued', 'assigned', 'running')
                      AND (created_at < ? OR (created_at = ? AND id < ?))
                    ORDER BY created_at ASC, id ASC
                    """,
                    (created_at, created_at, job_id),
                ).fetchall()
            finally:
                conn.close()
        ahead = 0
        for prev in rows:
            wk = self._resolve_row_worker_kind(
                prev["kind"], prev["worker_kind"], prev["payload"]
            )
            if wk == worker_kind:
                ahead += 1
        return ahead

    def estimate_queue_wait(self, job_id: int, worker_kind: str) -> dict:
        ahead = self.jobs_ahead_for_worker(job_id, worker_kind)
        measured = self.avg_job_duration_sec(worker_kind)
        avg_sec = measured if measured is not None else float(
            DEFAULT_DURATION_SEC.get(worker_kind, FALLBACK_DURATION_SEC)
        )
        eta_sec = estimate_wait_seconds(ahead, avg_sec)
        return {
            "job_id": job_id,
            "worker_kind": worker_kind,
            "ahead": ahead,
            "position": ahead + 1,
            "avg_sec": avg_sec,
            "eta_sec": eta_sec,
            "avg_from_history": measured is not None,
        }

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
                now = _utc_now()
                if worker_kind in GPU_KINDS:
                    kinds = tuple(sorted(GPU_KINDS))
                    placeholders = ",".join("?" * len(kinds))
                    busy = conn.execute(
                        f"""
                        SELECT 1 FROM jobs
                        WHERE status = 'running'
                          AND worker_kind IN ({placeholders})
                          AND (lease_until IS NULL OR lease_until > ?)
                        LIMIT 1
                        """,
                        (*kinds, now),
                    ).fetchone()
                else:
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
                pid = os.getpid()
                lease_until = _lease_until_iso(worker_kind)
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'running', started_at = ?,
                        worker_pid = ?, lease_until = ?,
                        attempts = COALESCE(attempts, 0) + 1
                    WHERE id = ? AND status = 'assigned'
                    """,
                    (now, pid, lease_until, row["id"]),
                )
                conn.commit()
                return dict(row)
            finally:
                conn.close()

    def complete(self, job_id: int, result: dict, worker_pid: int) -> int:
        now = _utc_now()
        blob = json.dumps(result, ensure_ascii=False)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'done', finished_at = ?, result = ?, error = NULL,
                        worker_pid = NULL, lease_until = NULL
                    WHERE id = ? AND status = 'running' AND worker_pid = ?
                    """,
                    (now, blob, job_id, int(worker_pid)),
                )
                conn.commit()
                return int(cur.rowcount)
            finally:
                conn.close()

    def fail(self, job_id: int, message: str, worker_pid: int) -> int:
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'error', finished_at = ?, error = ?, result = NULL,
                        worker_pid = NULL, lease_until = NULL
                    WHERE id = ? AND status = 'running' AND worker_pid = ?
                    """,
                    (now, message[:2000], job_id, int(worker_pid)),
                )
                conn.commit()
                return int(cur.rowcount)
            finally:
                conn.close()

    def job_status(self, job_id: int) -> str | None:
        row = self.get_job(job_id)
        return None if row is None else str(row["status"])

    def user_id_for_job(self, job_id: int) -> int | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT user_id FROM jobs WHERE id = ?", (int(job_id),)
                ).fetchone()
                return None if row is None else int(row["user_id"])
            finally:
                conn.close()

    def user_id_for_discord(self, discord_id: str) -> int | None:
        discord_id = self._normalize_discord_id(discord_id)
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id FROM users WHERE discord_id = ?", (discord_id,)
                ).fetchone()
                return None if row is None else int(row["id"])
            finally:
                conn.close()

    def user_stats(self, user_id: int) -> dict:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT kind, status, created_at, started_at, finished_at
                    FROM jobs WHERE user_id = ?
                    ORDER BY id ASC
                    """,
                    (int(user_id),),
                ).fetchall()
            finally:
                conn.close()
        payload = aggregate_user_job_rows([dict(r) for r in rows])
        payload["user_id"] = int(user_id)
        return payload

    def user_stats_for_discord(self, discord_id: str) -> dict:
        uid = self.user_id_for_discord(discord_id)
        if uid is None:
            payload = aggregate_user_job_rows([])
            payload["user_id"] = None
            return payload
        return self.user_stats(uid)

    def cancel(self, job_id: int, discord_id: str) -> bool:
        now = _utc_now()
        discord_id = self._normalize_discord_id(discord_id)
        with self._lock:
            conn = self._connect()
            try:
                user = conn.execute(
                    "SELECT id FROM users WHERE discord_id = ?", (discord_id,)
                ).fetchone()
                if user is None:
                    return False
                cur = conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'cancelled', finished_at = ?, error = 'cancelled',
                        worker_pid = NULL, lease_until = NULL
                    WHERE id = ? AND user_id = ?
                      AND status IN ('queued', 'assigned', 'running')
                    """,
                    (now, job_id, int(user["id"])),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def meta_fresh(self, key: str, max_age_sec: float) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT updated_at FROM worker_meta WHERE key = ?", (key,)
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return False
        try:
            updated = datetime.fromisoformat(str(row["updated_at"]))
        except (TypeError, ValueError):
            return False
        age = (_utc_now_dt() - updated).total_seconds()
        return age <= float(max_age_sec)


def _poll_row_result(row: dict) -> dict | None:
    status = row["status"]
    if status == "done":
        if not row["result"]:
            raise RuntimeError("job done without result")
        return json.loads(row["result"])
    if status == "error":
        raise RuntimeError(row["error"] or "job failed")
    if status == "cancelled":
        raise RuntimeError(row["error"] or "cancelled")
    return None


async def poll_until_done(
    store: JobStore,
    job_id: int,
    interval: float = 0.25,
    timeout: float = DEFAULT_POLL_TIMEOUT,
) -> dict:
    deadline = time.monotonic() + float(timeout)
    while True:
        row = await asyncio.to_thread(store.get_job, job_id)
        if row is None:
            raise RuntimeError("job not found")
        out = _poll_row_result(row)
        if out is not None:
            return out
        if time.monotonic() >= deadline:
            raise TimeoutError(f"job {job_id} poll timeout")
        await asyncio.sleep(interval)


def poll_until_done_sync(
    store: JobStore,
    job_id: int,
    interval: float = 0.25,
    timeout: float = DEFAULT_POLL_TIMEOUT,
) -> dict:
    deadline = time.monotonic() + float(timeout)
    while True:
        row = store.get_job(job_id)
        if row is None:
            raise RuntimeError("job not found")
        out = _poll_row_result(row)
        if out is not None:
            return out
        if time.monotonic() >= deadline:
            raise TimeoutError(f"job {job_id} poll timeout")
        time.sleep(interval)
