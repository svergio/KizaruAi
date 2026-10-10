"""Job queue shaping and job API endpoints."""
import json
import re
from pathlib import Path

from app.web.contracts import PAGE_SIZE
from core.bus.queue import WAIT_STATUSES

from app.web.queries.common import (
    ACTIVE_LIMIT,
    FIELD_HEAD_CHARS,
    HISTORY_LIMIT,
    STATUSES,
    STATUS_LABELS_RU,
    WORKER_KINDS_DB,
    _connect_ro,
    _display_name,
    _freshness,
    _human_message_from_payload,
    _iso_utc,
    _loose_fields,
    _one_line,
    _table_exists,
    _truncate_text,
    _who_parts,
)


def _prompt_summary(kind: str, head: str, total_len: int) -> str:
    if not head:
        return ""
    try:
        data = json.loads(head)
        if not isinstance(data, dict):
            data = {}
    except json.JSONDecodeError:
        data = _loose_fields(head)
    slim = {k: v for k, v in data.items() if k != "data_b64"}
    text = _human_message_from_payload(kind, json.dumps(slim, ensure_ascii=False))
    if total_len > len(head):
        note = f"(вложение не показано, payload {total_len} байт)"
        text = f"{text} {note}".strip()
    return _truncate_text(text, 700)


def _result_view(head: str, total_len: int) -> str:
    if not head and not total_len:
        return ""
    data = None
    try:
        parsed = json.loads(head)
        if isinstance(parsed, dict):
            data = parsed
    except json.JSONDecodeError:
        data = None
    if data is None:
        fields = _loose_fields(head)
        kind = str(fields.get("type") or "")
        if kind == "message":
            content = fields.get("content")
            if content:
                return _truncate_text(str(content), 1200)
            return "текст сообщения обрезан при чтении"
        if kind == "chunks":
            return "результат: chunks (длинный текст, на панели обрезан)"
        if kind == "file" or total_len > FIELD_HEAD_CHARS:
            name = fields.get("filename") or "файл"
            note = fields.get("content") or ""
            extra = f" — {note}" if note else ""
            return f"файл: {name}{extra}. base64 не показан ({total_len} байт JSON)"
        cleaned = re.sub(
            r'"data_b64"\s*:\s*"[^"]*',
            '"data_b64":"<скрыто>"',
            head,
        )
        return _truncate_text(cleaned, 500)
    kind = data.get("type")
    if kind == "file" or "data_b64" in data:
        name = data.get("filename") or "файл"
        note = data.get("content") or ""
        extra = f" — {note}" if note else ""
        n = len(str(data.get("data_b64") or ""))
        return f"файл: {name}{extra}. base64 не показан ({n} симв.)"
    if kind == "message":
        return _truncate_text(str(data.get("content") or ""), 1200)
    if kind == "chunks":
        chunks = data.get("chunks") or []
        if isinstance(chunks, list):
            return _truncate_text("\n".join(str(c) for c in chunks), 1200)
        return _truncate_text(str(chunks), 1200)
    slim = {k: v for k, v in data.items() if k != "data_b64"}
    if "data_b64" in data:
        slim["data_b64"] = "<скрыто>"
    return _truncate_text(json.dumps(slim, ensure_ascii=False), 1200)


def _shape_job(row) -> dict:
    kind = row["kind"] or ""
    status = row["status"] or ""
    payload_len = int(row["payload_len"] or 0)
    result_len = int(row["result_len"] or 0)
    prompt = _prompt_summary(kind, row["payload_head"] or "", payload_len)
    result = _result_view(row["result_head"] or "", result_len)
    error = _truncate_text(row["error"] or "", 1500)
    who_kind, who_id = _who_parts(row["actor"])
    if status == "error" and error:
        preview = _one_line(error)
    else:
        preview = _one_line(prompt)
    age = ""
    if status in ("done", "error"):
        age = _freshness(row["finished_at"], row["created_at"])
    return {
        "id": int(row["id"]),
        "kind": kind,
        "worker_kind": row["worker_kind"] or "",
        "status": status,
        "status_ru": STATUS_LABELS_RU.get(status, status),
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "who_kind": who_kind,
        "who_id": who_id,
        "prompt": prompt,
        "result": result,
        "error": error,
        "preview": preview,
        "age": age,
    }


_JOB_SELECT = """
SELECT j.id, j.kind, j.worker_kind, j.status,
       j.created_at, j.started_at, j.finished_at,
       substr(j.error, 1, 2000) AS error,
       substr(j.payload, 1, ?) AS payload_head,
       length(j.payload) AS payload_len,
       substr(j.result, 1, ?) AS result_head,
       length(j.result) AS result_len,
       u.discord_id AS actor
FROM jobs j
LEFT JOIN users u ON u.id = j.user_id
"""


def _fetch_jobs(conn, where_sql: str, order_sql: str, limit: int) -> list[dict]:
    if not _table_exists(conn, "jobs"):
        return []
    rows = conn.execute(
        _JOB_SELECT + f" WHERE {where_sql} ORDER BY {order_sql} LIMIT ?",
        (FIELD_HEAD_CHARS, FIELD_HEAD_CHARS, limit),
    ).fetchall()
    return [_shape_job(row) for row in rows]


def _active_jobs(conn) -> list[dict]:
    return _fetch_jobs(
        conn,
        "j.status IN ('queued', 'assigned', 'running')",
        "j.id DESC",
        ACTIVE_LIMIT,
    )


def _history_jobs(conn) -> list[dict]:
    return _fetch_jobs(
        conn,
        "j.status IN ('done', 'error')",
        "COALESCE(j.finished_at, j.created_at) DESC, j.id DESC",
        HISTORY_LIMIT,
    )


def _history_count(conn) -> int:
    if not _table_exists(conn, "jobs"):
        return 0
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM jobs WHERE status IN ('done', 'error')"
    ).fetchone()
    return int(row["c"]) if row else 0


def _active_by_actor(conn) -> dict[str, int]:
    out = {"cli": 0, "discord": 0}
    if not _table_exists(conn, "jobs") or not _table_exists(conn, "users"):
        return out
    rows = conn.execute(
        """
        SELECT
          CASE WHEN u.discord_id = 'cli' THEN 'cli' ELSE 'discord' END AS src,
          COUNT(*) AS c
        FROM jobs j
        JOIN users u ON u.id = j.user_id
        WHERE j.status IN ('queued', 'assigned', 'running')
        GROUP BY src
        """
    ).fetchall()
    for row in rows:
        src = row["src"]
        if src in out:
            out[src] = int(row["c"])
    return out


def _queue_by_status(conn):
    if not _table_exists(conn, "jobs"):
        return {s: 0 for s in STATUSES}
    rows = conn.execute(
        "SELECT status, COUNT(*) AS c FROM jobs GROUP BY status"
    ).fetchall()
    out = {s: 0 for s in STATUSES}
    for row in rows:
        st = row["status"]
        if st in out:
            out[st] = int(row["c"])
    return out


def _queue_by_worker(conn):
    if not _table_exists(conn, "jobs"):
        return {wk: {s: 0 for s in STATUSES} for wk in WORKER_KINDS_DB}
    rows = conn.execute(
        """
        SELECT COALESCE(worker_kind, '(не назначен)') AS wk, status, COUNT(*) AS c
        FROM jobs
        GROUP BY wk, status
        """
    ).fetchall()
    out: dict[str, dict[str, int]] = {wk: {s: 0 for s in STATUSES} for wk in WORKER_KINDS_DB}
    out["(не назначен)"] = {s: 0 for s in STATUSES}
    for row in rows:
        wk = row["wk"]
        if wk not in out:
            out[wk] = {s: 0 for s in STATUSES}
        st = row["status"]
        if st in out[wk]:
            out[wk][st] = int(row["c"])
    return out


def _jobs_total_by_worker_kind(conn):
    if not _table_exists(conn, "jobs"):
        return []
    rows = conn.execute(
        """
        SELECT COALESCE(worker_kind, '(не назначен)') AS wk, COUNT(*) AS c
        FROM jobs
        GROUP BY wk
        ORDER BY c DESC, wk
        """
    ).fetchall()
    return [(row["wk"], int(row["c"])) for row in rows]


def api_queue(db_path: Path) -> dict:
    db_path = Path(db_path)
    if not db_path.is_file():
        return {"jobs": []}
    conn = _connect_ro(db_path)
    try:
        rows = conn.execute(
            """
            SELECT j.id, u.discord_id, j.kind, j.status, j.worker_kind, j.created_at
            FROM jobs j JOIN users u ON u.id = j.user_id
            WHERE j.status IN ('queued', 'assigned', 'running')
            ORDER BY j.created_at ASC, j.id ASC
            """
        ).fetchall()
    finally:
        conn.close()
    return {"jobs": [dict(r) for r in rows]}


def api_jobs(db_path: Path, limit: int = 200) -> dict:
    db_path = Path(db_path)
    if not db_path.is_file():
        return {"jobs": []}
    conn = _connect_ro(db_path)
    try:
        rows = conn.execute(
            """
            SELECT j.id, u.discord_id, j.kind, j.status, j.worker_kind,
                   j.created_at, j.started_at, j.finished_at, j.error
            FROM jobs j JOIN users u ON u.id = j.user_id
            ORDER BY j.id DESC LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()
    return {"jobs": [dict(r) for r in rows]}


def api_queue_v2(db_path: Path) -> dict:
    db_path = Path(db_path)
    empty = {
        "jobs": [],
        "pipeline": {
            "clients": 0,
            "discord": 0,
            "cli": 0,
            "queued": 0,
            "assigned": 0,
            "running": 0,
        },
    }
    if not db_path.is_file():
        return empty
    conn = _connect_ro(db_path)
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        dn = "u.display_name" if "display_name" in cols else "NULL AS display_name"
        av = "u.avatar_url" if "avatar_url" in cols else "NULL AS avatar_url"
        rows = conn.execute(
            f"""
            SELECT j.id, u.discord_id AS user, {dn}, {av}, j.kind, j.status, j.worker_kind,
                   j.created_at, j.started_at
            FROM jobs j JOIN users u ON u.id = j.user_id
            WHERE j.status IN ('queued', 'assigned', 'running')
            ORDER BY j.created_at ASC, j.id ASC
            """
        ).fetchall()
        by_status = _queue_by_status(conn)
        source = _active_by_actor(conn)
    finally:
        conn.close()
    jobs = []
    for r in rows:
        jobs.append(
            {
                "id": int(r["id"]),
                "user": r["user"],
                "display_name": _display_name(r),
                "avatar_url": r["avatar_url"] if "avatar_url" in r.keys() else None,
                "kind": r["kind"],
                "worker_kind": r["worker_kind"],
                "status": r["status"],
                "created_at": _iso_utc(r["created_at"]),
                "started_at": _iso_utc(r["started_at"]),
                "eta_sec": None,
            }
        )
    discord = int(source.get("discord", 0))
    cli = int(source.get("cli", 0))
    return {
        "jobs": jobs,
        "pipeline": {
            "clients": discord + cli,
            "discord": discord,
            "cli": cli,
            "queued": int(by_status.get("queued", 0)),
            "assigned": int(by_status.get("assigned", 0)),
            "running": int(by_status.get("running", 0)),
        },
    }


def api_jobs_page(
    db_path: Path,
    *,
    status: str | None = None,
    kind: str | None = None,
    user: str | None = None,
    page: int = 1,
) -> dict:
    db_path = Path(db_path)
    page = max(1, int(page or 1))
    if not db_path.is_file():
        return {"page": page, "page_size": PAGE_SIZE, "total": 0, "jobs": []}
    where = ["1=1"]
    args: list = []
    if status:
        where.append("j.status = ?")
        args.append(status)
    if kind:
        where.append("j.kind = ?")
        args.append(kind)
    if user:
        where.append("u.discord_id = ?")
        args.append(user)
    where_sql = " AND ".join(where)
    conn = _connect_ro(db_path)
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        dn = "u.display_name" if "display_name" in cols else "NULL AS display_name"
        av = "u.avatar_url" if "avatar_url" in cols else "NULL AS avatar_url"
        total = int(
            conn.execute(
                f"""
                SELECT COUNT(*) AS c FROM jobs j
                JOIN users u ON u.id = j.user_id
                WHERE {where_sql}
                """,
                args,
            ).fetchone()["c"]
        )
        offset = (page - 1) * PAGE_SIZE
        rows = conn.execute(
            f"""
            SELECT j.id, u.discord_id AS user, {dn}, {av}, j.kind, j.worker_kind, j.status,
                   j.created_at, j.started_at, j.finished_at,
                   substr(j.error, 1, 500) AS error,
                   substr(j.payload, 1, ?) AS payload_head,
                   length(j.payload) AS payload_len
            FROM jobs j JOIN users u ON u.id = j.user_id
            WHERE {where_sql}
            ORDER BY j.id DESC
            LIMIT ? OFFSET ?
            """,
            (FIELD_HEAD_CHARS, *args, PAGE_SIZE, offset),
        ).fetchall()
    finally:
        conn.close()
    jobs = []
    for r in rows:
        preview = _one_line(
            _prompt_summary(r["kind"] or "", r["payload_head"] or "", int(r["payload_len"] or 0))
        )
        if r["status"] == "error" and r["error"]:
            preview = _one_line(r["error"])
        jobs.append(
            {
                "id": int(r["id"]),
                "user": r["user"],
                "display_name": _display_name(r),
                "avatar_url": r["avatar_url"] if "avatar_url" in r.keys() else None,
                "kind": r["kind"],
                "worker_kind": r["worker_kind"],
                "status": r["status"],
                "created_at": _iso_utc(r["created_at"]),
                "started_at": _iso_utc(r["started_at"]),
                "finished_at": _iso_utc(r["finished_at"]),
                "error": r["error"],
                "preview": preview,
            }
        )
    return {"page": page, "page_size": PAGE_SIZE, "total": total, "jobs": jobs}


def api_job_detail(db_path: Path, job_id: int) -> dict | None:
    db_path = Path(db_path)
    if not db_path.is_file():
        return None
    conn = _connect_ro(db_path)
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        dn = "u.display_name" if "display_name" in cols else "NULL AS display_name"
        av = "u.avatar_url" if "avatar_url" in cols else "NULL AS avatar_url"
        row = conn.execute(
            f"""
            SELECT j.id, u.discord_id AS user, {dn}, {av}, j.kind, j.worker_kind, j.status,
                   j.created_at, j.started_at, j.finished_at,
                   substr(j.error, 1, 2000) AS error,
                   substr(j.payload, 1, ?) AS payload_head,
                   length(j.payload) AS payload_len,
                   substr(j.result, 1, ?) AS result_head,
                   length(j.result) AS result_len,
                   j.worker_pid, j.lease_until, j.attempts
            FROM jobs j JOIN users u ON u.id = j.user_id
            WHERE j.id = ?
            """,
            (FIELD_HEAD_CHARS, FIELD_HEAD_CHARS, int(job_id)),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    status = row["status"] or ""
    return {
        "id": int(row["id"]),
        "user": row["user"],
        "display_name": _display_name(row),
        "avatar_url": row["avatar_url"] if "avatar_url" in row.keys() else None,
        "kind": row["kind"],
        "worker_kind": row["worker_kind"],
        "status": status,
        "created_at": _iso_utc(row["created_at"]),
        "started_at": _iso_utc(row["started_at"]),
        "finished_at": _iso_utc(row["finished_at"]),
        "payload_preview": _prompt_summary(
            row["kind"] or "", row["payload_head"] or "", int(row["payload_len"] or 0)
        ),
        "result_preview": _result_view(row["result_head"] or "", int(row["result_len"] or 0)),
        "error": row["error"],
        "worker_pid": row["worker_pid"],
        "lease_until": _iso_utc(row["lease_until"]),
        "attempts": int(row["attempts"] or 0),
        "cancellable": status in WAIT_STATUSES,
    }
