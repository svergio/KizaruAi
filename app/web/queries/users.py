"""User list/detail API and guild helpers (152-FZ: mask ids in lists)."""
from datetime import timedelta
from pathlib import Path

from app.web.formatters import mask_discord_id
from app.web.queries.home import _audience, _bot_stats_latest, _kpi
from core.bus.queue import ACTIVE_SEEN_WINDOW_SEC, JobStore, WAIT_STATUSES

from app.web.queries.common import (
    _connect_ro,
    _display_name,
    _iso_utc,
    _now_utc,
    _table_exists,
)


def _list_guilds(conn):
    if not _table_exists(conn, "guilds"):
        return []
    rows = conn.execute(
        """
        SELECT discord_id, name, member_count, updated_at
        FROM guilds
        ORDER BY name COLLATE NOCASE
        """
    ).fetchall()
    return [
        {
            "discord_id": row["discord_id"],
            "name": row["name"],
            "member_count": row["member_count"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


def api_user_stats(db_path: Path, discord_id: str) -> dict | None:
    store = JobStore(db_path, recover_running=False)
    if store.user_id_for_discord(discord_id) is None:
        return None
    return store.user_stats_for_discord(discord_id)


def _cli_user_sql(alias: str = "u", *, hide_cli: bool) -> str:
    if not hide_cli:
        return ""
    return (
        f" AND {alias}.discord_id != 'cli' "
        f"AND {alias}.discord_id NOT LIKE 'cli:%'"
    )


def _users_summary(conn, *, hide_cli: bool) -> dict:
    now = _now_utc()
    seen_15m = (now - timedelta(seconds=ACTIVE_SEEN_WINDOW_SEC)).isoformat()
    day_ago = (now - timedelta(days=1)).isoformat()
    week_ago = (now - timedelta(days=7)).isoformat()
    cli = _cli_user_sql("u", hide_cli=hide_cli)
    row = conn.execute(
        f"""
        SELECT
          COUNT(*) AS users_total,
          SUM(CASE WHEN u.last_seen >= ? THEN 1 ELSE 0 END) AS active_15m,
          SUM(CASE WHEN u.last_seen >= ? THEN 1 ELSE 0 END) AS active_24h,
          SUM(CASE WHEN u.last_seen >= ? THEN 1 ELSE 0 END) AS active_7d,
          SUM(CASE WHEN u.discord_id = 'cli' OR u.discord_id LIKE 'cli:%' THEN 1 ELSE 0 END) AS cli_rows
        FROM users u
        WHERE 1=1{cli}
        """,
        (seen_15m, day_ago, week_ago),
    ).fetchone()
    guilds = 0
    members_in_guilds = 0
    if _table_exists(conn, "guilds"):
        guilds = int(conn.execute("SELECT COUNT(*) FROM guilds").fetchone()[0])
        members_in_guilds = int(
            conn.execute(
                "SELECT COALESCE(SUM(member_count), 0) FROM guilds"
            ).fetchone()[0]
        )
    with_jobs = int(
        conn.execute(
            f"""
            SELECT COUNT(*) FROM (
              SELECT u.id
              FROM users u
              JOIN jobs j ON j.user_id = u.id
              WHERE 1=1{cli}
              GROUP BY u.id
            )
            """
        ).fetchone()[0]
    )
    jobs_active = int(
        conn.execute(
            f"""
            SELECT COUNT(*) FROM jobs j
            JOIN users u ON u.id = j.user_id
            WHERE j.status IN ({",".join("?" * len(WAIT_STATUSES))}){cli}
            """,
            WAIT_STATUSES,
        ).fetchone()[0]
    )
    return {
        "users_total": int(row["users_total"] or 0),
        "users_active_15m": int(row["active_15m"] or 0),
        "users_active_24h": int(row["active_24h"] or 0),
        "users_active_7d": int(row["active_7d"] or 0),
        "users_with_jobs": with_jobs,
        "guilds": guilds,
        "guild_members_sum": members_in_guilds,
        "jobs_active": jobs_active,
        "cli_rows": int(row["cli_rows"] or 0) if not hide_cli else 0,
    }


def _users_activity(conn, *, hide_cli: bool) -> dict:
    now = _now_utc()
    seen_15m = (now - timedelta(seconds=ACTIVE_SEEN_WINDOW_SEC)).isoformat()
    day_ago = (now - timedelta(days=1)).isoformat()
    week_ago = (now - timedelta(days=7)).isoformat()
    cli = _cli_user_sql("u", hide_cli=hide_cli)
    rec = conn.execute(
        f"""
        SELECT
          SUM(CASE WHEN u.last_seen >= ? THEN 1 ELSE 0 END) AS b15,
          SUM(CASE WHEN u.last_seen >= ? AND u.last_seen < ? THEN 1 ELSE 0 END) AS b24,
          SUM(CASE WHEN u.last_seen >= ? AND u.last_seen < ? THEN 1 ELSE 0 END) AS b7,
          SUM(CASE WHEN u.last_seen < ? OR u.last_seen IS NULL THEN 1 ELSE 0 END) AS b_old
        FROM users u
        WHERE 1=1{cli}
        """,
        (seen_15m, day_ago, seen_15m, week_ago, day_ago, week_ago),
    ).fetchone()
    by_recency = [
        {"label": "≤15 мин", "count": int(rec["b15"] or 0)},
        {"label": "15м–24ч", "count": int(rec["b24"] or 0)},
        {"label": "1–7 дней", "count": int(rec["b7"] or 0)},
        {"label": ">7 дней", "count": int(rec["b_old"] or 0)},
    ]
    jobs = conn.execute(
        f"""
        SELECT
          SUM(CASE WHEN cnt = 0 THEN 1 ELSE 0 END) AS j0,
          SUM(CASE WHEN cnt BETWEEN 1 AND 5 THEN 1 ELSE 0 END) AS j1,
          SUM(CASE WHEN cnt BETWEEN 6 AND 20 THEN 1 ELSE 0 END) AS j2,
          SUM(CASE WHEN cnt > 20 THEN 1 ELSE 0 END) AS j3
        FROM (
          SELECT u.id, COUNT(j.id) AS cnt
          FROM users u
          LEFT JOIN jobs j ON j.user_id = u.id
          WHERE 1=1{cli}
          GROUP BY u.id
        )
        """
    ).fetchone()
    by_jobs = [
        {"label": "0 задач", "count": int(jobs["j0"] or 0)},
        {"label": "1–5", "count": int(jobs["j1"] or 0)},
        {"label": "6–20", "count": int(jobs["j2"] or 0)},
        {"label": "21+", "count": int(jobs["j3"] or 0)},
    ]
    return {"by_recency": by_recency, "by_jobs": by_jobs}


def api_users_list(db_path: Path, *, hide_cli: bool = False) -> dict:
    """List: masked id only; no display_name / avatar_url (152-FZ)."""
    db_path = Path(db_path)
    empty = {
        "hide_cli": hide_cli,
        "generated_at": _now_utc().isoformat(),
        "summary": {
            "users_total": 0,
            "users_active_15m": 0,
            "users_active_24h": 0,
            "users_active_7d": 0,
            "users_with_jobs": 0,
            "guilds": 0,
            "guild_members_sum": 0,
            "jobs_active": 0,
            "cli_rows": 0,
        },
        "audience": {
            "strip": {
                "guilds": 0,
                "users_active": 0,
                "users_total": 0,
                "members_total": None,
                "members_online": None,
            },
            "users_7d": [],
            "members_7d": [],
        },
        "activity": {"by_recency": [], "by_jobs": []},
        "jobs_sparkline_7d": [],
        "bot_stats": {"latest": None, "members_online_available": False},
        "users": [],
    }
    if not db_path.is_file():
        return empty
    conn = _connect_ro(db_path)
    try:
        where = ""
        if hide_cli:
            where = "WHERE u.discord_id != 'cli' AND u.discord_id NOT LIKE 'cli:%'"
        rows = conn.execute(
            f"""
            SELECT u.id AS key, u.discord_id AS discord_id, u.first_seen, u.last_seen,
                   COUNT(j.id) AS total,
                   SUM(CASE WHEN j.status IN ('queued','assigned','running') THEN 1 ELSE 0 END) AS active
            FROM users u
            LEFT JOIN jobs j ON j.user_id = u.id
            {where}
            GROUP BY u.id
            ORDER BY u.last_seen DESC
            """
        ).fetchall()
        kpi = _kpi(conn, True)
        bot_stats = _bot_stats_latest(conn, True)
        audience = _audience(conn, True, kpi=kpi, bot_stats=bot_stats)
        summary = _users_summary(conn, hide_cli=hide_cli)
        activity = _users_activity(conn, hide_cli=hide_cli)
    finally:
        conn.close()
    users = [
        {
            "key": int(r["key"]),
            "id": mask_discord_id(r["discord_id"]),
            "id_mask": mask_discord_id(r["discord_id"]),
            "first_seen": _iso_utc(r["first_seen"]),
            "last_seen": _iso_utc(r["last_seen"]),
            "total": int(r["total"] or 0),
            "active": int(r["active"] or 0),
        }
        for r in rows
    ]
    return {
        "hide_cli": hide_cli,
        "generated_at": _now_utc().isoformat(),
        "summary": summary,
        "audience": audience,
        "activity": activity,
        "jobs_sparkline_7d": kpi.get("sparkline_7d") or [],
        "kpi_jobs": {
            "done_24h": int(kpi.get("done_24h") or 0),
            "error_24h": int(kpi.get("error_24h") or 0),
        },
        "bot_stats": bot_stats,
        "users": users,
    }


def _resolve_discord_id(db_path: Path, user_ref: str) -> str | None:
    """Accept discord_id or numeric users.key."""
    db_path = Path(db_path)
    if not db_path.is_file():
        return None
    conn = _connect_ro(db_path)
    try:
        if str(user_ref).isdigit():
            row = conn.execute(
                "SELECT discord_id FROM users WHERE id = ?", (int(user_ref),)
            ).fetchone()
            if row:
                return str(row["discord_id"])
        row = conn.execute(
            "SELECT discord_id FROM users WHERE discord_id = ?", (str(user_ref),)
        ).fetchone()
        return str(row["discord_id"]) if row else None
    finally:
        conn.close()


def api_user_detail(db_path: Path, discord_id: str) -> dict | None:
    """Public detail: masked id, no name/avatar until CSRF profile."""
    resolved = _resolve_discord_id(db_path, discord_id)
    if resolved is None:
        return None
    store = JobStore(db_path, recover_running=False)
    stats = store.user_stats_for_discord(resolved)
    conn = _connect_ro(db_path)
    try:
        row = conn.execute(
            "SELECT id AS key, discord_id AS id, first_seen, last_seen "
            "FROM users WHERE discord_id = ?",
            (resolved,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "key": int(row["key"]),
        "id": mask_discord_id(row["id"]),
        "id_mask": mask_discord_id(row["id"]),
        "display_name": None,
        "avatar_url": None,
        "first_seen": _iso_utc(row["first_seen"]),
        "last_seen": _iso_utc(row["last_seen"]),
        "total": stats.get("total", 0),
        "by_kind": stats.get("by_kind") or {},
        "by_status": {
            "queued": 0,
            "assigned": 0,
            "running": 0,
            "done": int(stats.get("done") or 0),
            "error": int(stats.get("error") or 0),
            "cancelled": int(stats.get("cancelled") or 0),
        },
        "mean_wait_sec": None,
        "mean_run_sec": stats.get("avg_duration_sec"),
    }


def api_user_profile(db_path: Path, user_ref: str) -> dict | None:
    """CSRF-gated full PII: discord id, display_name, avatar (Discord URL only)."""
    resolved = _resolve_discord_id(db_path, user_ref)
    if resolved is None:
        return None
    conn = _connect_ro(db_path)
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        dn = "display_name" if "display_name" in cols else "NULL AS display_name"
        av = "avatar_url" if "avatar_url" in cols else "NULL AS avatar_url"
        row = conn.execute(
            f"SELECT id AS key, discord_id AS id, {dn}, {av}, first_seen, last_seen "
            f"FROM users WHERE discord_id = ?",
            (resolved,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    avatar = row["avatar_url"] if "avatar_url" in row.keys() else None
    if avatar and not str(avatar).startswith("https://cdn.discordapp.com/"):
        if not str(avatar).startswith("https://"):
            avatar = None
    return {
        "key": int(row["key"]),
        "id": str(row["id"]),
        "id_mask": mask_discord_id(row["id"]),
        "display_name": _display_name(row),
        "avatar_url": avatar,
        "first_seen": _iso_utc(row["first_seen"]),
        "last_seen": _iso_utc(row["last_seen"]),
    }


def api_user_delete(db_path: Path, user_ref: str) -> dict:
    resolved = _resolve_discord_id(db_path, user_ref)
    store = JobStore(db_path, recover_running=False)
    if resolved is None:
        return {"ok": True, "deleted": False, "id_mask": mask_discord_id(user_ref)}
    ok = store.delete_user_by_discord_id(resolved)
    return {"ok": True, "deleted": ok, "id_mask": mask_discord_id(resolved)}
