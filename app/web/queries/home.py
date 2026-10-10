"""Admin home dashboard: cluster, KPI, bars, audience, storage, recent jobs."""
from __future__ import annotations

import time
from datetime import timedelta, timezone
from pathlib import Path

from app.web.formatters import mask_discord_id
from app.web.queries.common import ROOT, STATUSES, _connect_ro, _now_utc, _parse_ts, _table_exists
from app.web.queries.models import PROD_CHECKPOINT_NAMES, _read_sidecar_meta
from app.web.queries.workers import (
    SERVICE_TO_WORKER_META,
    SERVICES,
    WORKER_ACCEPTS,
    _format_bytes,
    _gpu_stats,
    _load_meta_map,
    _match_services,
    _system_stats,
)
from core.bus.queue import (
    ACTIVE_SEEN_WINDOW_SEC,
    BAR_ERROR_PCT,
    BAR_WARN_PCT,
    MAX_QUEUED_JOBS,
    ORCHESTRATOR_FRESH_SEC,
    WAIT_STATUSES,
    WORKER_META_FRESH_SEC,
)

_GPU_CACHE: dict = {"t": 0.0, "v": None}
_GPU_CACHE_SEC = 3.0

GPU_WORKER_KEYS = frozenset(
    k for k, wk in SERVICE_TO_WORKER_META.items() if wk.startswith("text_") or wk.startswith("pic_") or wk == "mood"
)


def _gpu_stats_cached():
    now = time.monotonic()
    if now - float(_GPU_CACHE["t"]) < _GPU_CACHE_SEC:
        return _GPU_CACHE["v"]
    try:
        val = _gpu_stats()
    except Exception:
        val = None
    _GPU_CACHE["t"] = now
    _GPU_CACHE["v"] = val
    return val


def _bar_level(pct: float | None) -> str:
    if pct is None:
        return "na"
    if pct >= BAR_ERROR_PCT:
        return "error"
    if pct >= BAR_WARN_PCT:
        return "warn"
    return "ok"


def _meta_age_sec(meta: dict | None) -> float | None:
    if not meta:
        return None
    dt = _parse_ts(meta.get("updated_at"))
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (_now_utc() - dt).total_seconds())


def _health(proc_running: bool, meta: dict | None, *, fresh_sec: float) -> str:
    age = _meta_age_sec(meta)
    meta_fresh = age is not None and age <= fresh_sec
    if proc_running and meta_fresh:
        return "ok"
    if proc_running and meta and not meta_fresh:
        return "warn"
    if proc_running and not meta:
        return "warn"
    if not proc_running and meta_fresh:
        return "warn"
    return "error"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    if path.is_file():
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += int(p.stat().st_size)
                except OSError:
                    pass
    except OSError:
        pass
    return total


_WORKER_LABELS = {
    "worker_lstm": "lstm",
    "worker_lstm_word": "lstm-word",
    "worker_transformer": "transformer",
    "worker_gpt": "gpt",
    "worker_deepseek": "deepseek",
    "worker_bert_gen": "bert-gen",
    "worker_mood": "mood",
    "worker_still": "pic still",
    "worker_video": "pic video",
}


def _fmt_age(sec: float | None) -> str | None:
    if sec is None:
        return None
    if sec < 60:
        return f"{int(sec)} с назад"
    if sec < 3600:
        return f"{int(sec // 60)} мин назад"
    return f"{int(sec // 3600)} ч назад"


def _service_card(
    *,
    sid: str,
    label: str,
    health: str,
    running: bool,
    procs: list | None = None,
    meta: dict | None = None,
    detail: str | None = None,
    group: str | None = None,
) -> dict:
    procs = procs or []
    pids = ", ".join(str(p) for p, _ in procs) if procs else ""
    age = _meta_age_sec(meta)
    bits = []
    if health:
        bits.append(health)
    if pids:
        bits.append(f"pid {pids}")
    age_s = _fmt_age(age)
    if age_s:
        bits.append(age_s)
    if detail:
        bits.append(detail)
    return {
        "id": sid,
        "label": label,
        "health": health,
        "running": running,
        "pids": pids,
        "heartbeat_sec": age,
        "detail": " / ".join(bits) if bits else health,
        "group": group,
    }


def _cluster(db_path: Path, proc_map: dict, meta_map: dict, db_ok: bool) -> dict:
    disc_procs = proc_map.get("discord") or []
    orch_procs = proc_map.get("orchestrator") or []
    web_procs = proc_map.get("web") or []
    disc_meta = meta_map.get("discord")
    orch_meta = meta_map.get("orchestrator")

    # Discord/web: process presence is enough (no meta heartbeat yet).
    disc_h = "ok" if disc_procs else "error"
    web_h = "ok" if web_procs else "error"
    orch_h = _health(bool(orch_procs), orch_meta, fresh_sec=ORCHESTRATOR_FRESH_SEC)

    gpu_up = other_up = 0
    gpu_exp = other_exp = 0
    worker_details = []
    for key, _script in SERVICES:
        if not key.startswith("worker_"):
            continue
        running = bool(proc_map.get(key))
        wk = SERVICE_TO_WORKER_META.get(key)
        meta = meta_map.get(f"worker:{wk}") if wk else None
        h = _health(running, meta, fresh_sec=WORKER_META_FRESH_SEC)
        is_gpu = key in GPU_WORKER_KEYS
        if is_gpu:
            gpu_exp += 1
            if running:
                gpu_up += 1
        else:
            other_exp += 1
            if running:
                other_up += 1
        if key.startswith("worker_") and key in (
            "worker_lstm",
            "worker_lstm_word",
            "worker_transformer",
            "worker_gpt",
            "worker_deepseek",
            "worker_bert_gen",
        ):
            group = "text"
        elif key == "worker_mood":
            group = "mood"
        else:
            group = "pic"
        worker_details.append(
            _service_card(
                sid=key,
                label=_WORKER_LABELS.get(key, key.replace("worker_", "")),
                health=h,
                running=running,
                procs=proc_map.get(key) or [],
                meta=meta,
                detail=WORKER_ACCEPTS.get(wk) if wk else None,
                group=group,
            )
        )

    models = []
    models_warn = 0
    ckpt_dir = ROOT / "checkpoints"
    if ckpt_dir.is_dir():
        for name in sorted(PROD_CHECKPOINT_NAMES):
            path = ckpt_dir / name
            present = path.is_file()
            side = _read_sidecar_meta(path) if present else None
            missing_meta = present and side is None
            if missing_meta or not present:
                models_warn += 1
            mh = "error" if not present else ("warn" if missing_meta else "ok")
            models.append(
                {
                    "id": f"model:{name}",
                    "name": name,
                    "label": name.replace(".pt", ""),
                    "present": present,
                    "missing_metadata": missing_meta,
                    "health": mh,
                    "detail": (
                        "нет файла"
                        if not present
                        else ("нет .meta.json" if missing_meta else "prod")
                    ),
                }
            )

    workers_health = (
        "ok"
        if gpu_up == gpu_exp and other_up == other_exp and gpu_exp
        else ("warn" if (gpu_up + other_up) else "error")
    )

    nodes = [
        {
            "id": "discord",
            "label": "Discord",
            "health": disc_h,
            "running": bool(disc_procs),
            "pids": ", ".join(str(p) for p, _ in disc_procs) if disc_procs else "",
            "detail": None,
        },
        {
            "id": "orchestrator",
            "label": "Оркестратор",
            "health": orch_h,
            "running": bool(orch_procs),
            "pids": ", ".join(str(p) for p, _ in orch_procs) if orch_procs else "",
            "detail": None,
        },
        {
            "id": "queue",
            "label": "Очередь + SQLite",
            "health": "ok" if db_ok else "error",
            "running": db_ok,
            "pids": "",
            "detail": str(db_path.name) if db_ok else "нет БД",
        },
        {
            "id": "workers",
            "label": "Воркеры",
            "health": workers_health,
            "running": (gpu_up + other_up) > 0,
            "gpu_up": gpu_up,
            "gpu_expected": gpu_exp,
            "other_up": other_up,
            "other_expected": other_exp,
            "workers": worker_details,
            "detail": f"{gpu_up + other_up}/{gpu_exp + other_exp}",
        },
        {
            "id": "models",
            "label": "Модели",
            "health": "ok" if models and models_warn == 0 else ("warn" if models else "error"),
            "running": bool(models),
            "models": models,
            "detail": f"warn={models_warn}" if models_warn else None,
        },
    ]

    edges = [
        {"from": "discord", "to": "queue", "count": 0, "label": "заявки"},
        {"from": "queue", "to": "orchestrator", "count": 0, "label": "queued"},
        {"from": "orchestrator", "to": "workers", "count": 0, "label": "assigned+running"},
        {"from": "workers", "to": "models", "count": 0, "label": "weights"},
    ]

    diagram = {
        "ingress": [
            _service_card(
                sid="discord",
                label="Discord-бот",
                health=disc_h,
                running=bool(disc_procs),
                procs=disc_procs,
                meta=disc_meta,
            ),
            _service_card(
                sid="web",
                label="Админка",
                health=web_h,
                running=bool(web_procs),
                procs=web_procs,
            ),
        ],
        "bus": [
            _service_card(
                sid="orchestrator",
                label="Оркестратор",
                health=orch_h,
                running=bool(orch_procs),
                procs=orch_procs,
                meta=orch_meta,
            ),
            _service_card(
                sid="queue",
                label="Очередь",
                health="ok" if db_ok else "error",
                running=db_ok,
                detail=str(db_path.name) if db_ok else "нет БД",
            ),
            _service_card(
                sid="sqlite",
                label="SQLite bot.db",
                health="ok" if db_ok else "error",
                running=db_ok,
                detail="WAL" if db_ok else "offline",
            ),
        ],
        "workers": {
            "text": [w for w in worker_details if w.get("group") == "text"],
            "mood": [w for w in worker_details if w.get("group") == "mood"],
            "pic": [w for w in worker_details if w.get("group") == "pic"],
        },
        "models": models,
    }
    return {"nodes": nodes, "edges": edges, "diagram": diagram}


def _kpi(conn, db_ok: bool) -> dict:
    now = _now_utc()
    day_ago = (now - timedelta(days=1)).isoformat()
    week_ago = (now - timedelta(days=7)).isoformat()
    active = 0
    done_24h = 0
    error_24h = 0
    users_active = 0
    guilds = 0
    spark: list[dict] = []
    if not db_ok or conn is None:
        return {
            "active": 0,
            "done_24h": 0,
            "error_24h": 0,
            "users_active": 0,
            "guilds": 0,
            "sparkline_7d": [],
        }
    active = int(
        conn.execute(
            f"SELECT COUNT(*) FROM jobs WHERE status IN ({','.join('?' * len(WAIT_STATUSES))})",
            WAIT_STATUSES,
        ).fetchone()[0]
    )
    done_24h = int(
        conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status='done' AND finished_at >= ?",
            (day_ago,),
        ).fetchone()[0]
    )
    error_24h = int(
        conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status='error' AND finished_at >= ?",
            (day_ago,),
        ).fetchone()[0]
    )
    seen_cut = (now - timedelta(seconds=ACTIVE_SEEN_WINDOW_SEC)).isoformat()
    users_active = int(
        conn.execute(
            "SELECT COUNT(*) FROM users WHERE last_seen >= ? AND discord_id != 'cli'",
            (seen_cut,),
        ).fetchone()[0]
    )
    if _table_exists(conn, "guilds"):
        guilds = int(conn.execute("SELECT COUNT(*) FROM guilds").fetchone()[0])
    rows = conn.execute(
        "SELECT created_at FROM jobs WHERE created_at >= ?", (week_ago,)
    ).fetchall()
    counts = [0] * 7
    for r in rows:
        dt = _parse_ts(r["created_at"])
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (now.date() - dt.astimezone(timezone.utc).date()).days
        if 0 <= delta < 7:
            counts[6 - delta] += 1
    spark = [
        {"t": (now - timedelta(days=6 - i)).date().isoformat(), "v": counts[i]}
        for i in range(7)
    ]
    return {
        "active": active,
        "done_24h": done_24h,
        "error_24h": error_24h,
        "users_active": users_active,
        "guilds": guilds,
        "sparkline_7d": spark,
    }


def _bars(conn, db_ok: bool) -> dict:
    cpu_pct, mem_used, mem_total = _system_stats()
    gpu = _gpu_stats_cached()
    queued = 0
    if db_ok and conn is not None:
        queued = int(
            conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status='queued'"
            ).fetchone()[0]
        )
    q_pct = (100.0 * queued / MAX_QUEUED_JOBS) if MAX_QUEUED_JOBS else None
    mem_pct = None
    if mem_used is not None and mem_total:
        mem_pct = 100.0 * float(mem_used) / float(mem_total)
    gpu_pct = None
    if gpu and isinstance(gpu, list) and gpu:
        try:
            gpu_pct = float(gpu[0].get("util_pct"))
        except (TypeError, ValueError):
            gpu_pct = None
    return {
        "queue": {
            "value": queued,
            "max": MAX_QUEUED_JOBS,
            "pct": q_pct,
            "level": _bar_level(q_pct),
        },
        "cpu": {
            "value": cpu_pct,
            "max": 100,
            "pct": cpu_pct,
            "level": _bar_level(cpu_pct),
        },
        "mem": {
            "value": mem_used,
            "max": mem_total,
            "pct": mem_pct,
            "level": _bar_level(mem_pct),
        },
        "gpu": {
            "value": gpu_pct,
            "max": 100,
            "pct": gpu_pct,
            "level": _bar_level(gpu_pct) if gpu_pct is not None else "na",
            "detail": gpu,
        },
        "thresholds": {"warn": BAR_WARN_PCT, "error": BAR_ERROR_PCT},
    }


def _series_7d_empty(now) -> list[dict]:
    return [
        {"t": (now - timedelta(days=6 - i)).date().isoformat(), "v": 0}
        for i in range(7)
    ]


def _heatmap(conn, db_ok: bool) -> dict:
    rows_n, cols_n = 7, 24
    grid = [[0] * cols_n for _ in range(rows_n)]
    if db_ok and conn is not None:
        now = _now_utc()
        week_ago = (now - timedelta(days=7)).isoformat()
        for row in conn.execute(
            "SELECT created_at FROM jobs WHERE created_at >= ?", (week_ago,)
        ).fetchall():
            dt = _parse_ts(row["created_at"])
            if dt is None:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = (now.date() - dt.astimezone(timezone.utc).date()).days
            if 0 <= delta < 7:
                d_idx = 6 - delta
                h_idx = dt.astimezone(timezone.utc).hour
                grid[d_idx][h_idx] += 1
    cells = [
        {"d": d, "h": h, "v": grid[d][h]} for d in range(rows_n) for h in range(cols_n)
    ]
    return {"rows": rows_n, "cols": cols_n, "cells": cells}


def _audience(conn, db_ok: bool, *, kpi: dict, bot_stats: dict) -> dict:
    """Bot audience: snapshot strip + 7d unique job users and Discord members trend."""
    now = _now_utc()
    week_ago = (now - timedelta(days=7)).isoformat()
    latest = (bot_stats or {}).get("latest") or {}
    online_ok = bool((bot_stats or {}).get("members_online_available"))
    strip = {
        "guilds": int(kpi.get("guilds") or 0),
        "users_active": int(kpi.get("users_active") or 0),
        "users_total": 0,
        "members_total": latest.get("members_total"),
        "members_online": latest.get("members_online") if online_ok else None,
    }
    users_7d = _series_7d_empty(now)
    members_7d = _series_7d_empty(now)
    if not db_ok or conn is None:
        return {"strip": strip, "users_7d": users_7d, "members_7d": members_7d}

    strip["users_total"] = int(
        conn.execute(
            "SELECT COUNT(*) FROM users WHERE discord_id != 'cli'"
        ).fetchone()[0]
    )

    day_users: dict[int, set] = {i: set() for i in range(7)}
    rows = conn.execute(
        """
        SELECT j.created_at, j.user_id
        FROM jobs j
        JOIN users u ON u.id = j.user_id
        WHERE j.created_at >= ? AND u.discord_id != 'cli'
        """,
        (week_ago,),
    ).fetchall()
    for r in rows:
        dt = _parse_ts(r["created_at"])
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (now.date() - dt.astimezone(timezone.utc).date()).days
        if 0 <= delta < 7:
            day_users[6 - delta].add(int(r["user_id"]))
    users_7d = [
        {"t": (now - timedelta(days=6 - i)).date().isoformat(), "v": len(day_users[i])}
        for i in range(7)
    ]

    if _table_exists(conn, "bot_stats"):
        day_members: dict[int, int] = {}
        for r in conn.execute(
            "SELECT ts, members_total FROM bot_stats WHERE ts >= ? ORDER BY ts",
            (week_ago,),
        ).fetchall():
            dt = _parse_ts(r["ts"])
            if dt is None:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = (now.date() - dt.astimezone(timezone.utc).date()).days
            if 0 <= delta < 7:
                day_members[6 - delta] = int(r["members_total"] or 0)
        members_7d = [
            {
                "t": (now - timedelta(days=6 - i)).date().isoformat(),
                "v": day_members.get(i, 0),
            }
            for i in range(7)
        ]

    return {"strip": strip, "users_7d": users_7d, "members_7d": members_7d}


def _storage() -> dict:
    items = []
    for label, rel in (
        ("БД / data", "data"),
        ("Чекпоинты", "checkpoints"),
        ("Логи / var", "var"),
    ):
        path = ROOT / rel
        n = _dir_size(path)
        items.append({"label": label, "path": rel, "bytes": n, "human": _format_bytes(n)})
    return {"items": items}


def _recent_jobs(conn, db_ok: bool, limit: int = 12) -> list[dict]:
    if not db_ok or conn is None:
        return []
    rows = conn.execute(
        """
        SELECT j.id, j.kind, j.status, j.created_at, j.started_at, j.finished_at,
               j.worker_kind, u.discord_id AS user
        FROM jobs j
        JOIN users u ON u.id = j.user_id
        ORDER BY j.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    out = []
    for r in rows:
        stages = [
            {"key": "created", "at": r["created_at"]},
            {"key": "started", "at": r["started_at"]},
            {"key": "finished", "at": r["finished_at"]},
        ]
        out.append(
            {
                "id": int(r["id"]),
                "id_str": str(r["id"]),
                "user_mask": mask_discord_id(r["user"]),
                "kind": r["kind"],
                "worker_kind": r["worker_kind"],
                "status": r["status"],
                "created_at": r["created_at"],
                "stages": stages,
            }
        )
    return out


def _bot_stats_latest(conn, db_ok: bool) -> dict:
    empty = {
        "latest": None,
        "members_online_available": False,
    }
    if not db_ok or conn is None or not _table_exists(conn, "bot_stats"):
        return empty
    row = conn.execute(
        """
        SELECT ts, guilds, members_total, members_online, latency_ms
        FROM bot_stats ORDER BY ts DESC LIMIT 1
        """
    ).fetchone()
    if row is None:
        return empty
    return {
        "latest": {
            "ts": row["ts"],
            "guilds": row["guilds"],
            "members_total": row["members_total"],
            "members_online": row["members_online"],
            "latency_ms": row["latency_ms"],
        },
        "members_online_available": row["members_online"] is not None,
    }


def api_home(db_path: Path) -> dict:
    db_path = Path(db_path)
    db_ok = db_path.is_file()
    proc_map = _match_services()
    conn = None
    meta_map: dict = {}
    by_status = {s: 0 for s in STATUSES}
    try:
        if db_ok:
            conn = _connect_ro(db_path)
            meta_map = _load_meta_map(conn)
            for row in conn.execute(
                "SELECT status, COUNT(*) AS c FROM jobs GROUP BY status"
            ).fetchall():
                by_status[str(row["status"])] = int(row["c"])
            # edge counts from queue depth
            cluster = _cluster(db_path, proc_map, meta_map, db_ok)
            for e in cluster["edges"]:
                if e["from"] == "queue" and e["to"] == "orchestrator":
                    e["count"] = by_status.get("queued", 0)
                elif e["from"] == "orchestrator" and e["to"] == "workers":
                    e["count"] = by_status.get("assigned", 0) + by_status.get("running", 0)
                elif e["from"] == "discord" and e["to"] == "queue":
                    e["count"] = sum(by_status.get(s, 0) for s in WAIT_STATUSES)
            kpi = _kpi(conn, db_ok)
            bars = _bars(conn, db_ok)
            bot_stats = _bot_stats_latest(conn, db_ok)
            audience = _audience(conn, db_ok, kpi=kpi, bot_stats=bot_stats)
            heatmap = _heatmap(conn, db_ok)
            recent = _recent_jobs(conn, db_ok)
        else:
            cluster = _cluster(db_path, proc_map, meta_map, db_ok)
            kpi = _kpi(None, False)
            bars = _bars(None, False)
            bot_stats = _bot_stats_latest(None, False)
            audience = _audience(None, False, kpi=kpi, bot_stats=bot_stats)
            heatmap = _heatmap(None, False)
            recent = []
    finally:
        if conn is not None:
            conn.close()
    return {
        "generated_at": _now_utc().isoformat(),
        "db_ok": db_ok,
        "cluster": cluster,
        "kpi": kpi,
        "bars": bars,
        "audience": audience,
        "heatmap": heatmap,
        "storage": _storage(),
        "recent_jobs": recent,
        "bot_stats": bot_stats,
        "by_status": by_status,
        "thresholds": {"warn": BAR_WARN_PCT, "error": BAR_ERROR_PCT},
    }
