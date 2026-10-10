"""Dashboard aggregation, overview/stats/timeseries/logs API."""
import html
import json
from datetime import timedelta, timezone
from pathlib import Path

from app.web.contracts import TIMESERIES_METRICS, TIMESERIES_RANGES
from core.bus.queue import TERMINAL_STATUSES, WAIT_STATUSES

from app.web.queries.common import (
    ROOT,
    STATUSES,
    _connect_ro,
    _fmt_ts,
    _now_utc,
    _parse_ts,
    _utc_now_iso,
)
from app.web.queries.jobs import (
    _active_by_actor,
    _active_jobs,
    _history_count,
    _history_jobs,
    _jobs_total_by_worker_kind,
    _queue_by_status,
    _queue_by_worker,
)
from app.web.queries.users import _list_guilds
from app.web.queries.workers import (
    SERVICE_TO_WORKER_META,
    SERVICES,
    _load_meta_map,
    _HAS_PSUTIL,
)


def _collect_dashboard(db_path: Path):
    from app.web import queries as q

    db_path = Path(db_path)
    db_ok = db_path.is_file()
    proc_map = q._match_services()
    running_count = sum(1 for key, _ in SERVICES if proc_map[key])
    expected = len(SERVICES)

    conn = None
    meta_map: dict[str, dict] = {}
    by_status = {s: 0 for s in STATUSES}
    by_worker = {}
    active_jobs = []
    history_jobs = []
    history_total = 0
    source_active = {"cli": 0, "discord": 0}
    guilds = []
    jobs_by_wk = []

    if db_ok:
        try:
            conn = _connect_ro(db_path)
            by_status = _queue_by_status(conn)
            by_worker = _queue_by_worker(conn)
            active_jobs = _active_jobs(conn)
            history_jobs = _history_jobs(conn)
            history_total = _history_count(conn)
            source_active = _active_by_actor(conn)
            guilds = _list_guilds(conn)
            jobs_by_wk = _jobs_total_by_worker_kind(conn)
            meta_map = _load_meta_map(conn)
        finally:
            if conn:
                conn.close()

    services = []
    for key, script in SERVICES:
        procs = proc_map[key]
        proc_running = len(procs) > 0
        meta_key = None
        if key == "orchestrator":
            meta_key = "orchestrator"
        elif key in SERVICE_TO_WORKER_META:
            meta_key = f"worker:{SERVICE_TO_WORKER_META[key]}"
        rss_sum = sum(r for _, r in procs)
        pids = ", ".join(str(p) for p, _ in procs) if procs else "—"
        services.append(
            {
                "key": key,
                "script": script,
                "running": proc_running,
                "pids": pids,
                "rss": rss_sum,
                "db_meta": meta_key in meta_map if meta_key else False,
            }
        )

    cpu_pct, mem_used, mem_total = q._system_stats()
    gpu = q._gpu_stats()
    models = q._collect_model_scores()

    return {
        "generated_at": _utc_now_iso(),
        "db_path": str(db_path.relative_to(ROOT) if db_path.is_relative_to(ROOT) else db_path),
        "db_ok": db_ok,
        "services": services,
        "running_count": running_count,
        "expected_count": expected,
        "by_status": by_status,
        "by_worker": by_worker,
        "active_jobs": active_jobs,
        "history_jobs": history_jobs,
        "history_total": history_total,
        "source_active": source_active,
        "meta_map": meta_map,
        "guilds": guilds,
        "jobs_by_worker_kind": jobs_by_wk,
        "models": models,
        "cpu_pct": cpu_pct,
        "mem_used": mem_used,
        "mem_total": mem_total,
        "gpu": gpu,
        "psutil": _HAS_PSUTIL,
    }


def _fmt_num(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return html.escape(str(v))


def _svc_index(data: dict) -> dict:
    return {s["key"]: s for s in data["services"]}


def _state_label(proc_running: bool, meta_row: dict | None) -> tuple[str, str]:
    if proc_running:
        return "on", "жив"
    if meta_row:
        return "meta", "только meta"
    return "off", "нет"


def _dot(state: str) -> str:
    return f'<span class="dot {state}"></span>'


def api_overview(db_path: Path) -> dict:
    data = _collect_dashboard(db_path)
    return {
        "generated_at": _now_utc().isoformat(),
        "db_ok": data["db_ok"],
        "by_status": data["by_status"],
        "active": sum(data["by_status"].get(s, 0) for s in WAIT_STATUSES),
        "history_total": data["history_total"],
        "source_active": data["source_active"],
        "services_up": data["running_count"],
        "services_expected": data["expected_count"],
        "cpu_pct": data["cpu_pct"],
        "mem_used": data["mem_used"],
        "mem_total": data["mem_total"],
        "gpu": data["gpu"],
    }


def api_stats(db_path: Path) -> dict:
    db_path = Path(db_path)
    if not db_path.is_file():
        return {"by_status": {}, "history_total": 0, "active": 0, "db_ok": False}
    conn = _connect_ro(db_path)
    try:
        by_status = {}
        for row in conn.execute(
            "SELECT status, COUNT(*) AS c FROM jobs GROUP BY status"
        ).fetchall():
            by_status[str(row["status"])] = int(row["c"])
        history_total = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status IN ('done', 'error', 'cancelled')"
        ).fetchone()[0]
        active = conn.execute(
            f"SELECT COUNT(*) FROM jobs WHERE status IN ({','.join('?' * len(WAIT_STATUSES))})",
            WAIT_STATUSES,
        ).fetchone()[0]
    finally:
        conn.close()
    return {
        "by_status": by_status,
        "history_total": int(history_total),
        "active": int(active),
        "db_ok": True,
        "terminal_statuses": list(TERMINAL_STATUSES),
    }


def _range_bounds(range_key: str):
    now = _now_utc()
    mapping = {
        "1h": (timedelta(hours=1), 300),
        "24h": (timedelta(hours=24), 3600),
        "7d": (timedelta(days=7), 21600),
        "30d": (timedelta(days=30), 86400),
    }
    delta, bucket = mapping.get(range_key, mapping["24h"])
    return now - delta, now, bucket


def api_timeseries(db_path: Path, *, metric: str = "jobs_created", range_key: str = "24h") -> dict:
    if metric not in TIMESERIES_METRICS:
        metric = "jobs_created"
    if range_key not in TIMESERIES_RANGES:
        range_key = "24h"
    start, end, bucket = _range_bounds(range_key)
    points = []
    db_path = Path(db_path)
    if not db_path.is_file():
        return {"metric": metric, "range": range_key, "bucket_sec": bucket, "points": points}
    col = "created_at"
    status_filter = None
    if metric == "jobs_done":
        col, status_filter = "finished_at", "done"
    elif metric == "jobs_error":
        col, status_filter = "finished_at", "error"
    elif metric == "jobs_cancelled":
        col, status_filter = "finished_at", "cancelled"
    elif metric == "active":
        col, status_filter = "created_at", None
    conn = _connect_ro(db_path)
    try:
        start_s, end_s = start.isoformat(), end.isoformat()
        if metric == "active":
            rows = conn.execute(
                """
                SELECT created_at, finished_at, status FROM jobs
                WHERE created_at <= ? AND (finished_at IS NULL OR finished_at >= ?)
                """,
                (end_s, start_s),
            ).fetchall()
            t = start
            while t < end:
                t2 = t + timedelta(seconds=bucket)
                c = 0
                for r in rows:
                    c0 = _parse_ts(r["created_at"])
                    f0 = _parse_ts(r["finished_at"])
                    if c0 is None:
                        continue
                    if c0.tzinfo is None:
                        c0 = c0.replace(tzinfo=timezone.utc)
                    if f0 and f0.tzinfo is None:
                        f0 = f0.replace(tzinfo=timezone.utc)
                    if c0 <= t2 and (f0 is None or f0 >= t) and r["status"] in WAIT_STATUSES:
                        c += 1
                points.append({"t": t.isoformat(), "v": c})
                t = t2
        else:
            sql = f"SELECT {col} AS ts FROM jobs WHERE {col} >= ? AND {col} <= ?"
            args: list = [start_s, end_s]
            if status_filter:
                sql += " AND status = ?"
                args.append(status_filter)
            rows = conn.execute(sql, args).fetchall()
            buckets: dict[int, int] = {}
            for r in rows:
                dt = _parse_ts(r["ts"])
                if dt is None:
                    continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                slot = int((dt - start).total_seconds() // bucket)
                buckets[slot] = buckets.get(slot, 0) + 1
            n = max(1, int((end - start).total_seconds() // bucket))
            for i in range(n):
                t = start + timedelta(seconds=i * bucket)
                points.append({"t": t.isoformat(), "v": buckets.get(i, 0)})
    finally:
        conn.close()
    return {"metric": metric, "range": range_key, "bucket_sec": bucket, "points": points}


def api_logs(*, service: str | None = None, level: str | None = None, limit: int = 200) -> dict:
    log_dir = ROOT / "var" / "logs"
    lines = []
    if log_dir.is_dir():
        files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files[:5]:
            try:
                raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in raw[-limit:]:
                entry = {"t": None, "level": "INFO", "service": path.stem, "msg": line}
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        entry = {
                            "t": data.get("ts") or data.get("t"),
                            "level": str(data.get("level") or "INFO"),
                            "service": str(data.get("service") or path.stem),
                            "msg": str(data.get("message") or data.get("msg") or line),
                        }
                except json.JSONDecodeError:
                    pass
                if service and entry["service"] != service:
                    continue
                if level and entry["level"].upper() != level.upper():
                    continue
                lines.append(entry)
    lines = lines[-limit:]
    return {"service": service, "level": level, "lines": lines}
