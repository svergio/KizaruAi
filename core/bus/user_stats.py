import math
from datetime import datetime


def _parse_duration_sec(started_at, finished_at) -> float | None:
    if not started_at or not finished_at:
        return None
    try:
        started = datetime.fromisoformat(str(started_at))
        finished = datetime.fromisoformat(str(finished_at))
        sec = (finished - started).total_seconds()
    except (TypeError, ValueError):
        return None
    if sec < 0:
        return None
    return float(sec)


def _p95_nearest_rank(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    idx = min(n - 1, max(0, math.ceil(0.95 * n) - 1))
    return float(ordered[idx])


def aggregate_user_job_rows(rows: list[dict]) -> dict:
    total = len(rows)
    by_kind: dict[str, int] = {}
    done = error = cancelled = 0
    durations: list[float] = []
    first_created = last_created = None

    for row in rows:
        kind = str(row["kind"])
        by_kind[kind] = by_kind.get(kind, 0) + 1
        status = row["status"]
        if status == "done":
            done += 1
            dur = _parse_duration_sec(row.get("started_at"), row.get("finished_at"))
            if dur is not None:
                durations.append(dur)
        elif status == "error":
            error += 1
        elif status == "cancelled":
            cancelled += 1
        created = row.get("created_at")
        if created:
            if first_created is None or str(created) < str(first_created):
                first_created = created
            if last_created is None or str(created) > str(last_created):
                last_created = created

    avg_duration = None
    if durations:
        avg_duration = float(sum(durations) / len(durations))

    return {
        "total": total,
        "by_kind": by_kind,
        "done": done,
        "error": error,
        "cancelled": cancelled,
        "avg_duration_sec": avg_duration,
        "p95_duration_sec": _p95_nearest_rank(durations),
        "first_created_at": first_created,
        "last_created_at": last_created,
    }


def format_me_embed_lines(stats: dict, display_name: str) -> list[tuple[str, str]]:
    if int(stats.get("total") or 0) == 0:
        return [("Статистика", f"{display_name}, заявок пока нет.")]

    lines = [
        ("Всего заявок", str(stats["total"])),
        ("Готово / ошибка / отмена", f"{stats['done']} / {stats['error']} / {stats['cancelled']}"),
    ]
    if stats.get("by_kind"):
        kinds = ", ".join(f"{k}: {v}" for k, v in sorted(stats["by_kind"].items()))
        lines.append(("По типам", kinds))
    if stats.get("avg_duration_sec") is not None:
        lines.append(("Средняя длительность (done)", f"{stats['avg_duration_sec']:.1f} с"))
    if stats.get("p95_duration_sec") is not None:
        lines.append(("p95 длительность (done)", f"{stats['p95_duration_sec']:.1f} с"))
    if stats.get("first_created_at"):
        lines.append(("Первая заявка", str(stats["first_created_at"])))
    if stats.get("last_created_at"):
        lines.append(("Последняя заявка", str(stats["last_created_at"])))
    return lines
