"""Duration / MSK / ellipsis — same rules as static/js/app.js (KizaruFmt)."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

MSK = ZoneInfo("Europe/Moscow")

_MONTHS_RU = (
    "",
    "янв",
    "фев",
    "мар",
    "апр",
    "май",
    "июн",
    "июл",
    "авг",
    "сен",
    "окт",
    "ноя",
    "дек",
)


def fmt_dur(sec: float | None) -> str:
    if sec is None:
        return "—"
    try:
        s = float(sec)
    except (TypeError, ValueError):
        return "—"
    if s != s:  # NaN
        return "—"
    if s < 0:
        s = 0.0
    if s < 1:
        return f"{int(round(s * 1000))} мс"
    if s < 60:
        t = round(s * 10) / 10
        return f"{t}".replace(".", ",") + " с"
    total = int(round(s))
    h = total // 3600
    m = (total % 3600) // 60
    sec_part = total % 60
    if h > 0:
        return f"{h} ч {m:02d} мин"
    return f"{m} мин {sec_part} с"


def fmt_msk(iso: str | None, *, now: datetime | None = None) -> str:
    if not iso:
        return "—"
    text = str(iso).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return str(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(MSK)
    ref = now if now is not None else datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    today = ref.astimezone(MSK).date()
    if local.date() == today:
        return local.strftime("%H:%M:%S")
    mon = _MONTHS_RU[local.month]
    return f"{local.day} {mon}, {local.strftime('%H:%M:%S')}"


def mask_discord_id(value: str | int | None) -> str:
    if value is None:
        return "—"
    s = str(value)
    if s == "cli" or s.startswith("cli:"):
        return s
    if len(s) <= 8:
        return s
    return f"{s[:4]}…{s[-2:]}"


def trunc_ellipsis(text: str | None, max_len: int) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= max_len:
        return s
    if max_len <= 1:
        return "…"
    return s[: max_len - 1] + "…"
