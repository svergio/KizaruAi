"""Shared constants and helpers for dashboard queries."""
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.web.contracts import STATUS, WORKER_KIND

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DB = ROOT / "data" / "bot.db"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
REFRESH_SEC = 3
NEW_WINDOW_SEC = 15 * 60
HISTORY_LIMIT = 40
ACTIVE_LIMIT = 80
FIELD_HEAD_CHARS = 4000

STATUSES = ("queued", "assigned", "running", "done", "error", "cancelled")
STATUS_LABELS_RU = {k: v["label"] for k, v in STATUS.items()}
KIND_LABELS_RU = {
    "text": "текст (/ai text)",
    "mood": "настроение (/ai mood)",
    "pic": "картинка (/ai pic)",
}
PIC_MODE_RU = {"1": "line", "2": "simple", "3": "render", "4": "details"}
WORKER_KINDS_DB = tuple(WORKER_KIND.keys())

_JSON_STR = re.compile(r'"([A-Za-z0-9_]+)"\s*:\s*"((?:\\.|[^"\\])*)"')
_JSON_NUM = re.compile(r'"([A-Za-z0-9_]+)"\s*:\s*(-?\d+(?:\.\d+)?)')
_JSON_BOOL = re.compile(r'"([A-Za-z0-9_]+)"\s*:\s*(true|false)')


def _now_utc():
    return datetime.now(timezone.utc)


def _utc_now_iso():
    return _now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")


def _connect_ro(db_path: Path):
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _parse_ts(value: str | None):
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _fmt_ts(value: str | None) -> str:
    dt = _parse_ts(value)
    if dt is None:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%d.%m %H:%M:%S")


def _iso_utc(value: str | None) -> str | None:
    if not value:
        return None
    dt = _parse_ts(value)
    if dt is None:
        return str(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _display_name(row) -> str | None:
    name = row["display_name"] if "display_name" in row.keys() else None
    if name:
        return str(name)
    return None


def _who_parts(actor: str | None) -> tuple[str, str]:
    if not actor:
        return "—", ""
    text = str(actor)
    if text == "cli":
        return "cli", text
    return "discord", text


def _truncate_text(text, max_len: int):
    if text is None:
        return ""
    s = str(text)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _one_line(text: str | None, limit: int = 110) -> str:
    if not text:
        return "—"
    flat = " ".join(str(text).split())
    return _truncate_text(flat, limit) or "—"


def _freshness(finished_at: str | None, created_at: str | None) -> str:
    dt = _parse_ts(finished_at) or _parse_ts(created_at)
    if dt is None:
        return "старое"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age = (_now_utc() - dt).total_seconds()
    if age <= NEW_WINDOW_SEC:
        return "новое"
    return "старое"


def _loose_fields(raw: str) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    out: dict = {}
    for match in _JSON_STR.finditer(raw):
        key = match.group(1)
        if key == "data_b64":
            continue
        val = match.group(2)
        try:
            out[key] = json.loads(f'"{val}"')
        except json.JSONDecodeError:
            out[key] = val
    for match in _JSON_NUM.finditer(raw):
        if match.group(1) not in out:
            out[match.group(1)] = match.group(2)
    for match in _JSON_BOOL.finditer(raw):
        if match.group(1) not in out:
            out[match.group(1)] = match.group(2) == "true"
    return out


def _human_message_from_payload(kind: str, raw: str | None) -> str:
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _truncate_text(raw, 500)
    if not isinstance(data, dict):
        return _truncate_text(str(raw), 500)
    if kind == "text":
        model = data.get("model", "lstm")
        start = data.get("start", "Я")
        return f"модель {model}, начало: {start}"
    if kind == "mood":
        source = data.get("source") or data.get("text") or ""
        return str(source).strip()
    if kind == "pic":
        mode_key = str(data.get("mode_key", "?"))
        mode = PIC_MODE_RU.get(mode_key, mode_key)
        fname = data.get("filename") or "вложение"
        bits = [f"файл: {fname}", f"режим: {mode}"]
        if data.get("as_file") or mode_key == "3":
            bits.append("файл")
            if "scale" in data:
                bits.append(f"scale={data['scale']}")
            if "cells" in data:
                bits.append(f"cells={data['cells']}")
            if "animated" in data:
                bits.append("анимация" if data["animated"] else "статично")
        return ", ".join(bits)
    compact = json.dumps(
        {k: v for k, v in data.items() if k != "data_b64"},
        ensure_ascii=False,
    )
    return _truncate_text(compact, 500)
