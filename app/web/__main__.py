import argparse
import html
import json
import re
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB = ROOT / "data" / "bot.db"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
REFRESH_SEC = 3
NEW_WINDOW_SEC = 15 * 60
HISTORY_LIMIT = 40
ACTIVE_LIMIT = 80
# Image jobs store base64 in payload/result; the panel reads only a prefix.
FIELD_HEAD_CHARS = 4000

STATUSES = ("queued", "assigned", "running", "done", "error")
STATUS_LABELS_RU = {
    "queued": "в очереди",
    "assigned": "назначена",
    "running": "выполняется",
    "done": "готово",
    "error": "ошибка",
}
KIND_LABELS_RU = {
    "text": "текст (/ai text)",
    "mood": "настроение (/ai mood)",
    "pic": "картинка (/ai pic)",
}
PIC_MODE_RU = {"1": "line", "2": "simple", "3": "cli"}
WORKER_KINDS_DB = (
    "text_lstm",
    "text_lstm_word",
    "text_transformer",
    "text_gpt",
    "text_deepseek",
    "mood",
    "pic_still",
    "pic_video",
)

SERVICES = (
    ("discord", "app.discord"),
    ("orchestrator", "app.orchestrator"),
    ("worker_lstm", "app.workers.lstm"),
    ("worker_lstm_word", "app.workers.lstm_word"),
    ("worker_transformer", "app.workers.transformer"),
    ("worker_gpt", "app.workers.gpt"),
    ("worker_deepseek", "app.workers.deepseek"),
    ("worker_mood", "app.workers.mood"),
    ("worker_still", "app.workers.still"),
    ("worker_video", "app.workers.video"),
    ("web", "app.web"),
)

SERVICE_TO_WORKER_META = {
    "worker_lstm": "text_lstm",
    "worker_lstm_word": "text_lstm_word",
    "worker_transformer": "text_transformer",
    "worker_gpt": "text_gpt",
    "worker_deepseek": "text_deepseek",
    "worker_mood": "mood",
    "worker_still": "pic_still",
    "worker_video": "pic_video",
}

WORKER_META_TO_SERVICE = {v: k for k, v in SERVICE_TO_WORKER_META.items()}

WORKER_ACCEPTS = {
    "text_lstm": "kind=text, model=lstm (char)",
    "text_lstm_word": "kind=text, model=lstm-word",
    "text_transformer": "kind=text, model=transformer",
    "text_gpt": "kind=text, model=gpt",
    "text_deepseek": "kind=text, model=deepseek",
    "mood": "kind=mood",
    "pic_still": "kind=pic, mode 1/2 или статичный 3",
    "pic_video": "kind=pic, mode 3 animated",
}

_JSON_STR = re.compile(r'"([A-Za-z0-9_]+)"\s*:\s*"((?:\\.|[^"\\])*)"')
_JSON_NUM = re.compile(r'"([A-Za-z0-9_]+)"\s*:\s*(-?\d+(?:\.\d+)?)')
_JSON_BOOL = re.compile(r'"([A-Za-z0-9_]+)"\s*:\s*(true|false)')

CHECKPOINTS = (
    ("LSTM", ROOT / "checkpoints" / "kizaru.pt"),
    ("Transformer", ROOT / "checkpoints" / "transformer.pt"),
    ("GPT", ROOT / "checkpoints" / "gpt.pt"),
    ("DeepSeek", ROOT / "checkpoints" / "deepseek.pt"),
    ("Mood", ROOT / "checkpoints" / "mood.pt"),
)

_meta_cache: dict[str, tuple[float, dict | None]] = {}
_meta_lock = threading.Lock()
META_TTL_SEC = 120.0

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    psutil = None
    _HAS_PSUTIL = False


def _utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


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


def _load_meta_map(conn) -> dict[str, dict]:
    if not _table_exists(conn, "worker_meta"):
        return {}
    rows = conn.execute(
        "SELECT key, value, updated_at FROM worker_meta"
    ).fetchall()
    out: dict[str, dict] = {}
    for row in rows:
        raw = row["value"] or ""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            parsed = {}
        out[row["key"]] = {
            "pid": parsed.get("pid"),
            "updated_at": row["updated_at"],
        }
    return out


def _iter_python_processes():
    if _HAS_PSUTIL:
        for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if "python" not in name:
                    continue
                cmdline = proc.info.get("cmdline") or []
                if not cmdline:
                    continue
                joined = " ".join(str(x) for x in cmdline).lower()
                if "python" not in joined and name not in ("python.exe", "python3.exe"):
                    continue
                mem = proc.info.get("memory_info")
                rss = int(mem.rss) if mem else 0
                yield int(proc.info["pid"]), joined, rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return

    if sys.platform == "win32":
        ps = (
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
            "| Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress"
        )
        try:
            raw = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
            text = (raw.stdout or "").strip()
            if not text:
                return
            data = json.loads(text)
            if isinstance(data, dict):
                data = [data]
            for item in data:
                pid = int(item.get("ProcessId") or 0)
                cmd = (item.get("CommandLine") or "").lower()
                if pid and cmd:
                    yield pid, cmd, 0
        except (subprocess.SubprocessError, json.JSONDecodeError, ValueError, OSError):
            return


def _service_cmdline_needles(key: str, script: str) -> tuple[str, ...]:
    return (script.lower().replace("\\", "/"),)


def _cmdline_has_module(cmd: str, module: str) -> bool:
    mod = module.lower()
    start = 0
    while True:
        i = cmd.find(mod, start)
        if i < 0:
            return False
        end = i + len(mod)
        before_ok = i == 0 or cmd[i - 1] not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        after_ok = end >= len(cmd) or cmd[end] not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        if before_ok and after_ok:
            return True
        start = i + 1


def _match_services():
    matches: dict[str, list[tuple[int, int]]] = {key: [] for key, _ in SERVICES}
    for pid, cmdline, rss in _iter_python_processes():
        cmd = cmdline.replace("\\", "/")
        for key, script in SERVICES:
            if any(_cmdline_has_module(cmd, n) for n in _service_cmdline_needles(key, script)):
                matches[key].append((pid, rss))
    return matches


def _format_bytes(n: int) -> str:
    if n <= 0:
        return "—"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    v = float(n)
    u = 0
    while v >= 1024 and u < len(units) - 1:
        v /= 1024
        u += 1
    if u == 0:
        return f"{int(v)} {units[u]}"
    return f"{v:.1f} {units[u]}"


def _system_stats():
    cpu_pct = None
    mem_used = None
    mem_total = None
    if _HAS_PSUTIL:
        cpu_pct = psutil.cpu_percent(interval=0.05)
        vm = psutil.virtual_memory()
        mem_used = int(vm.used)
        mem_total = int(vm.total)
    elif sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                mem_total = int(stat.ullTotalPhys)
                mem_used = int(stat.ullTotalPhys - stat.ullAvailPhys)
        except OSError:
            pass
    return cpu_pct, mem_used, mem_total


def _gpu_stats():
    exe = "nvidia-smi"
    try:
        raw = subprocess.run(
            [
                exe,
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if raw.returncode != 0:
        return None
    lines = [ln.strip() for ln in (raw.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None
    out = []
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            out.append(
                {
                    "index": i,
                    "util_pct": parts[0],
                    "mem_used_mib": parts[1],
                    "mem_total_mib": parts[2],
                }
            )
    return out or None


def _read_sidecar_meta(path: Path) -> dict | None:
    side = path.with_name(path.name + ".meta.json")
    if not side.is_file():
        side = path.with_suffix(path.suffix + ".meta.json")
    if not side.is_file():
        return None
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _read_pt_meta(path: Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    side = _read_sidecar_meta(path)
    if side:
        return side
    cache_key = str(path.resolve())
    now = time.monotonic()
    with _meta_lock:
        hit = _meta_cache.get(cache_key)
        if hit and now - hit[0] < META_TTL_SEC:
            return hit[1]
    meta = None
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            keep = (
                "arch",
                "epoch",
                "mean_loss",
                "best_loss",
                "train_axis_acc",
                "train_phrase_acc",
            )
            meta = {k: payload[k] for k in keep if k in payload}
            if meta:
                for k, v in list(meta.items()):
                    if hasattr(v, "item"):
                        try:
                            meta[k] = float(v.item())
                        except (TypeError, ValueError):
                            meta[k] = str(v)
    except Exception:
        meta = None
    with _meta_lock:
        _meta_cache[cache_key] = (now, meta)
    return meta


def _collect_model_scores():
    rows = []
    for label, path in CHECKPOINTS:
        meta = _read_pt_meta(path)
        if meta is None and not path.is_file():
            rows.append(
                {
                    "label": label,
                    "path": str(path.relative_to(ROOT)),
                    "present": False,
                    "mean_loss": None,
                    "train_axis_acc": None,
                    "extra": "",
                }
            )
            continue
        mean_loss = meta.get("mean_loss") if meta else None
        if mean_loss is None and meta:
            mean_loss = meta.get("best_loss")
        axis_acc = meta.get("train_axis_acc") if meta else None
        extra_bits = []
        if meta:
            if "epoch" in meta:
                extra_bits.append(f"epoch={meta['epoch']}")
            if "arch" in meta:
                extra_bits.append(f"arch={meta['arch']}")
            if meta.get("train_phrase_acc") is not None:
                extra_bits.append(f"phrase_acc={meta['train_phrase_acc']}")
        rows.append(
            {
                "label": label,
                "path": str(path.relative_to(ROOT)),
                "present": path.is_file(),
                "mean_loss": mean_loss,
                "train_axis_acc": axis_acc,
                "extra": ", ".join(extra_bits),
            }
        )
    return rows


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


def _freshness(finished_at: str | None, created_at: str | None) -> str:
    dt = _parse_ts(finished_at) or _parse_ts(created_at)
    if dt is None:
        return "старое"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - dt).total_seconds()
    if age <= NEW_WINDOW_SEC:
        return "новое"
    return "старое"


def _one_line(text: str | None, limit: int = 110) -> str:
    if not text:
        return "—"
    flat = " ".join(str(text).split())
    return _truncate_text(flat, limit) or "—"


def _who_parts(actor: str | None) -> tuple[str, str]:
    if not actor:
        return "—", ""
    text = str(actor)
    if text.startswith("cli:"):
        return "cli", text
    return "discord", text


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
          CASE WHEN u.discord_id LIKE 'cli:%' THEN 'cli' ELSE 'discord' END AS src,
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


def _truncate_text(text, max_len: int):
    if text is None:
        return ""
    s = str(text)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


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
        if mode_key == "3":
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


def _svg_hbars(
    title: str,
    items: list[tuple[str, float]],
    *,
    unit: str = "",
    bar_color: str = "#3a6ea5",
    label_w: int = 130,
    chart_w: int = 260,
) -> str:
    if not items:
        return '<p class="muted">Нет данных.</p>'
    max_val = max(v for _, v in items) or 1.0
    row_h = 24
    top = 22
    height = top + len(items) * row_h + 4
    width = label_w + chart_w + 56
    parts = [
        f'<svg class="chart" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img">',
        f'<text x="0" y="14" font-size="13" font-weight="600" fill="#111">'
        f"{html.escape(title)}</text>",
    ]
    y = top
    for label, val in items:
        w = max(0, int((float(val) / max_val) * chart_w))
        if val > 0 and w < 2:
            w = 2
        val_s = str(int(val)) if float(val) == int(val) else f"{float(val):.1f}"
        parts.append(
            f'<text x="0" y="{y + 15}" font-size="11" fill="#333">'
            f"{html.escape(label)}</text>"
        )
        parts.append(
            f'<rect x="{label_w}" y="{y}" width="{w}" height="17" '
            f'fill="{bar_color}" rx="2"/>'
        )
        parts.append(
            f'<text x="{label_w + w + 6}" y="{y + 14}" font-size="11" fill="#111">'
            f"{html.escape(val_s)}{html.escape(unit)}</text>"
        )
        y += row_h
    parts.append("</svg>")
    return "".join(parts)


def _svg_pct_bars(title: str, items: list[tuple[str, float | None]]) -> str:
    rows = [(lbl, v if v is not None else 0.0) for lbl, v in items if v is not None]
    if not rows:
        return f'<p class="muted">{html.escape(title)}: н/д</p>'
    return _svg_hbars(title, rows, unit="%", bar_color="#2d7a4f", label_w=72, chart_w=320)


def _collect_dashboard(db_path: Path):
    db_path = Path(db_path)
    db_ok = db_path.is_file()
    proc_map = _match_services()
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

    cpu_pct, mem_used, mem_total = _system_stats()
    gpu = _gpu_stats()
    models = _collect_model_scores()

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


def _meta_line(proc_running: bool, pids: str, meta_row: dict | None, *, mention_meta: bool = True) -> str:
    bits = []
    if proc_running:
        bits.append(f"процесс PID {pids}")
    else:
        bits.append("процесса нет")
    if not mention_meta:
        return ", ".join(bits)
    if meta_row:
        pid = meta_row.get("pid")
        bits.append(f"meta pid {pid}" if pid else "meta есть")
        stamp = _fmt_ts(meta_row.get("updated_at"))
        if stamp != "—":
            bits.append(f"регистрация {stamp}")
    else:
        bits.append("meta нет")
    return ", ".join(bits)


def _dot(state: str) -> str:
    return f'<span class="dot {state}"></span>'


def _render_scheme(data: dict) -> str:
    svc = _svc_index(data)
    meta = data.get("meta_map") or {}
    by_status = data["by_status"]
    by_worker = data.get("by_worker") or {}
    source = data.get("source_active") or {"cli": 0, "discord": 0}
    queued = int(by_status.get("queued", 0))
    assigned = int(by_status.get("assigned", 0))
    running = int(by_status.get("running", 0))

    bot = svc.get("discord") or {}
    orch = svc.get("orchestrator") or {}
    orch_meta = meta.get("orchestrator")
    bot_state, bot_label = _state_label(bool(bot.get("running")), None)
    orch_state, orch_label = _state_label(bool(orch.get("running")), orch_meta)

    workers_html = []
    for wk in WORKER_KINDS_DB:
        key = WORKER_META_TO_SERVICE[wk]
        proc = svc.get(key) or {}
        meta_row = meta.get(f"worker:{wk}")
        state, label = _state_label(bool(proc.get("running")), meta_row)
        counts = by_worker.get(wk) or {}
        assigned_n = int(counts.get("assigned", 0))
        running_n = int(counts.get("running", 0))
        script = proc.get("script") or ""
        workers_html.append(
            '<div class="node">'
            f"{_dot(state)}<strong>{html.escape(label)}</strong> "
            f'<span class="muted">{html.escape(wk)}</span>'
            f"<div><code>{html.escape(script)}</code></div>"
            f"<div>принимает {html.escape(WORKER_ACCEPTS[wk])}</div>"
            f'<div class="count-num">assigned {assigned_n} · running {running_n}</div>'
            f'<div class="muted">{html.escape(_meta_line(bool(proc.get("running")), proc.get("pids") or "—", meta_row))}</div>'
            "</div>"
        )

    flow = (
        '<div class="scheme-flow">'
        '<div class="hop"><h3>Клиенты</h3>'
        '<div class="node">'
        f"{_dot(bot_state)}<strong>{html.escape(bot_label)}</strong>"
        "<div>Discord bot</div>"
        "<div><code>app.discord</code> — только ввод-вывод</div>"
        f'<div class="count-num">в работе: {int(source.get("discord", 0))}</div>'
        f'<div class="muted">{html.escape(_meta_line(bool(bot.get("running")), bot.get("pids") or "—", None, mention_meta=False))}</div>'
        "</div>"
        '<div class="node">'
        "<strong>по запросу</strong>"
        "<div>CLI</div>"
        "<div><code>app.cli.generate</code> / <code>app.cli.image</code></div>"
        "<div>автор <code>cli:&lt;pid&gt;</code></div>"
        f'<div class="count-num">в работе: {int(source.get("cli", 0))}</div>'
        "</div></div>"
        '<div class="arrow"><div>далее</div><small>kind + payload, автор discord или cli</small></div>'
        '<div class="hop"><h3>Шина</h3>'
        '<div class="node">'
        f'<div class="big-count">{queued}</div>'
        "<div>queued</div>"
        "<div><code>data/bot.db</code>, таблица jobs</div>"
        '<div class="muted">worker_kind ещё пуст, ждёт оркестратор</div>'
        "</div></div>"
        '<div class="arrow"><div>далее</div><small>kind переходит в worker_kind, status=assigned</small></div>'
        '<div class="hop"><h3>Оркестратор</h3>'
        '<div class="node">'
        f"{_dot(orch_state)}<strong>{html.escape(orch_label)}</strong>"
        "<div><code>app.orchestrator</code></div>"
        "<div>маршрутизация, без генерации</div>"
        f'<div class="big-count">{assigned}</div>'
        "<div>assigned</div>"
        f'<div class="muted">{html.escape(_meta_line(bool(orch.get("running")), orch.get("pids") or "—", orch_meta))}</div>'
        "</div></div>"
        '<div class="arrow"><div>далее</div><small>worker_kind, assigned затем running</small></div>'
        '<div class="hop workers"><h3>Воркеры</h3>'
        f'<p class="muted">Один процесс на worker_kind. Забирает assigned, пишет result, статус done или error. Сейчас running: {running}.</p>'
        f'<div class="worker-grid">{"".join(workers_html)}</div>'
        "</div></div>"
    )
    table = (
        "<table><thead><tr>"
        "<th>Шаг</th><th>Куда в jobs</th><th>Что передаётся</th><th>Сейчас</th>"
        "</tr></thead><tbody>"
        f"<tr><td>Клиенты</td><td>status=queued</td>"
        "<td>kind (text, mood, pic) и payload. Автор: discord id или cli:pid. worker_kind пуст.</td>"
        f"<td>{queued}</td></tr>"
        "<tr><td>Оркестратор</td><td>status=assigned</td>"
        "<td>worker_kind: text_lstm, text_lstm_word, text_transformer, text_gpt, text_deepseek, mood, pic_still, pic_video.</td>"
        f"<td>{assigned}</td></tr>"
        "<tr><td>Воркер</td><td>status=running, затем done или error</td>"
        "<td>result (текст или файл) либо error.</td>"
        f"<td>{running}</td></tr>"
        "</tbody></table>"
    )
    return flow + table


def _who_cell(job: dict) -> str:
    kind = html.escape(job["who_kind"])
    ident = html.escape(job["who_id"]) if job["who_id"] else ""
    if ident:
        return f'<span class="who"><strong>{kind}</strong> <span class="muted">{ident}</span></span>'
    return f"<strong>{kind}</strong>"


def _render_job_body(job: dict) -> str:
    blocks = []
    if job["prompt"]:
        blocks.append(
            '<div class="detail"><div class="muted">Запрос</div>'
            f'<pre>{html.escape(job["prompt"])}</pre></div>'
        )
    if job["result"]:
        blocks.append(
            '<div class="detail"><div class="muted">Результат</div>'
            f'<pre>{html.escape(job["result"])}</pre></div>'
        )
    if job["error"]:
        blocks.append(
            '<div class="detail"><div class="muted">Ошибка</div>'
            f'<pre class="bad">{html.escape(job["error"])}</pre></div>'
        )
    if not blocks:
        blocks.append('<p class="muted">Нет текста запроса и результата.</p>')
    who = html.escape(job["who_id"] or job["who_kind"])
    times = (
        f"создана {_fmt_ts(job['created_at'])}"
        f" · старт {_fmt_ts(job['started_at'])}"
        f" · конец {_fmt_ts(job['finished_at'])}"
        f" · status={html.escape(job['status'])}"
    )
    blocks.append(f'<p class="muted">Кто: {who}. {html.escape(times)} UTC.</p>')
    return "".join(blocks)


def _job_head(mode: str) -> str:
    cols = "cols-now" if mode == "now" else "cols-hist"
    labels = ["", "Время", "id", "Кто", "kind", "worker_kind", "Статус"]
    if mode == "history":
        labels.append("Метка")
    labels.append("Превью")
    cells = "".join(
        f"<span>{html.escape(label)}</span>" if label else "<span></span>"
        for label in labels
    )
    return f'<div class="job-head {cols}">{cells}</div>'


def _render_job_row(job: dict, mode: str) -> str:
    cols = "cols-now" if mode == "now" else "cols-hist"
    when = job["created_at"] if mode == "now" else (job["finished_at"] or job["created_at"])
    wk = job["worker_kind"] or "—"
    if mode == "now":
        status_cell = f"в работе: {job['status_ru']}"
    else:
        status_cell = job["status_ru"]
    bits = [
        f"<span>{html.escape(_fmt_ts(when))}</span>",
        f"<span>{int(job['id'])}</span>",
        f"<span>{_who_cell(job)}</span>",
        f"<span>{html.escape(job['kind'] or '—')}</span>",
        f"<span>{html.escape(wk)}</span>",
        f"<span>{html.escape(status_cell)}</span>",
    ]
    if mode == "history":
        age = job["age"] or "старое"
        css = "badge-new" if age == "новое" else "badge-old"
        bits.append(f'<span class="badge {css}">{html.escape(age)}</span>')
    bits.append(f'<span class="preview">{html.escape(job["preview"])}</span>')
    return (
        f'<details class="job-row" data-job-id="{int(job["id"])}">'
        f'<summary class="{cols}">{"".join(bits)}</summary>'
        f'<div class="job-body">{_render_job_body(job)}</div>'
        "</details>"
    )


def _render_now_section(data: dict) -> str:
    jobs = data.get("active_jobs") or []
    inflight = sum(int(data["by_status"].get(s, 0)) for s in ("queued", "assigned", "running"))
    if jobs:
        note = ""
        if inflight > len(jobs):
            note = f'<p class="muted">Показаны {len(jobs)} из {inflight}.</p>'
        rows = "".join(_render_job_row(job, "now") for job in jobs)
        body = note + f'<div class="job-list">{_job_head("now")}{rows}</div>'
    else:
        body = '<p class="muted">Сейчас заявок нет.</p>'
    return (
        '<section class="zone zone-now" data-live="now">'
        f'<h2>Сейчас <span class="pill">в работе {inflight}</span></h2>'
        '<p class="muted">Статусы queued, assigned и running. Новые сверху. '
        "Клик по строке открывает заявку. Это не история.</p>"
        f"{body}</section>"
    )


def _render_history_section(data: dict) -> str:
    jobs = data.get("history_jobs") or []
    total = int(data.get("history_total") or 0)
    new_n = sum(1 for job in jobs if job.get("age") == "новое")
    if jobs:
        shown = f"показаны последние {len(jobs)}"
        if total > len(jobs):
            shown += f" из {total}"
        rows = "".join(_render_job_row(job, "history") for job in jobs)
        body = (
            f'<p class="muted">{html.escape(shown)}. Среди показанных новых: {new_n}.</p>'
            f'<div class="job-list">{_job_head("history")}{rows}</div>'
        )
    else:
        body = '<p class="muted">История пуста.</p>'
    return (
        '<section class="zone zone-hist" data-live="history">'
        f'<h2>История <span class="pill">done/error {total}</span></h2>'
        '<p class="muted">Короткая строка: время, id, кто, kind, worker_kind, статус, превью. '
        "«новое» — завершено за последние 15 минут, «старое» — раньше. Клик открывает детали.</p>"
        f"{body}</section>"
    )


def _render_services_section(data: dict) -> str:
    rc = data["running_count"]
    ec = data["expected_count"]
    rows = []
    for s in data["services"]:
        st = '<span class="ok">да</span>' if s["running"] else '<span class="bad">нет</span>'
        meta = "да" if s["db_meta"] else "—"
        rss = _format_bytes(s["rss"]) if s["rss"] else ("—" if not s["running"] else "н/д")
        rows.append(
            "<tr>"
            f"<td>{html.escape(s['key'])}</td>"
            f"<td><code>{html.escape(s['script'])}</code></td>"
            f"<td>{st}</td>"
            f"<td>{html.escape(s['pids'])}</td>"
            f"<td>{html.escape(rss)}</td>"
            f"<td>{meta}</td>"
            "</tr>"
        )
    body = "".join(rows)
    return (
        '<section data-live="services">'
        "<h2>Серверы (процессы на этой машине)</h2>"
        f"<p>Запущено <strong>{rc}</strong> из <strong>{ec}</strong> ожидаемых сервисов.</p>"
        "<table><thead><tr>"
        "<th>Сервис</th><th>Скрипт</th><th>Процесс</th><th>PID</th><th>RSS</th><th>Запись в worker_meta</th>"
        "</tr></thead><tbody>"
        f"{body}</tbody></table></section>"
    )


def _render_guilds_section(data: dict) -> str:
    if not data["guilds"]:
        body = (
            '<p class="muted">В БД пока нет записей. После on_ready app.discord сохраняет список '
            "серверов в таблицу guilds (нужен запущенный бот).</p>"
        )
    else:
        rows = []
        for g in data["guilds"]:
            mc = g["member_count"]
            mc_s = str(mc) if mc is not None else "—"
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(g['discord_id']))}</td>"
                f"<td>{html.escape(str(g['name']))}</td>"
                f"<td>{html.escape(mc_s)}</td>"
                f"<td>{html.escape(str(g['updated_at'] or ''))}</td>"
                "</tr>"
            )
        body = (
            "<table><thead><tr>"
            "<th>ID</th><th>Название</th><th>Участников</th><th>Обновлено</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
    return (
        '<section data-live="guilds">'
        "<h2>Серверы Discord (гильдии бота)</h2>"
        f"{body}</section>"
    )


def _render_resources_section(data: dict) -> str:
    cpu = data["cpu_pct"]
    cpu_s = f"{cpu:.1f}%" if cpu is not None else "н/д"
    mem_u = data["mem_used"]
    mem_t = data["mem_total"]
    if mem_u is not None and mem_t is not None:
        mem_s = f"{_format_bytes(mem_u)} / {_format_bytes(mem_t)}"
    else:
        mem_s = "н/д"
    ps_note = "psutil" if data["psutil"] else "без psutil"
    items = []
    for s in data["services"]:
        if not s["running"]:
            continue
        rss = _format_bytes(s["rss"]) if s["rss"] else "н/д"
        items.append(
            f"<li>{html.escape(s['key'])}: {html.escape(rss)} (PID {html.escape(s['pids'])})</li>"
        )
    if data["gpu"]:
        bits = []
        for g in data["gpu"]:
            bits.append(
                f"#{g['index']}: загрузка {g['util_pct']}%, "
                f"VRAM {g['mem_used_mib']}/{g['mem_total_mib']} MiB"
            )
        gpu = html.escape("; ".join(bits))
    else:
        gpu = "<strong>нет</strong>"
    lis = "".join(items) if items else "<li>нет запущенных сервисов из списка</li>"
    return (
        '<section data-live="resources">'
        "<h2>Ресурсы сервера</h2>"
        f"<p>CPU: <strong>{html.escape(cpu_s)}</strong> ({ps_note}). "
        f"RAM: <strong>{html.escape(mem_s)}</strong>.</p>"
        f"<p>RSS по сервисам (если процесс найден):</p><ul>{lis}</ul>"
        f"<p>GPU (nvidia-smi): {gpu}</p>"
        "</section>"
    )


PAGE_CSS = """
body{font-family:system-ui,sans-serif;margin:1rem 1.5rem;background:#f6f6f8;color:#111}
h1,h2{margin:0.6rem 0 0.4rem}
table{border-collapse:collapse;width:100%;margin-bottom:1rem;background:#fff}
th,td{border:1px solid #ccc;padding:0.35rem 0.5rem;text-align:left;font-size:0.92rem;vertical-align:top}
th{background:#e8e8ec}
.ok{color:#0a6b0a;font-weight:600}
.bad{color:#a00;font-weight:600}
.muted{color:#555;font-size:0.85rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}
.charts{display:flex;flex-wrap:wrap;gap:1.5rem;align-items:flex-start;margin:0.5rem 0 1rem}
.chart{background:#fff;border:1px solid #ccc;padding:0.5rem;border-radius:4px}
.zone{background:#fff;border:1px solid #ccc;border-radius:6px;padding:0.6rem 0.8rem;margin:0.8rem 0}
.zone h2{margin-top:0}
.zone-now{border-top:4px solid #b45c3a}
.zone-hist{border-top:4px solid #3a6ea5}
.pill{display:inline-block;background:#e8e8ec;border-radius:999px;padding:0.05rem 0.55rem;font-size:0.78rem;font-weight:600;vertical-align:middle}
.scheme-flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:0.45rem;margin:0.4rem 0 0.8rem}
.hop{flex:1 1 14rem;background:#fff;border:1px solid #ccc;border-radius:6px;padding:0.55rem 0.65rem}
.hop h3{margin:0 0 0.35rem;font-size:1rem}
.hop.workers{flex:2 1 28rem}
.arrow{flex:0 0 7.2rem;align-self:center;text-align:center;font-size:0.78rem;color:#222}
.arrow small{display:block;color:#444}
.node{border:1px solid #ddd;border-radius:4px;padding:0.35rem 0.45rem;margin:0.3rem 0;background:#fafafa;font-size:0.86rem}
.big-count{font-size:1.5rem;font-weight:700;line-height:1.1}
.count-num{font-weight:700}
.worker-grid{display:grid;grid-template-columns:1fr 1fr;gap:0.35rem}
.dot{display:inline-block;width:0.55rem;height:0.55rem;border-radius:50%;margin-right:0.25rem;vertical-align:middle;background:#888}
.dot.on{background:#0a6b0a}
.dot.off{background:#a00}
.dot.meta{background:#a66b00}
.job-list{overflow-x:auto;border:1px solid #ddd;background:#fff}
.job-head,.job-row>summary{display:grid;gap:0.35rem 0.45rem;padding:0.32rem 0.5rem;align-items:baseline;min-width:68rem}
.cols-now{grid-template-columns:1.1rem 8rem 3.2rem minmax(9rem,14rem) 4.6rem 9.2rem 11rem minmax(8rem,1fr)}
.cols-hist{grid-template-columns:1.1rem 8rem 3.2rem minmax(9rem,14rem) 4.6rem 9.2rem 6.4rem 4.8rem minmax(8rem,1fr)}
.job-head{background:#e8e8ec;font-size:0.8rem;font-weight:600}
.job-row{border-top:1px solid #ddd}
.job-row>summary{cursor:pointer;list-style:none}
.job-row>summary::-webkit-details-marker{display:none}
.job-row>summary::before{content:"+";font-weight:700;color:#036}
.job-row[open]>summary::before{content:"-"}
.job-row>summary:hover{background:#f3f6fb}
.job-row[open]>summary{background:#eef3f8}
.job-body{padding:0.35rem 0.7rem 0.65rem}
.job-body pre{white-space:pre-wrap;word-break:break-word;margin:0.15rem 0 0.55rem;font-size:0.88rem}
.preview{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.badge-new{color:#083e78;font-weight:700}
.badge-old{color:#555;font-weight:600}
"""

PAGE_JS = """
(function () {
  var KEY = "kizaruOpenJobs";
  function readStored() {
    try { return JSON.parse(sessionStorage.getItem(KEY) || "[]"); }
    catch (e) { return []; }
  }
  function writeStored(ids) {
    try { sessionStorage.setItem(KEY, JSON.stringify(ids)); }
    catch (e) {}
  }
  function collect() {
    return Array.prototype.map.call(
      document.querySelectorAll("details[data-job-id][open]"),
      function (el) { return el.getAttribute("data-job-id"); }
    );
  }
  function restore(ids) {
    ids.forEach(function (id) {
      if (!/^[0-9]+$/.test(String(id))) return;
      var el = document.querySelector('details[data-job-id="' + id + '"]');
      if (el) el.open = true;
    });
  }
  restore(readStored());
  document.addEventListener("toggle", function (ev) {
    var t = ev.target;
    if (t && t.matches && t.matches("details[data-job-id]")) writeStored(collect());
  }, true);
  function applyHtml(html) {
    var ids = collect();
    var doc = new DOMParser().parseFromString(html, "text/html");
    var nodes = document.querySelectorAll("[data-live]");
    for (var i = 0; i < nodes.length; i++) {
      var key = nodes[i].getAttribute("data-live");
      var next = doc.querySelector('[data-live="' + key + '"]');
      if (next && nodes[i].parentNode) nodes[i].replaceWith(next);
    }
    restore(ids);
    writeStored(collect());
    var state = document.getElementById("refresh-state");
    if (state) state.textContent = "";
  }
  function tick() {
    fetch(location.pathname, { cache: "no-store" })
      .then(function (r) {
        if (!r.ok) throw new Error(String(r.status));
        return r.text();
      })
      .then(applyHtml)
      .catch(function () {
        var state = document.getElementById("refresh-state");
        if (state) state.textContent = "Не удалось обновить, показаны прошлые данные.";
      });
  }
  setInterval(tick, %d);
})();
""" % (REFRESH_SEC * 1000)


def _render_dashboard(data: dict) -> bytes:
    title = "KizaruAi — панель администратора"
    db_note = " — файл не найден" if not data["db_ok"] else ""
    parts = [
        "<!DOCTYPE html>",
        '<html lang="ru">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        f"<style>{PAGE_CSS}</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="muted" data-live="stamp">Обновлено: {html.escape(data["generated_at"])}. '
        f"Блоки обновляются каждые {REFRESH_SEC} с, открытая заявка остаётся открытой. "
        f'БД: <code>{html.escape(data["db_path"])}</code>{html.escape(db_note)}.</p>',
        '<p id="refresh-state" class="bad"></p>',
        '<section data-live="scheme">',
        "<h2>Как идут заявки</h2>",
        '<p class="muted">Клиенты кладут kind в SQLite. Оркестратор ставит worker_kind и status=assigned. '
        "Воркер переводит assigned в running и пишет done или error.</p>",
        _render_scheme(data),
        "</section>",
        _render_now_section(data),
        _render_history_section(data),
    ]

    parts.append('<section data-live="queue"><h2>Очередь задач</h2>')
    status_chart = [
        (STATUS_LABELS_RU.get(st, st), float(data["by_status"].get(st, 0)))
        for st in STATUSES
    ]
    wk_chart = [(wk, float(c)) for wk, c in data.get("jobs_by_worker_kind", [])]
    cpu = data["cpu_pct"]
    mem_u = data["mem_used"]
    mem_t = data["mem_total"]
    mem_pct = None
    if mem_u is not None and mem_t and mem_t > 0:
        mem_pct = 100.0 * float(mem_u) / float(mem_t)
    res_items: list[tuple[str, float | None]] = [
        ("CPU", cpu),
        ("RAM", mem_pct),
    ]
    if data["gpu"]:
        for g in data["gpu"]:
            try:
                util = float(str(g["util_pct"]).strip().replace("%", ""))
            except ValueError:
                util = None
            res_items.append((f"GPU #{g['index']}", util))
    parts.append('<div class="charts">')
    parts.append(
        '<div class="chart">'
        + _svg_hbars("Задачи по статусу", status_chart, bar_color="#5c4db1")
        + "</div>"
    )
    parts.append(
        '<div class="chart">'
        + _svg_hbars("Задачи по worker_kind (всего)", wk_chart, bar_color="#b45c3a")
        + "</div>"
    )
    parts.append(
        '<div class="chart">' + _svg_pct_bars("Загрузка системы", res_items) + "</div>"
    )
    parts.append("</div>")

    parts.append('<div class="grid">')
    parts.append("<div><h3>По статусу</h3><table><thead><tr><th>Статус</th><th>Кол-во</th></tr></thead><tbody>")
    for st in STATUSES:
        parts.append(
            f"<tr><td>{html.escape(st)}</td><td>{data['by_status'].get(st, 0)}</td></tr>"
        )
    parts.append("</tbody></table></div>")

    parts.append("<div><h3>По worker_kind</h3>")
    parts.append(
        "<table><thead><tr><th>worker_kind</th>"
        + "".join(f"<th>{html.escape(st)}</th>" for st in STATUSES)
        + "</tr></thead><tbody>"
    )
    for wk, counts in sorted(data["by_worker"].items(), key=lambda x: x[0]):
        cells = "".join(f"<td>{counts.get(st, 0)}</td>" for st in STATUSES)
        parts.append(f"<tr><td>{html.escape(wk)}</td>{cells}</tr>")
    parts.append("</tbody></table></div></div></section>")

    parts.append('<section data-live="models"><h2>Метрики моделей</h2>')
    parts.append(
        "<p class=\"muted\">mean_loss / best_loss для текстовых чекпоинтов; train_axis_acc для mood. "
        "Сначала sidecar .meta.json, иначе чтение .pt (кэш 120 с).</p>"
    )
    parts.append(
        "<table><thead><tr>"
        "<th>Модель</th><th>Файл</th><th>mean_loss</th><th>train_axis_acc</th><th>Прочее</th>"
        "</tr></thead><tbody>"
    )
    for m in data["models"]:
        if m["label"] == "Mood":
            loss = "—"
            acc = _fmt_num(m["train_axis_acc"])
        else:
            loss = _fmt_num(m["mean_loss"])
            acc = "—"
        present = "есть" if m["present"] else "нет"
        parts.append(
            "<tr>"
            f"<td>{html.escape(m['label'])}</td>"
            f"<td><code>{html.escape(m['path'])}</code> ({present})</td>"
            f"<td>{loss}</td>"
            f"<td>{acc}</td>"
            f"<td>{html.escape(m['extra'] or '—')}</td>"
            "</tr>"
        )
    parts.append("</tbody></table></section>")
    parts.append(_render_resources_section(data))
    parts.append(_render_services_section(data))
    parts.append(_render_guilds_section(data))
    parts.append(f"<script>{PAGE_JS}</script>")
    parts.append("</body></html>")
    return "".join(parts).encode("utf-8")


class AdminHandler(BaseHTTPRequestHandler):
    db_path: Path = DEFAULT_DB

    def log_message(self, format, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        try:
            data = _collect_dashboard(self.db_path)
            body = _render_dashboard(data)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except sqlite3.Error as exc:
            msg = f"Ошибка SQLite: {exc}"
            self.send_error(500, msg)

    def do_HEAD(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()


def parse_args():
    parser = argparse.ArgumentParser(description="Локальная read-only панель KizaruAi")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Адрес (только локальный)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Порт HTTP")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Путь к SQLite bot.db")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("Разрешён только локальный bind (127.0.0.1).", file=sys.stderr)
        sys.exit(1)
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path

    AdminHandler.db_path = db_path
    server = ThreadingHTTPServer((args.host, args.port), AdminHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"admin panel: {url} db={db_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstop")
        server.shutdown()


if __name__ == "__main__":
    main()
