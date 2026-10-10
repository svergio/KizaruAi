"""Process matching, system stats, and workers API."""
import json
import subprocess
import sys

from pathlib import Path

from app.web.queries.common import (
    _connect_ro,
    _fmt_ts,
    _table_exists,
)

SERVICES = (
    ("discord", "app.discord"),
    ("orchestrator", "app.orchestrator"),
    ("worker_lstm", "app.workers.lstm"),
    ("worker_lstm_word", "app.workers.lstm_word"),
    ("worker_transformer", "app.workers.transformer"),
    ("worker_gpt", "app.workers.gpt"),
    ("worker_deepseek", "app.workers.deepseek"),
    ("worker_bert_gen", "app.workers.bert_gen"),
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
    "worker_bert_gen": "text_bert_gen",
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
    "text_bert_gen": "kind=text, model=bert-gen",
    "mood": "kind=mood",
    "pic_still": "kind=pic, mode 1/2 или статичный 3",
    "pic_video": "kind=pic, mode 3 animated",
}

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    psutil = None
    _HAS_PSUTIL = False


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
    from app.web import queries as q

    matches: dict[str, list[tuple[int, int]]] = {key: [] for key, _ in SERVICES}
    for pid, cmdline, rss in q._iter_python_processes():
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


def api_workers(db_path: Path) -> dict:
    from app.web.queries.stats import _collect_dashboard

    data = _collect_dashboard(db_path)
    svc = {s["key"]: s for s in data["services"]}
    workers = []
    for key, script in SERVICES:
        if not key.startswith("worker_"):
            continue
        s = svc.get(key, {})
        wk = SERVICE_TO_WORKER_META.get(key)
        workers.append(
            {
                "key": key,
                "module": script,
                "worker_kind": wk,
                "running": bool(s.get("running")),
                "pids": s.get("pids") or "",
                "rss": int(s.get("rss") or 0),
                "db_meta": bool(s.get("db_meta")),
                "accepts": WORKER_ACCEPTS.get(wk or "", ""),
            }
        )
    orch = svc.get("orchestrator", {})
    disc = svc.get("discord", {})
    return {
        "workers": workers,
        "orchestrator": {
            "running": bool(orch.get("running")),
            "pids": orch.get("pids") or "",
            "db_meta": bool(orch.get("db_meta")),
        },
        "discord": {
            "running": bool(disc.get("running")),
            "pids": disc.get("pids") or "",
            "db_meta": bool(disc.get("db_meta")),
        },
    }


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
