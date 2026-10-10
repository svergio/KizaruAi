"""Services ops API: process status and runbook hints (no start/stop from admin)."""
from __future__ import annotations

from pathlib import Path

from app.web.queries.stats import _collect_dashboard, _state_label
from app.web.queries.workers import SERVICE_TO_WORKER_META, SERVICES, _format_bytes

DEPLOY_ALL_CMD = r"powershell -ExecutionPolicy Bypass -File .\start_all.ps1"
OPS_NOTE = (
    "Перезапуск и остановка из админки недоступны: бот поднимается отдельными "
    "консолями через start_all.ps1. Закройте окно процесса или перезапустите стек командой ниже."
)

SERVICE_UI: dict[str, dict[str, str]] = {
    "discord": {
        "label": "Discord-бот",
        "log_stem": "discord.out",
        "group": "core",
        "manual_cmd": "python -m app.discord --db data/bot.db",
    },
    "orchestrator": {
        "label": "Оркестратор",
        "log_stem": "orchestrator.out",
        "group": "core",
        "manual_cmd": "python -m app.orchestrator --db data/bot.db",
    },
    "worker_lstm": {
        "label": "Воркер LSTM (char)",
        "log_stem": "worker-lstm.out",
        "group": "worker",
        "manual_cmd": "python -m app.workers.lstm checkpoints/kizaru.pt --db data/bot.db",
    },
    "worker_lstm_word": {
        "label": "Воркер LSTM (word)",
        "log_stem": "worker-lstm-word.out",
        "group": "worker",
        "manual_cmd": "python -m app.workers.lstm_word checkpoints/kizaru_word.pt --db data/bot.db",
    },
    "worker_transformer": {
        "label": "Воркер Transformer",
        "log_stem": "worker-transformer.out",
        "group": "worker",
        "manual_cmd": "python -m app.workers.transformer checkpoints/transformer.pt --db data/bot.db",
    },
    "worker_gpt": {
        "label": "Воркер GPT",
        "log_stem": "worker-gpt.out",
        "group": "worker",
        "manual_cmd": "python -m app.workers.gpt checkpoints/gpt_ru_rap_v1-prod.pt --db data/bot.db",
    },
    "worker_deepseek": {
        "label": "Воркер DeepSeek",
        "log_stem": "worker-deepseek.out",
        "group": "worker",
        "manual_cmd": "python -m app.workers.deepseek checkpoints/deepseek_v2_step14500.pt --db data/bot.db",
    },
    "worker_bert_gen": {
        "label": "Воркер BERT-gen",
        "log_stem": "worker-bert-gen.out",
        "group": "worker",
        "manual_cmd": "python -m app.workers.bert_gen checkpoints/bert_gen.pt --db data/bot.db",
    },
    "worker_mood": {
        "label": "Воркер mood",
        "log_stem": "worker-mood.out",
        "group": "worker",
        "manual_cmd": "python -m app.workers.mood checkpoints/mood.pt --db data/bot.db",
    },
    "worker_still": {
        "label": "Воркер pic (still)",
        "log_stem": "worker-pic-still.out",
        "group": "worker",
        "manual_cmd": "python -m app.workers.still --db data/bot.db",
    },
    "worker_video": {
        "label": "Воркер pic (video)",
        "log_stem": "worker-pic-video.out",
        "group": "worker",
        "manual_cmd": "python -m app.workers.video --db data/bot.db",
    },
    "web": {
        "label": "Админка (этот процесс)",
        "log_stem": "admin.out",
        "group": "admin",
        "manual_cmd": "python -m app.web --db data/bot.db",
    },
}


def _meta_key_for_service(key: str) -> str | None:
    if key == "orchestrator":
        return "orchestrator"
    if key in SERVICE_TO_WORKER_META:
        return f"worker:{SERVICE_TO_WORKER_META[key]}"
    return None


def api_services(db_path: Path) -> dict:
    data = _collect_dashboard(db_path)
    by_key = {s["key"]: s for s in data["services"]}
    meta_map = data.get("meta_map") or {}
    items = []
    for key, script in SERVICES:
        svc = by_key.get(key, {})
        ui = SERVICE_UI.get(key, {"label": key, "log_stem": "", "group": "other", "manual_cmd": ""})
        meta_key = _meta_key_for_service(key)
        meta_row = meta_map.get(meta_key) if meta_key else None
        proc_running = bool(svc.get("running"))
        state, state_ru = _state_label(proc_running, meta_row)
        wk = SERVICE_TO_WORKER_META.get(key)
        items.append(
            {
                "key": key,
                "label": ui["label"],
                "module": script,
                "group": ui["group"],
                "worker_kind": wk,
                "running": proc_running,
                "health": state,
                "health_label": state_ru,
                "pids": svc.get("pids") or "",
                "rss": int(svc.get("rss") or 0),
                "rss_human": _format_bytes(int(svc.get("rss") or 0)),
                "db_meta": bool(svc.get("db_meta")),
                "log_stem": ui["log_stem"],
                "manual_cmd": ui["manual_cmd"],
            }
        )
    return {
        "generated_at": data["generated_at"],
        "db_ok": data["db_ok"],
        "running_count": data["running_count"],
        "expected_count": data["expected_count"],
        "deploy_all": DEPLOY_ALL_CMD,
        "ops_note": OPS_NOTE,
        "pid_file": "var/logs/start_all_pids.txt",
        "runbook_anchor": "docs/runbook.md#деплой-бота",
        "services": items,
    }
