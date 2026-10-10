"""Admin v2 JSON API contract and shared dictionaries (Wave 0).

Implementation lives in api/views/queries; this module is the single source
for status/kind labels, colors, and documented response shapes.
"""
from __future__ import annotations

from typing import Any

# API timestamps: ISO-8601 UTC (Z or +00:00). UI displays Europe/Moscow.
API_TIMEZONE = "UTC"
UI_TIMEZONE = "Europe/Moscow"
PAGE_SIZE = 50
CSRF_HEADER = "X-CSRF-Token"
CSRF_COOKIE = "csrf_token"
CSRF_FORM_FIELD = "csrf_token"

STATUSES = ("queued", "assigned", "running", "done", "error", "cancelled")

STATUS: dict[str, dict[str, str]] = {
    "queued": {
        "label": "в очереди",
        "bg": "#e5e7eb",
        "fg": "#374151",
        "css": "status-queued",
    },
    "assigned": {
        "label": "назначена",
        "bg": "#dbeafe",
        "fg": "#1e40af",
        "css": "status-assigned",
    },
    "running": {
        "label": "выполняется",
        "bg": "#cffafe",
        "fg": "#155e75",
        "css": "status-running",
    },
    "done": {
        "label": "готово",
        "bg": "#dcfce7",
        "fg": "#166534",
        "css": "status-done",
    },
    "error": {
        "label": "ошибка",
        "bg": "#fee2e2",
        "fg": "#991b1b",
        "css": "status-error",
    },
    "cancelled": {
        "label": "отменена",
        "bg": "#e5e7eb",
        "fg": "#6b7280",
        "css": "status-cancelled",
    },
}

KIND: dict[str, str] = {
    "text": "текст",
    "mood": "настроение",
    "pic": "картинка",
}

WORKER_KIND: dict[str, str] = {
    "text_lstm": "LSTM (символы)",
    "text_lstm_word": "LSTM (слова)",
    "text_transformer": "Transformer",
    "text_gpt": "GPT",
    "text_deepseek": "DeepSeek",
    "text_bert_gen": "BERT-gen",
    "mood": "Mood",
    "pic_still": "Pic (кадр)",
    "pic_video": "Pic (видео)",
}

TIMESERIES_METRICS = (
    "jobs_created",
    "jobs_done",
    "jobs_error",
    "jobs_cancelled",
    "active",
)
TIMESERIES_RANGES = ("1h", "24h", "7d", "30d")

ERROR_BODY = {"error": "string"}


def status_labels() -> dict[str, str]:
    return {k: v["label"] for k, v in STATUS.items()}


def status_badge_classes() -> dict[str, str]:
    return {k: v["css"] for k, v in STATUS.items()}


# --- Example responses (documentation / contract tests) ---

EXAMPLE_HOME: dict[str, Any] = {
    "generated_at": "2026-10-10T09:00:00+00:00",
    "db_ok": True,
    "cluster": {
        "nodes": [
            {"id": "discord", "label": "Discord", "health": "ok"},
            {"id": "orchestrator", "label": "Оркестратор", "health": "ok"},
            {"id": "queue", "label": "Очередь + SQLite", "health": "ok"},
            {"id": "workers", "label": "Воркеры", "health": "warn", "gpu_up": 2, "gpu_expected": 9},
            {"id": "models", "label": "Модели", "health": "ok"},
        ],
        "edges": [
            {"from": "discord", "to": "queue", "count": 2},
            {"from": "queue", "to": "orchestrator", "count": 1},
            {"from": "orchestrator", "to": "workers", "count": 1},
            {"from": "workers", "to": "models", "count": 0},
        ],
        "diagram": {
            "ingress": [{"id": "discord", "label": "Discord-бот", "health": "ok"}],
            "bus": [{"id": "orchestrator", "label": "Оркестратор", "health": "ok"}],
            "workers": {
                "text": [{"id": "worker_gpt", "label": "gpt", "health": "ok"}],
                "mood": [{"id": "worker_mood", "label": "mood", "health": "ok"}],
                "pic": [{"id": "worker_still", "label": "pic still", "health": "ok"}],
            },
            "models": [
                {
                    "id": "model:gpt.pt",
                    "name": "gpt.pt",
                    "label": "gpt",
                    "health": "ok",
                    "present": True,
                    "missing_metadata": False,
                }
            ],
        },
    },
    "kpi": {
        "active": 2,
        "done_24h": 10,
        "error_24h": 1,
        "users_active": 3,
        "guilds": 1,
        "sparkline_7d": [{"t": "2026-10-04", "v": 1}, {"t": "2026-10-10", "v": 4}],
    },
    "bars": {
        "queue": {"value": 1, "max": 64, "pct": 1.56, "level": "ok"},
        "cpu": {"value": 12.5, "max": 100, "pct": 12.5, "level": "ok"},
        "mem": {"value": 1073741824, "max": 4294967296, "pct": 25.0, "level": "ok"},
        "gpu": {"value": None, "max": 100, "pct": None, "level": "na"},
        "thresholds": {"warn": 70, "error": 90},
    },
    "audience": {
        "strip": {
            "guilds": 1,
            "users_active": 3,
            "users_total": 12,
            "members_total": 5,
            "members_online": None,
        },
        "users_7d": [{"t": "2026-10-04", "v": 1}, {"t": "2026-10-10", "v": 4}],
        "members_7d": [{"t": "2026-10-04", "v": 5}, {"t": "2026-10-10", "v": 5}],
    },
    "heatmap": {
        "rows": 7,
        "cols": 24,
        "cells": [{"d": 0, "h": 0, "v": 0}, {"d": 6, "h": 12, "v": 3}],
    },
    "storage": {
        "items": [{"label": "БД / data", "path": "data", "bytes": 1024, "human": "1.0 KiB"}]
    },
    "recent_jobs": [
        {
            "id": 42,
            "id_str": "42",
            "user_mask": "1234…56",
            "kind": "text",
            "status": "done",
            "created_at": "2026-10-10T08:59:00+00:00",
            "stages": [{"key": "created", "at": "2026-10-10T08:59:00+00:00"}],
        }
    ],
    "bot_stats": {
        "latest": {
            "ts": "2026-10-10T09:00:00+00:00",
            "guilds": 1,
            "members_total": 5,
            "members_online": None,
            "latency_ms": 42.0,
        },
        "members_online_available": False,
    },
    "by_status": {
        "queued": 1,
        "assigned": 0,
        "running": 1,
        "done": 10,
        "error": 2,
        "cancelled": 1,
    },
    "thresholds": {"warn": 70, "error": 90},
}

EXAMPLE_QUEUE: dict[str, Any] = {
    "jobs": [
        {
            "id": 42,
            "user": "123456",
            "display_name": "alice",
            "kind": "text",
            "worker_kind": "text_gpt",
            "status": "running",
            "created_at": "2026-10-10T08:59:00+00:00",
            "started_at": "2026-10-10T08:59:30+00:00",
            "eta_sec": 12.0,
        }
    ],
    "pipeline": {
        "clients": 2,
        "discord": 2,
        "cli": 0,
        "queued": 1,
        "assigned": 0,
        "running": 1,
    },
}

EXAMPLE_JOBS_LIST: dict[str, Any] = {
    "page": 1,
    "page_size": 50,
    "total": 1,
    "jobs": [
        {
            "id": 42,
            "user": "123456",
            "display_name": "alice",
            "kind": "text",
            "worker_kind": "text_gpt",
            "status": "done",
            "created_at": "2026-10-10T08:59:00+00:00",
            "started_at": "2026-10-10T08:59:30+00:00",
            "finished_at": "2026-10-10T09:00:00+00:00",
            "error": None,
            "preview": "модель gpt, начало: Я",
        }
    ],
}

EXAMPLE_JOB_DETAIL: dict[str, Any] = {
    "id": 42,
    "user": "123456",
    "display_name": "alice",
    "avatar_url": None,
    "kind": "text",
    "worker_kind": "text_gpt",
    "status": "running",
    "created_at": "2026-10-10T08:59:00+00:00",
    "started_at": "2026-10-10T08:59:30+00:00",
    "finished_at": None,
    "payload_preview": "модель gpt, начало: Я",
    "result_preview": "",
    "error": None,
    "worker_pid": 4242,
    "lease_until": "2026-10-10T09:01:00+00:00",
    "attempts": 1,
    "cancellable": True,
}

EXAMPLE_CANCEL_OK: dict[str, Any] = {"ok": True, "id": 42, "status": "cancelled"}
EXAMPLE_CANCEL_FAIL: dict[str, Any] = {"error": "not cancellable"}

EXAMPLE_USERS_LIST: dict[str, Any] = {
    "hide_cli": True,
    "generated_at": "2026-10-10T09:00:00+00:00",
    "summary": {
        "users_total": 12,
        "users_active_15m": 2,
        "users_active_24h": 5,
        "users_active_7d": 9,
        "users_with_jobs": 8,
        "guilds": 3,
        "guild_members_sum": 120,
        "jobs_active": 1,
        "cli_rows": 0,
    },
    "audience": {
        "strip": {
            "guilds": 3,
            "users_active": 2,
            "users_total": 12,
            "members_total": 120,
            "members_online": 18,
        },
        "users_7d": [{"t": "2026-10-04", "v": 1}],
        "members_7d": [{"t": "2026-10-04", "v": 120}],
    },
    "activity": {
        "by_recency": [{"label": "≤15 мин", "count": 2}],
        "by_jobs": [{"label": "0 задач", "count": 4}],
    },
    "jobs_sparkline_7d": [{"t": "2026-10-04", "v": 3}],
    "kpi_jobs": {"done_24h": 10, "error_24h": 1},
    "bot_stats": {"latest": None, "members_online_available": True},
    "users": [
        {
            "key": 1,
            "id": "1234…56",
            "id_mask": "1234…56",
            "first_seen": "2026-10-01T00:00:00+00:00",
            "last_seen": "2026-10-10T08:00:00+00:00",
            "total": 5,
            "active": 0,
        }
    ],
}

EXAMPLE_USER_DETAIL: dict[str, Any] = {
    "key": 1,
    "id": "1234…56",
    "id_mask": "1234…56",
    "display_name": None,
    "avatar_url": None,
    "first_seen": "2026-10-01T00:00:00+00:00",
    "last_seen": "2026-10-10T08:00:00+00:00",
    "total": 5,
    "by_kind": {"text": 3, "mood": 1, "pic": 1},
    "by_status": {
        "queued": 0,
        "assigned": 0,
        "running": 0,
        "done": 4,
        "error": 1,
        "cancelled": 0,
    },
    "mean_wait_sec": 2.5,
    "mean_run_sec": 8.0,
}

EXAMPLE_USER_PROFILE: dict[str, Any] = {
    "key": 1,
    "id": "123456789012345678",
    "id_mask": "1234…78",
    "display_name": "alice",
    "avatar_url": "https://cdn.discordapp.com/avatars/1/a.png",
    "first_seen": "2026-10-01T00:00:00+00:00",
    "last_seen": "2026-10-10T08:00:00+00:00",
}

EXAMPLE_SERVICES: dict[str, Any] = {
    "generated_at": "2026-10-10T09:00:00+00:00",
    "db_ok": True,
    "running_count": 2,
    "expected_count": 12,
    "deploy_all": r"powershell -ExecutionPolicy Bypass -File .\start_all.ps1",
    "ops_note": "Перезапуск из админки недоступен.",
    "pid_file": "var/logs/start_all_pids.txt",
    "runbook_anchor": "docs/runbook.md#деплой-бота",
    "services": [
        {
            "key": "orchestrator",
            "label": "Оркестратор",
            "module": "app.orchestrator",
            "group": "core",
            "worker_kind": None,
            "running": True,
            "health": "on",
            "health_label": "жив",
            "pids": "100",
            "rss": 1048576,
            "rss_human": "1.0 MiB",
            "db_meta": True,
            "log_stem": "orchestrator.out",
            "manual_cmd": "python -m app.orchestrator --db data/bot.db",
        }
    ],
}

EXAMPLE_WORKERS: dict[str, Any] = {
    "workers": [
        {
            "key": "worker_gpt",
            "module": "app.workers.gpt",
            "worker_kind": "text_gpt",
            "running": True,
            "pids": "4242",
            "rss": 1048576,
            "db_meta": True,
            "accepts": "kind=text, model=gpt",
        }
    ],
    "orchestrator": {
        "running": True,
        "pids": "100",
        "db_meta": True,
    },
    "discord": {"running": False, "pids": "", "db_meta": False},
}

EXAMPLE_MODELS: dict[str, Any] = {
    "models": [
        {
            "name": "gpt_ru_rap_v1-prod.pt",
            "path": "checkpoints/gpt_ru_rap_v1-prod.pt",
            "present": True,
            "prod": True,
            "missing_metadata": False,
            "tokenizer": "bpe",
            "vocab_size": 99868,
            "step": 15000,
            "val_ru_rap": 1.23,
            "bpc": 2.1,
            "date": "2026-10-01T12:00:00+00:00",
            "arch": "gpt",
            "level": "bpe",
            "note": None,
        }
    ]
}

EXAMPLE_TIMESERIES: dict[str, Any] = {
    "metric": "jobs_created",
    "range": "24h",
    "bucket_sec": 3600,
    "points": [
        {"t": "2026-10-09T10:00:00+00:00", "v": 3},
        {"t": "2026-10-09T11:00:00+00:00", "v": 1},
    ],
}

EXAMPLE_LOGS: dict[str, Any] = {
    "service": "web",
    "level": "INFO",
    "lines": [
        {
            "t": "2026-10-10T09:00:00+00:00",
            "level": "INFO",
            "service": "web",
            "msg": "admin panel started",
        }
    ],
}

EXAMPLE_STATS_LEGACY: dict[str, Any] = {
    "by_status": {"done": 10, "error": 2, "queued": 1},
    "history_total": 13,
    "active": 2,
    "db_ok": True,
    "terminal_statuses": ["done", "error", "cancelled"],
}

ENDPOINTS: dict[str, dict[str, Any]] = {
    "GET /api/home": {
        "query": {},
        "response": EXAMPLE_HOME,
    },
    "GET /api/queue": {
        "query": {},
        "response": EXAMPLE_QUEUE,
    },
    "GET /api/jobs": {
        "query": {
            "status": "optional str, one of STATUSES or empty",
            "kind": "optional str, one of KIND keys",
            "user": "optional discord_id / cli",
            "page": "optional int >= 1, default 1",
        },
        "response": EXAMPLE_JOBS_LIST,
    },
    "GET /api/jobs/<id>": {
        "query": {},
        "response": EXAMPLE_JOB_DETAIL,
    },
    "POST /api/jobs/<id>/cancel": {
        "headers": {CSRF_HEADER: "csrf cookie value"},
        "response_ok": EXAMPLE_CANCEL_OK,
        "response_error": EXAMPLE_CANCEL_FAIL,
    },
    "GET /api/users": {
        "query": {"hide_cli": "1 to hide user id=cli"},
        "response": EXAMPLE_USERS_LIST,
    },
    "GET /api/users/<id>": {
        "query": {},
        "response": EXAMPLE_USER_DETAIL,
    },
    "POST /api/users/<id>/profile": {
        "headers": {CSRF_HEADER: "csrf cookie value"},
        "response": EXAMPLE_USER_PROFILE,
    },
    "POST /api/users/<id>/delete": {
        "headers": {CSRF_HEADER: "csrf cookie value"},
        "response": {"ok": True, "deleted": True, "id_mask": "1234…78"},
    },
    "GET /api/services": {
        "query": {},
        "response": EXAMPLE_SERVICES,
        "note": "process status + runbook hints; no start/stop actions",
    },
    "GET /api/workers": {
        "query": {},
        "response": EXAMPLE_WORKERS,
    },
    "GET /api/models": {
        "query": {},
        "response": EXAMPLE_MODELS,
    },
    "GET /api/stats/timeseries": {
        "query": {
            "metric": "one of TIMESERIES_METRICS",
            "range": "one of TIMESERIES_RANGES",
        },
        "response": EXAMPLE_TIMESERIES,
    },
    "GET /api/logs": {
        "query": {
            "service": "optional service key",
            "level": "optional DEBUG|INFO|WARNING|ERROR",
        },
        "response": EXAMPLE_LOGS,
    },
    "GET /api/stats": {
        "query": {},
        "response": EXAMPLE_STATS_LEGACY,
        "note": "legacy alias used by older clients/tests",
    },
}
