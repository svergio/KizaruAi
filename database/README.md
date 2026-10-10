# База бота (SQLite)

Файл шины заявок: **`data/bot.db`** (WAL, рядом `bot.db-wal` / `bot.db-shm`).

| Что | Где в коде |
|-----|------------|
| Схема, лимиты очереди | `core/bus/queue.py` |
| Маршрутизация `worker_kind` | `core/bus/routing.py`, `app/orchestrator/` |
| Discord / CLI payload | `core/bus/handlers.py`, `app/discord/`, `app/cli/generate.py`, `app/cli/image.py` |

Таблица `jobs`: помимо базовых полей — `worker_kind`, `worker_pid`, `lease_until`, `attempts` (DEFAULT 0), `result`. Терминальные статусы: `done`, `error`, `cancelled`. `MAX_QUEUED_JOBS` считает `queued`+`assigned`. После `MAX_RUNTIME_SEC[kind]` running → `error` (`timeout`). `complete`/`fail` только при `status=running` и своём `worker_pid`.

Обзор процессов и статусов: [docs/architecture.md](../docs/architecture.md). Запуск: [docs/runbook.md](../docs/runbook.md).

Корпус и импорт/экспорт pic — в `data/` ([data/README.md](../data/README.md)), не в этой папке. Здесь только описание БД.

Путь по умолчанию у всех процессов: `--db data/bot.db`.
