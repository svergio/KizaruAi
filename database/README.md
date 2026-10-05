# База бота (SQLite)

Файл шины заявок: **`data/bot.db`** (WAL, рядом `bot.db-wal` / `bot.db-shm`).

| Что | Где в коде |
|-----|------------|
| Схема, лимиты очереди | `core/bus/queue.py` |
| Маршрутизация `worker_kind` | `core/bus/routing.py`, `app/orchestrator/` |
| Discord / CLI payload | `core/bus/handlers.py`, `app/discord/`, `app/cli/generate.py`, `app/cli/image.py` |

Обзор процессов и статусов (`queued` -> … -> `done`): [docs/architecture.md](../docs/architecture.md). Запуск: [docs/runbook.md](../docs/runbook.md).

Корпус и импорт/экспорт pic — в `data/` ([data/README.md](../data/README.md)), не в этой папке. Здесь только описание БД.

Путь по умолчанию у всех процессов: `--db data/bot.db`.
