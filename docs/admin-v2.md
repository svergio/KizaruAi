# Admin v2 (Flask + D3)

Контракт и владение файлами. Реализация — волны 1–4 плана Admin v2.

## Стек

- Flask + waitress (или `app.run(threaded=True)`), `debug=False`
- Bind только `127.0.0.1:8765`
- Один CSS `app/web/static/admin.css`, JS `static/js/app.js` + `static/js/charts/*.js`
- Vendor: `static/vendor/d3.v7.min.js` (без CDN)
- БД `data/bot.db`: read-only в админке, кроме cancel и миграций схемы

## Время

| Слой | Часовой пояс |
|------|----------------|
| JSON API | ISO-8601 **UTC** |
| UI | **Europe/Moscow** |

## Словари

Источник правды: `app/web/contracts.py`.

### STATUS

| key | RU | bg | fg | CSS |
|-----|----|----|----|-----|
| queued | в очереди | `#e5e7eb` | `#374151` | `status-queued` |
| assigned | назначена | `#dbeafe` | `#1e40af` | `status-assigned` |
| running | выполняется | `#cffafe` | `#155e75` | `status-running` |
| done | готово | `#dcfce7` | `#166534` | `status-done` |
| error | ошибка | `#fee2e2` | `#991b1b` | `status-error` |
| cancelled | отменена | `#e5e7eb` | `#6b7280` | `status-cancelled` |

### KIND / WORKER_KIND

См. `KIND` и `WORKER_KIND` в `contracts.py` (человеческие имена для UI).

## JSON API

Ошибки `/api/*` → `{"error": "..."}` (без трейсбека). HTML-ошибки — короткая страница без стека.

Пагинация списков: `page` (с 1), `page_size` = 50 (`PAGE_SIZE`).

CSRF для `POST /api/jobs/<id>/cancel`: cookie `csrf_token` + заголовок `X-CSRF-Token` (или form-поле `csrf_token`).

| Метод | Путь | Query / тело | Ответ (см. examples в contracts) |
|-------|------|--------------|----------------------------------|
| GET | `/api/home` | — | главная: кластер, KPI, полоски, heatmap, хранение, задачи, bot_stats |
| GET | `/api/queue` | — | активные заявки (queued/assigned/running) |
| GET | `/api/jobs` | `status`, `kind`, `user`, `page` | страница задач |
| GET | `/api/jobs/<id>` | — | деталь задачи |
| POST | `/api/jobs/<id>/cancel` | CSRF | `{ok, id, status}` |
| GET | `/api/users` | `hide_cli=1` | список: только маска id (152-ФЗ), `key` |
| GET | `/api/users/<id>` | — | статистика без PII (`discord_id` / `cli` / `key`) |
| POST | `/api/users/<id>/profile` | CSRF | полный id / ник / avatar (Discord URL) |
| POST | `/api/users/<id>/delete` | CSRF | удалить пользователя и задачи (идемпотентно) |
| GET | `/api/workers` | — | процессы + meta |
| GET | `/api/models` | — | чекпоинты `checkpoints/*.pt` + meta без удержания весов |
| GET | `/api/stats/timeseries` | `metric`, `range` | точки `{t,v}` |
| GET | `/api/logs` | `service`, `level` | хвост логов |
| GET | `/api/stats` | — | legacy-сводка (совместимость) |

Примеры JSON — константы `EXAMPLE_*` и `ENDPOINTS` в `app/web/contracts.py`.

### Timeseries

- `metric`: `jobs_created` \| `jobs_done` \| `jobs_error` \| `jobs_cancelled` \| `active`
- `range`: `1h` \| `24h` \| `7d` \| `30d`

## HTML-страницы

| Путь | Вкладка |
|------|---------|
| `/` | Главная |
| `/queue` | Очередь |
| `/jobs` | Задачи |
| `/jobs/<id>` | Задача |
| `/users` | Пользователи |
| `/users/<id>` | Пользователь |
| `/workers` | Воркеры |
| `/models` | Модели |
| `/logs` | Логи |

Автообновление активной таблицы: 3 с, пауза в localStorage. Символы UI: `● ▲ ▼ ▸ › ‹`.

## Миграции (волна 1 B)

1. `users.display_name`, `users.avatar_url` — пишет Discord-бот при запросе; админка читает.
2. CLI-пользователь: `broker.cli_user_id()` → `"cli"`; склейка старых `cli:*` → один `cli`.
3. User `cli` без лимита «одна активная задача» в `enqueue` (лимит очереди и GPU-замок остаются).

## Владение файлами

| Область | Агент | Файлы |
|---------|-------|-------|
| Контракт | 0 | `docs/admin-v2.md`, `app/web/contracts.py` |
| Бэкенд + миграции | B | `app/web/__init__.py` (`create_app`), `api.py`, `views.py`, `queries/`, `__main__.py`; миграции в `core/bus/queue.py`; точечно `broker.py`, `app/discord`, `app/cli` |
| UI | F | `static/admin.css`, `static/js/app.js`, `templates/**/*.html` |
| D3 | V | `static/js/charts/*.js`, `static/vendor/d3.v7.min.js` |
| QA | T | `tests/test_admin_v2_*.py`, `tests/snapshots/`; порт `test_admin_web.py` / `test_stage_b_admin.py` |
| Интеграция | 2 | склейка; удаление `handler.py` / `render.py` / PAGE_CSS / http.server |
| Доки | 4 | `architecture.md`, `runbook.md`, `install.md`, `licensing.md`, финал этого файла + `LINKS.md` |

## Запуск

```bash
python -m app.web
# флаги: --host 127.0.0.1 --port 8765 --db data/bot.db --server waitress|flask
```

## Тесты (mapping)

| Было | Стало |
|------|--------|
| `tests/test_admin_web.py` (http.server) | тот же файл → Flask `test_client` / `create_app` |
| `tests/test_stage_b_admin.py` | тот же файл → Flask; снимок `tests/snapshots/admin_dashboard.html` |
| — | `tests/test_admin_v2_api.py` — контракт API/UI/vendor/миграции |

Удалено при интеграции: `app/web/handler.py`, `app/web/render.py` (и SVG-bar helpers в `queries.py`).

## Главная (мокап)

Статический мокап: [`docs/design/admin-home.html`](design/admin-home.html).

## Статус документа

Реализовано (волны 0–4 + T/H): контракт, Flask API/UI, главная `/api/home`, 152-ФЗ маски, `bot_stats`/retention, D3 charts, миграции `cli`/`display_name`, тесты `test_admin_v2_*` + порт старых admin-тестов. Удалены `handler.py` / `render.py` / overview donut.
