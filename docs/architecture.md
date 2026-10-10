# Архитектура

Точка входа везде — `python -m ...` из корня проекта. Корневых shim-скриптов (`bot.py`, `train.py` и т.п.) нет.

## Дерево

```
KizaruAi/
  core/
    text/          # rnn, lm, deepseek, bert_gen, sample, tokenize, checkpoint, params, pretrained_vocab
    mood/          # model
    image/         # ascii, anim, render, line_mask_cuda, mask_pool, probe, assets/
    corpus/        # lyrics, genius, genius_pool, clean, pack
    bus/           # queue, routing, handlers, worker_loop, cancel_flag, cli, broker
    llm/           # заготовка под клиент LLM-сервера (не реализован)
  app/
    discord/       # python -m app.discord
    orchestrator/  # python -m app.orchestrator
    workers/       # lstm, lstm_word, transformer, gpt, deepseek, bert_gen, mood, still, video
    web/           # python -m app.web (админка)
    cli/           # generate, image
  training/        # обучение и подготовка данных (см. docs/training.md)
    scripts/       # build_jef_dataset, import_verse_schema, download_poetry
    artist_pools*/ # списки артистов Genius
  experiments/     # bench.py, build_split.py, shared/ (замороженные сплиты), результаты турниров
  docs/            # документация
  data/            # корпус, import/ и export/ (пустые; вход/выход CLI image по умолчанию), эмбеддинги, bot.db
  database/        # описание SQLite-шины (сам файл: data/bot.db)
  var/             # логи, временные файлы и ops-скрипты ночных прогонов
  checkpoints/     # веса
  notebooks/       # ноутбуки и зонды pic
  start_all.ps1    # запуск всех процессов бота
```

| Папка | Роль |
|-------|------|
| `core/` | библиотека предметной логики, без Discord |
| `app/discord/` | ввод-вывод Discord |
| `app/orchestrator/` | маршрутизация `queued` → `assigned` |
| `app/workers/` | инференс text / mood / pic |
| `app/web/` | админ-панель |
| `app/cli/` | постановка заявок generate / image из консоли |
| `training/` | обучение, prepare_data, пул Genius, JEF/RL |
| `experiments/` | бенч, сплиты, артефакты турниров |
| `data/` | корпус, import/ и export/ (пустые; вход/выход CLI image по умолчанию), **`bot.db`** |
| `database/` | документация схемы БД |
| `var/` | логи рантайма, temp, ops-скрипты (см. [training.md](training.md#ops-скрипты-в-var)) |
| `checkpoints/`, `notebooks/` | веса и исследования |

## Процессы и шина

- `python -m app.discord` — только ввод-вывод Discord.
- `python -m app.orchestrator` — маршрутизация заявок.
- `python -m app.workers.*` — инференс.
- `python -m app.web` — админка Admin v2 (Flask + waitress + D3) на http://127.0.0.1:8765/ (только `127.0.0.1`, `debug=False`).
  - Точка входа: `python -m app.web` → `app/web/__main__.py`; фабрика `create_app` в `app/web/__init__.py`.
  - Blueprints: `app/web/api.py` (JSON), `app/web/views.py` (Jinja).
  - Данные: `app/web/queries/`; контракт: `app/web/contracts.py`; CSS/JS: `static/admin.css`, `static/js/app.js`, `static/js/charts/*`, vendor `d3.v7.min.js`.
  - JSON: `/api/home`, `/api/queue`, `/api/jobs`, `/api/jobs/<id>`, `POST /api/jobs/<id>/cancel`, `/api/users`, `/api/users/<id>`, `POST /api/users/<id>/profile`, `POST /api/users/<id>/delete`, `/api/workers`, `/api/models`, `/api/stats/timeseries`, `/api/logs`, legacy `/api/stats`.
  - Страницы: `/`, `/queue`, `/jobs`, `/jobs/<id>`, `/users`, `/users/<id>`, `/workers`, `/models`, `/logs`. Подробнее: [admin-v2.md](admin-v2.md).
- Шина — SQLite `data/bot.db` (WAL, без Redis). Схема и лимиты: `core/bus/queue.py`; маршрутизация: `core/bus/routing.py`.
- Статусы: `queued` → `assigned` → `running` → `done` / `error` / `cancelled`.
- Колонки jobs: `worker_pid`, `lease_until`, `attempts`. Аренда продлевается daemon-потоком воркера; после `MAX_RUNTIME_SEC[kind]` — `error` с причиной `timeout`, аренда не продлевается.
- Одна GPU-задача: `GPU_KINDS` в `core/bus/queue.py` — все `worker_kind`, которые трогают CUDA (text/mood через `cuda_for_job`, `pic_still` / `pic_video` через `pick_device`). `claim_assigned` для kind из `GPU_KINDS` в том же `BEGIN IMMEDIATE` возвращает `None`, если уже есть `running` с любым kind из `GPU_KINDS` и неистёкшей арендой; истёкшая аренда слот освобождает. `done` / `error` пишутся только после `model.to("cpu")` и `empty_cache` в `worker_loop`. Задача в `assigned`, ждущая GPU при живом воркере, жнецом `reap_offline_assigned` не снимается.
- Отмена running: `JobStore.cancel` → `cancelled`; lease-daemon / `MAX_RUNTIME_SEC` ставят `threading.Event` (`core/bus/cancel_flag.py`). Флаг читают: `core/text/sample.py` (LSTM/GPT/transformer), `CancelFlagStoppingCriteria` в `core/text/deepseek.py`, итерации `bert_gen`, `predict_mood` (до/после шага), `job_pic_still` / `job_pic_video` (между шагами), `stream_anim` (между батчами; kill ffmpeg + удаление недописанного mp4). Без опроса БД в генерации; после прерывания — `empty_cache`.

Все процессы бота разом поднимает `start_all.ps1` — см. [runbook.md](runbook.md#деплой-бота).

Оркестратор переводит `queued` в `assigned` по `worker_kind`:

| `worker_kind` | Воркер |
|---------------|--------|
| `text_lstm` | `app.workers.lstm` |
| `text_lstm_word` | `app.workers.lstm_word` |
| `text_transformer` | `app.workers.transformer` |
| `text_gpt` | `app.workers.gpt` |
| `text_deepseek` | `app.workers.deepseek` |
| `text_bert_gen` | `app.workers.bert_gen` |
| `mood` | `app.workers.mood` |
| `pic_still` | `app.workers.still` |
| `pic_video` | `app.workers.video` |

Воркер забирает свой inbox, ставит `running`, затем `done` или `error`. Про файл БД: [../database/README.md](../database/README.md).

## Модули (кратко)

- **Текст** — char LSTM (`checkpoints/kizaru.pt`, vocab ~2613) и word-стек на общем словаре ~99868: `lstm-word`, `transformer`, `gpt`, `deepseek`, `bert-gen` (emb 300). Токенизатор слов — `core/text/tokenize.py`. Transformer/GPT — `core/text/lm.py`; MLA — `core/text/deepseek.py`; MLM — `core/text/bert_gen.py`.
- **mood** — `core/mood/model.py`, `checkpoints/mood.pt`, `kind=vector2`. Оси: добрый/злой и веселый/грустный. Отдельного CLI нет: `/ai mood` или заявка `kind=mood`.
- **pic** — `core/image/ascii.py`. Четыре режима: **simple** (2 глифа dark/light), **line** (глифы `line_patterns.json`, Rec.601), **render** (OKLab L* + цветовые края, файл), **details** (мягкий словарь `details_patterns.json`). Зонды глифов — `notebooks/research/`.
- **LLM** — `core/llm/` пока пустая заготовка: клиента к OpenAI-совместимому серверу нет.

Математика режимов, train CE, оси mood и глифы pic: [methods.md](methods.md). Замеры smoke / CE / RPS: [services.md](services.md).
