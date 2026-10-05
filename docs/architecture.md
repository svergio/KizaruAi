# Архитектура

Точка входа везде — `python -m ...`. Корневых shim-скриптов (`bot.py` и т.п.) нет.

## Дерево

```
KizaruAi/
  core/
    text/          # rnn, lm, sample, tokenize, checkpoint
    mood/          # model
    image/         # ascii, anim, render, probe, assets/
    corpus/        # lyrics, genius, genius_pool, pack
    bus/           # queue, routing, handlers, worker_loop, cli, broker
  app/
    discord/       # python -m app.discord
    orchestrator/  # python -m app.orchestrator
    workers/       # lstm, lstm_word, transformer, gpt, deepseek, mood, still, video
    web/           # python -m app.web
    cli/           # generate, image
  training/        # train, train_lm, train_mood, prepare_data, genius_orchestrator
  tests/
  docs/            # LINKS + research + runbook/architecture/services
  data/            # corpus, import/export, bot.db
  database/        # описание SQLite-шины (файл: data/bot.db)
  var/             # logs/, tmp/ рантайма (gitignore)
  experiments/
  checkpoints/
  notebooks/
```

| Папка | Роль |
|-------|------|
| `core/` | Domain lib, без Discord |
| `app/discord/` | Discord IO |
| `app/orchestrator/` | маршрутизация `queued` -> `assigned` |
| `app/workers/` | инференс text / mood / pic |
| `app/web/` | admin panel |
| `app/cli/` | generate / image enqueue |
| `training/` | обучение, prepare_data, genius pool |
| `docs/` | хаб ссылок и техдоки |
| `tests/` | unittest |
| `data/` | corpus, import/export, **`bot.db`** |
| `database/` | документация схемы БД |
| `var/` | runtime logs/temp |
| `experiments/` | night contest artifacts |
| `checkpoints/`, `notebooks/` | веса и исследования |

## Процессы и шина

- `python -m app.discord` — только ввод-вывод Discord.
- `python -m app.orchestrator` — маршрутизация заявок.
- `python -m app.workers.*` — инференс.
- Шина — SQLite `data/bot.db` (WAL, без Redis). Схема и лимиты: `core/bus/queue.py`; маршрутизация: `core/bus/routing.py`.
- Статусы: `queued` -> `assigned` -> `running` -> `done` / `error`.

Оркестратор переводит `queued` в `assigned` по `worker_kind`:

| `worker_kind` | Воркер |
|---------------|--------|
| `text_lstm` | `app.workers.lstm` |
| `text_lstm_word` | `app.workers.lstm_word` |
| `text_transformer` | `app.workers.transformer` |
| `text_gpt` | `app.workers.gpt` |
| `text_deepseek` | `app.workers.deepseek` |
| `mood` | `app.workers.mood` |
| `pic_still` | `app.workers.still` |
| `pic_video` | `app.workers.video` |

Воркер забирает свой inbox, ставит `running`, затем `done` или `error`. Лимиты очереди и размера payload — в `core/bus/queue.py`. Подробнее про файл БД: [../database/README.md](../database/README.md).

## Модули (кратко)

- **Текст** — LSTM char (`checkpoints/kizaru.pt`) и word (`checkpoints/kizaru_word.pt`); tokenizer слов — `core/text/tokenize.py`. Transformer / GPT — `core/text/lm.py`; DeepSeek-style MLA — `core/text/deepseek.py` (char по умолчанию; `--level word` — тот же словарь слов).
- **mood** — `core/mood/model.py`, `checkpoints/mood.pt`, `kind=vector2`. Оси: добрый/злой и веселый/грустный. Отдельного CLI нет: `/ai mood` или заявка `kind=mood`, payload `{"source": "..."}`.
- **pic** — `core/image/ascii.py`. **line**: градация и глифы из `core/image/assets/line_patterns.json`. **simple**: одиннадцатитоновая эмодзи-рампа. В Discord `line` / `simple` — текст, 31 колонка; `cli` и CLI `--mode line` — PNG/GIF, тайл 8×8. Зонды глифов — `notebooks/research/`.

Математика режимов, CE победителей, оси mood и глифы pic: [methods.md](methods.md).  
Живые smoke / CE / RPS: [services.md](services.md).
