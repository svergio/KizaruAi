# Первый запуск (свежий clone)

После `git clone` в репозитории **нет** корпуса, логов, `bot.db`, чекпоинтов и pic-экспортов — они создаются у вас локально и перечислены в [.gitignore](../.gitignore). В git могут быть только README по `data/`, `checkpoints/.gitkeep` и (опционально) `.pt` / embeddings — см. [licensing.md](licensing.md).

## 1. Окружение

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Скопируйте [.env.example](../.env.example) в `.env` и заполните по необходимости:

- `DISCORD_TOKEN` — для `app.discord`
- `GENIUS_ACCESS_TOKEN` — опционально, для загрузки корпуса через Genius API

## 2. Корпус (обязательно перед обучением)

Целевой merged-файл после сборки: `data/corpus/kizaru_lyrics.txt` (в репозитории его нет).

**Вариант A — только fallback (быстро, без Genius):** положите или сгенерируйте `data/corpus/messages_fallback.csv`, затем:

```bash
python -m training.prepare_data
```

**Вариант B — Genius:** скачайте кэш `data/corpus/genius_<slug>.json`, затем соберите текст:

```bash
python -m training.genius_orchestrator --workers 5
python -m training.prepare_data --source genius-cache
```

Подробнее: [runbook.md](runbook.md) (корпус / Genius), [corpus-cleaning.md](corpus-cleaning.md).

Сплиты для bench/обучения: `experiments/shared/train.txt` и `test.txt` **не в git** — их нужно пересобрать у себя из корпуса (см. `experiments/shared/README.md` и скрипты в `experiments/`).

## 3. Embeddings (опционально, word LSTM)

```bash
huggingface-cli download BorisTM/natasha_navec_hudlit_v1_12B_500K_300d_100q --local-dir data/embeddings/navec_hudlit
```

Инструкция: [research/embeddings-vocab.md](research/embeddings-vocab.md). Каталог `.cache/` внутри embeddings не коммитится.

## 4. Обучение или готовые веса

Либо обучите сами ([runbook.md](runbook.md)), либо положите `.pt` в `checkpoints/` (имена как в README: `kizaru.pt`, `kizaru_word.pt`, …).

## 5. Запуск бота

SQLite-шина создаётся при первом обращении: `data/bot.db`. Логи процессов — в `var/logs/` (каталог появится при запуске).

```bash
python -m app.orchestrator --db data/bot.db
python -m app.workers.lstm checkpoints/kizaru.pt --db data/bot.db
python -m app.workers.lstm_word checkpoints/kizaru_word.pt --db data/bot.db
python -m app.workers.transformer checkpoints/transformer.pt --db data/bot.db
python -m app.workers.gpt checkpoints/gpt.pt --db data/bot.db
python -m app.workers.deepseek checkpoints/deepseek.pt --db data/bot.db
python -m app.workers.mood checkpoints/mood.pt --db data/bot.db
python -m app.workers.still --db data/bot.db
python -m app.workers.video --db data/bot.db
python -m app.discord --db data/bot.db
```

Админка: `python -m app.web` → http://127.0.0.1:8765/

## 6. Проверка без Discord

```bash
python -m app.cli.generate --arch lstm --start "Я" --db data/bot.db
python -m unittest discover -s tests -p "test_*.py"
```

Часть тестов ожидает локальный `data/corpus/messages_fallback.csv` или чекпоинты — на пустом clone это нормально; сначала выполните шаги 2–4.

Полный деплой и флаги: [runbook.md](runbook.md). Обзор бота: [../README.md](../README.md).
