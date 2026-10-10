# Первый запуск (свежий clone)

После `git clone` в репозитории **нет** корпуса, логов, `bot.db`, чекпоинтов и pic-экспортов — они создаются локально и перечислены в `.gitignore`. Что можно и нельзя публиковать — [licensing.md](licensing.md).

## Требования к стенду

Всё, что ниже, в проект не входит и ставится на машину отдельно.

| Что | Нужно для | Обязательно? |
|-----|-----------|--------------|
| Windows + PowerShell | `start_all.ps1` и ops-скрипты в `var/` (используют `powershell`, `conhost`, `Get-CimInstance`) | для деплоя через `start_all.ps1` |
| Python 3.10 | всё | да |
| NVIDIA GPU с CUDA | обучение; инференс работает и на CPU | для обучения |
| LLM-сервер с OpenAI-совместимым API (llama-swap или аналог), который поднимает Qwen | LLM-часть пайплайна (учитель/судья в RLAIF-цикле) | **да** — без него LLM-часть не заработает. Сейчас клиент к серверу в `core/llm/` не реализован, см. [research/rlaif-song-loop.md](research/rlaif-song-loop.md) |
| MLflow-сервер (по умолчанию `http://127.0.0.1:5000`) | только логирование метрик обучения | нет, см. [training.md](training.md#mlflow-необязательно) |
| `ffmpeg` и `ffprobe` в PATH | GIF / видео в pic | для `anim` и видео в `/ai pic` |

Python-пакеты — все в `requirements.txt`:

- `torch`, `numpy`, `discord.py`, `pillow` — бот и обучение;
- `mlflow` — логирование метрик (сервер необязателен);
- `transformers`, `peft`, `bitsandbytes`, `accelerate`, `scikit-learn` — JEF / RL (`training/train_jef.py`, `train_rl_deepseek.py`, `build_jef_dataset.py --with-base-jeff`);
- `safetensors` — инициализация эмбеддингов из Navec (`--init-navec`);
- `datasets` — выборка stihi_ru в `download_poetry.py`; `pyarrow` — `import_verse_schema.py` (poetry_form);
- `lyricsgenius` — поиск артистов через Genius API при наличии токена;
- `psutil` — информация о процессах в админке;
- `flask`, `waitress` — Admin v2 (`python -m app.web`).

Qwen для JEF/RL скрипты грузят через `transformers` по имени модели (`--model` / `--jeff-model`, по умолчанию `Qwen/Qwen3-4B-Instruct-2507`); можно передать локальный путь.

## 1. Окружение

```bash
python -m pip install -r requirements.txt
```

На рабочей машине используется глобальный Python 3.10 без `.venv`; venv тоже подойдёт.

Скопируйте `.env.example` в `.env` и заполните:

- `DISCORD_TOKEN` — для `app.discord` (без него `start_all.ps1` просто не запускает Discord);
- `GENIUS_ACCESS_TOKEN` — опционально, для загрузки корпуса через Genius API.
- `LOKI_URL` — опционально, URL push API Loki для JSON-логов (`core/logging_setup.py`).

## 2. Корпус (обязательно перед обучением)

Итоговый файл: `data/corpus/kizaru_lyrics.txt` (в репозитории его нет).

**Вариант A — fallback без Genius:** положите `data/corpus/messages_fallback.csv`, затем:

```bash
python -m training.prepare_data
```

**Вариант B — Genius:**

```bash
python -m training.genius_orchestrator --workers 5
python -m training.prepare_data --source genius-cache
```

Подробнее: [runbook.md](runbook.md#корпус--genius), [corpus-cleaning.md](corpus-cleaning.md).

Сплиты для бенча `experiments/shared/train.txt` и `test.txt` в git не входят — их собирает `experiments/build_split.py` (см. `experiments/shared/README.md`).

## 3. Эмбеддинги (опционально, word-модели)

```bash
huggingface-cli download BorisTM/natasha_navec_hudlit_v1_12B_500K_300d_100q --local-dir data/embeddings/navec_hudlit
```

Подробно: [research/embeddings-vocab.md](research/embeddings-vocab.md).

## 4. Обучение или готовые веса

Либо обучите сами ([training.md](training.md)), либо положите `.pt` в `checkpoints/` с именами из README: `kizaru.pt`, `kizaru_word.pt`, `transformer.pt`, `gpt.pt`, `deepseek.pt`, `bert_gen.pt`, `mood.pt`.

## 5. Запуск бота

Все процессы разом (Windows): `powershell -ExecutionPolicy Bypass -File .\start_all.ps1`. По одному и подробности — [runbook.md](runbook.md#деплой-бота). SQLite-шина `data/bot.db` создаётся при первом обращении, логи — в `var/logs/`.

Админка: `python -m app.web` → http://127.0.0.1:8765/

## 6. Проверка без Discord

```bash
python -m app.cli.generate --arch lstm --start "Я" --db data/bot.db
```

Нужны запущенные оркестратор и воркер `lstm`.
