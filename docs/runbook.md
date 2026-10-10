# Runbook

Запуск, корпус Genius, деплой бота, Discord и pic CLI. Обзор бота — [../README.md](../README.md). Обучение — [training.md](training.md). Методы / CE — [methods.md](methods.md). Архитектура шины — [architecture.md](architecture.md). Требования к стенду — [install.md](install.md#требования-к-стенду).

## Установка

Пошагово с нуля: **[install.md](install.md)**.

```bash
python -m pip install -r requirements.txt
python -m training.prepare_data
```

После `prepare_data` появляется локальный корпус `data/corpus/kizaru_lyrics.txt` (в свежем clone его нет).

## Корпус / Genius

`python -m training.prepare_data` (по умолчанию `--source auto`) пересобирает текст из кэша `genius_*.json`; если кэша нет — берёт `messages_fallback.csv`.

| Режим | Команда |
|-------|---------|
| Загрузка пулом | `python -m training.genius_orchestrator --workers 5` |
| Один поток Genius | `python -m training.prepare_data --source genius` |
| Только кэш, без сети | `python -m training.prepare_data --source genius-cache` |
| Старый HF | `python -m training.prepare_data --source cis` |
| Свой набор slug | `python -m training.prepare_data --artists Slug1 Slug2` |

Оркестратор раздаёт воркерам по одному артисту (по умолчанию до 25 воркеров, `--workers` ограничивает); воркер пишет только свой `genius_<slug>.json` и берёт следующего. Когда кэш всех slug полный, оркестратор сам собирает `kizaru_lyrics.txt` (если не указан `--download-only`).

Перед склейкой: `strip_genius_annotations` → нормализация символов → `core/corpus/clean.py` (`LM_STREAM`: пустые строки внутри песни выкидываются, каждая строка куплета остаётся своей строкой, между песнями один `\n`). Подробнее — [corpus-cleaning.md](corpus-cleaning.md). Лирика с [Genius](https://genius.com) — чужой контент, см. [LICENSE](../LICENSE).

После полной загрузки пула прогоните `python -m training.prepare_data --source genius-cache`, чтобы применить актуальные правила очистки ко всему кэшу.

### Пул Genius

Кого качать — файлы `training/artist_pools/*.txt` (slug по строке); если папки нет — встроенный список в `training/prepare_data.py`. Файл без `"complete": true` артиста не закрывает: оборванный json качается заново.

Раскладка `data/`: [../data/README.md](../data/README.md).

## Обучение

Все скрипты, флаги, MLflow и ops-скрипты — **[training.md](training.md)**. Кратко:

```bash
python -m training.train --level char --out checkpoints/kizaru.pt
python -m training.train_lm --arch gpt --level word --corpus data/mixes/ru_rap_v1/train.txt --out checkpoints/gpt.pt --patience 5
# чекпоинты: checkpoints/gpt_best.pt (лучший val_ru_rap), checkpoints/gpt_last.pt
python -m training.train_mood --out checkpoints/mood.pt
```

Пока пишутся `.pt`, не поднимайте текстовые воркеры на той же GPU.

## Деплой бота

Отдельных деплой-скриптов нет. Деплой = `start_all.ps1`, который поднимает оркестратор, воркеры и Discord-процесс, каждый в своём видимом окне консоли.

```powershell
powershell -ExecutionPolicy Bypass -File .\start_all.ps1
# опционально: -SkipDiscord / -SkipMood / -SkipPic
```

Что делает `start_all.ps1`:

- читает `.env` (KEY=VALUE) в переменные процесса;
- берёт `python` из PATH;
- запускает `app.orchestrator`, затем text-воркеры `lstm`, `lstm_word`, `transformer`, `gpt`, `deepseek`, `bert_gen`, затем `mood` и pic-воркеры `still` / `video` (если не пропущены);
- запускает `app.discord`, только если задан `DISCORD_TOKEN`;
- логи — `var/logs/<имя>.out.log`, список PID — `var/logs/start_all_pids.txt`.

Админку `start_all.ps1` не запускает — её поднимают отдельно.

Вручную, по одному процессу:

```bash
set DISCORD_TOKEN=...
python -m app.orchestrator --db data/bot.db
python -m app.workers.lstm checkpoints/kizaru.pt --db data/bot.db
python -m app.workers.lstm_word checkpoints/kizaru_word.pt --db data/bot.db
python -m app.workers.transformer checkpoints/transformer.pt --db data/bot.db
python -m app.workers.gpt checkpoints/gpt_ru_rap_v1-prod.pt --db data/bot.db
python -m app.workers.deepseek checkpoints/deepseek_v2_step14500.pt --db data/bot.db
python -m app.workers.bert_gen checkpoints/bert_gen.pt --db data/bot.db
python -m app.workers.mood checkpoints/mood.pt --db data/bot.db
python -m app.workers.still --db data/bot.db
python -m app.workers.video --db data/bot.db
python -m app.web --db data/bot.db
python -m app.discord --db data/bot.db
```

**VRAM:** текстовые и mood-воркеры держат веса на CPU; CUDA открывается только на первой джобе (`resolve_infer_device`). На время джобы модель переносится на GPU, после — обратно на CPU + `empty_cache` (`cuda_for_job`). Без джоб веса на GPU не висят; на Windows после первой джобы процесс может держать CUDA-контекст (~сотни MiB).

**Админ-панель (Admin v2):** в `start_all.ps1` как `kizaru-admin`, либо `python -m app.web` → http://127.0.0.1:8765/ (Flask+waitress, только loopback, `data/bot.db`; флаги `--host`, `--port`, `--db`, `--server waitress|flask`).

Вкладки: Обзор, Очередь, Задачи, Пользователи, Воркеры, Модели, Логи (+ детали задачи/пользователя). Автообновление 3 с (пауза в UI). Время в UI — Europe/Moscow; API — UTC. Контракт: [admin-v2.md](admin-v2.md).

Токен только из `DISCORD_TOKEN`. В файлы репозитория и в логи его не кладите.

Инвайт нужен со scope **`bot`** и **`applications.commands`** (иначе slash-команды не появятся):

https://discord.com/oauth2/authorize?client_id=1556306587511951380&permissions=380704&scope=bot%20applications.commands

Префикса нет, только slash. Бот синкает команды на свои гильдии, глобальный sync не делает.

### Slash `/ai`

| Команда | Параметры | Действие |
|---------|-----------|----------|
| `/ai text` | `model`: `lstm` \| `lstm-word` \| `transformer` \| `gpt` \| `deepseek` \| `bert-gen`; `start` (по умолчанию «Я»), `temp`, `lines`, `length`, `iters` | Продолжение лирики. Если воркер не запущен, бот сразу отвечает, что модель не запущена. |
| `/ai mood` | `text` (опционально) | Две оси: добрый/злой и веселый/грустный. Пустой `text` берётся из reply. |
| `/ai pic` | `mode`: `simple` \| `line` \| `render` \| `details`; `file`; `cli`, `scale`, `cells`, `dark`/`light`, `line_thr`/`details_thr` | `simple`/`line`/`details` — текст в чат (или файл при `cli`), `render` — PNG/GIF. |
| `/ai me` | — | Статистика заявок пользователя (без записи имени в БД). |
| `/ai cancel` | — | Отмена своей активной заявки, в том числе уже запущенной — на ближайшем шаге генерации (sample / deepseek StoppingCriteria / mood / pic / stream_anim). |
| `/ai help` | — | Справка в чате. |

Все параметры команд — в исходнике `app/discord/__main__.py`.

## Pic CLI

`python -m app.cli.image` только ставит заявку в очередь (оркестратор и `app.workers.still` / `app.workers.video` должны работать). Файл ищется по указанному пути, иначе в `data/import/`.

Флаги: `--db`, `--mode simple|line|render|details` (алиасы `2|1|3|4`), `--cells`, `--scale`, `--font` (`--smile-px`), `--no-upscale`, `--out`, `--dark`/`--light` (simple), `--line-thr`, `--details-thr`.

```bash
python -m app.cli.image still picture.jpg --mode simple
python -m app.cli.image still picture.jpg --mode line
python -m app.cli.image still picture.jpg --mode render
python -m app.cli.image still picture.jpg --mode details
python -m app.cli.image anim clip.mp4 --mode line
```

| Команда | Очередь | Результат |
|---------|---------|-----------|
| `still --mode simple` | `pic_still` | текст в stdout (PNG, если указан `--out`) |
| `still --mode line\|render\|details` | `pic_still` | PNG |
| `anim --mode line\|render\|details` | `pic_video` | GIF |
| `anim --mode simple` | — | exit 1 |

Для GIF нужны `ffmpeg` и `ffprobe`.

## Проверка

```bash
python -m app.cli.image still --help
python -m app.cli.image anim --help
python -m app.cli.generate --arch lstm --start "Я" --db data/bot.db
```
