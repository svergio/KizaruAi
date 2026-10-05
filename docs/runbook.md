# Runbook

Команды запуска, корпус Genius, обучение, Discord и pic CLI. Showcase бота — [../README.md](../README.md). Методы / CE — [methods.md](methods.md). Архитектура шины — [architecture.md](architecture.md).

## Установка

Пошаговый первый запуск без предположения, что корпус уже в git: **[install.md](install.md)**.

```bash
python -m pip install -r requirements.txt
python -m training.prepare_data
```

После успешного `prepare_data` merged-корпус появляется локально: `data/corpus/kizaru_lyrics.txt` (файла нет в свежем clone).

## Корпус / Genius

`python -m training.prepare_data` (`--source auto`) пересобирает текст из кэша `genius_*.json`. Если кэша нет — берёт `messages_fallback.csv`.

| Режим | Команда |
|-------|---------|
| Полная загрузка пачкой | `python -m training.genius_orchestrator --workers 5` |
| Один поток Genius | `python -m training.prepare_data --source genius` |
| Только кэш, без сети | `python -m training.prepare_data --source genius-cache` |
| Старый HF | `python -m training.prepare_data --source cis` |
| Свой набор slug | `python -m training.prepare_data --artists Slug1 Slug2` |

Оркестратор держит не больше пяти воркеров, каждому даёт одного артиста; воркер пишет только `genius_<slug>.json` и берёт следующего. Когда кэш всех slug полный, оркестратор сам собирает `kizaru_lyrics.txt`.

Перед сборкой текста: `strip_genius_annotations` → нормализация символов → `core/corpus/clean.py` (`LM_STREAM`: пустые строки внутри песни выкидываются, каждая строка куплета остаётся своей строкой, между песнями один `\n`). Подробнее — [corpus-cleaning.md](corpus-cleaning.md). Сырой кэш — `data/corpus/genius_<slug>.json`. Лирика с [Genius](https://genius.com) — third-party, см. [LICENSE](../LICENSE).

После завершения оркестратора на полном пуле (300 slug): `python -m training.prepare_data --source genius-cache` — пересобрать `kizaru_lyrics.txt` с актуальными правилами без сети.

### Пул Genius

Кого качать — `GENIUS_ARTISTS` в `training/prepare_data.py`: 300 slug. Файл без `"complete": true` артист не закрывает: оборванный json качается заново.

Раскладка `data/`: [../data/README.md](../data/README.md).

## Обучение

Это не RL. Next-token LM, лосс `CrossEntropy`, Adam. RL / reward / RLHF не используем. Скрипты — в `training/`.

Текущие текстовые `.pt` — после обучения на `experiments/shared/` train/test (2026-10-05). История отбора: [research/night-contest.md](research/night-contest.md).

```bash
python -m training.train --level char --out checkpoints/kizaru.pt
python -m training.train --level word --out checkpoints/kizaru_word.pt
python -m training.train_lm --arch transformer --level char --out checkpoints/transformer.pt
python -m training.train_lm --arch gpt --level char --out checkpoints/gpt.pt
python -m training.train_lm --arch deepseek --level char --out checkpoints/deepseek.pt
python -m training.train_mood --out checkpoints/mood.pt
```

Два способа текста на LSTM: `--level char` (следующий символ) и `--level word` (следующее слово после токенизации). Один и тот же сырой корпус; word-сплит в `core/text/tokenize.py` (`min-freq` по умолчанию 3). Тем же `--level word` можно гонять transformer/gpt/deepseek.

### Флаги (дефолты)

**LSTM** (`training.train`): `256/256×2`; char `--epochs 12000 --bptt 64`; word seq 64 / bptt 32. Ещё: `--corpus`, `--out`, `--batch-size`, `--hidden-size`, `--embedding-size`, `--n-layers`, `--min-freq`, `--whitelist`, `--init-navec`, `--navec-tokenizer`, `--navec-weights`, `--lr`, `--eval-every`, `--gen-len`, `--temp`, `--start`, `--seed`.

**Transformer / GPT / DeepSeek** (`training.train_lm`): char `d_model=128`, `--block-size 256`, `--steps 25000`. Ещё: `--arch`, `--level`, `--corpus`, `--out`, `--batch-size`, `--n-layers`, `--n-heads`, `--min-freq`, `--lr`, `--eval-every`, `--gen-len`, `--temp`, `--start`, `--seed`. Для `--arch deepseek`: `--kv-lora-rank`, `--q-lora-rank` (дефолт `d_model//4`).

Пока пишутся `.pt`, не поднимай текстовые воркеры на той же GPU.

### Генерация через очередь

```bash
python -m app.cli.generate --arch lstm --start "Я" --db data/bot.db
python -m app.cli.generate --arch lstm-word --start "Я"
python -m app.cli.generate --arch transformer --start "Я"
python -m app.cli.generate --arch gpt --start "Я"
python -m app.cli.generate --arch deepseek --start "Я"
```

`--arch`: `lstm` | `lstm-word` | `transformer` | `gpt` | `deepseek`.

## Бот (процессы)

```bash
set DISCORD_TOKEN=...
python -m app.orchestrator --db data/bot.db
python -m app.workers.lstm checkpoints/kizaru.pt --db data/bot.db
python -m app.workers.lstm_word checkpoints/kizaru_word.pt --db data/bot.db
python -m app.workers.transformer checkpoints/transformer.pt --db data/bot.db
python -m app.workers.gpt checkpoints/gpt.pt --db data/bot.db
python -m app.workers.deepseek checkpoints/deepseek.pt --db data/bot.db
python -m app.workers.bert_gen checkpoints/bert_gen.pt --db data/bot.db
python -m app.workers.mood checkpoints/mood.pt --db data/bot.db
python -m app.workers.still --db data/bot.db
python -m app.workers.video --db data/bot.db
python -m app.discord --db data/bot.db
```

**Админ-панель:** `python -m app.web` — http://127.0.0.1:8765/ (БД `data/bot.db`).

Токен только из `DISCORD_TOKEN`. В файлы репозитория и в логи его не клади.

Инвайт нужен со scope **`bot`** и **`applications.commands`** (иначе slash не появятся). Повторный инвайт, если бота добавляли только с `bot`:

https://discord.com/oauth2/authorize?client_id=1556306587511951380&permissions=380704&scope=bot%20applications.commands

Префикса нет, только slash. Бот синкает команды на свои гильдии и глобальный sync не делает. В чате: `/ai`.

### Slash `/ai`

| Команда | Параметры | Действие |
|---------|-----------|----------|
| `/ai text` | `model`: **lstm** \| **lstm-word** \| **transformer** \| **gpt**; `start` (по умолчанию «Я») | Продолжение лирики. lstm — символы, lstm-word — слова. Если воркер не запущен, бот отвечает сразу. |
| `/ai mood` | `text` (опционально) | Две оси: добрый/злой и веселый/грустный. Пустой `text` берётся из reply; иначе бот просит вставить текст. |
| `/ai pic` | `mode`: **line** \| **simple** \| **cli**; `file`; у **cli** ещё `scale`, `cells` | **line** и **simple** — текст в чат, 31 колонка. **cli** — файл: PNG для картинки, GIF для gif/видео. `scale` по умолчанию 2, `cells` 320. |
| `/ai help` | — | Та же справка в чате. |

## Pic CLI

`python -m app.cli.image` только ставит заявку (orchestrator и `app.workers.still` или `app.workers.video` уже должны работать). Файл ищется по указанному пути, иначе в `data/import/`.

Флаги: `--db`, `--mode line|simple`, `--cells`, `--scale`, `--out`. Флагов `--audio`, `--mp4`, `--gif`, `--upscale` и `--line-*` нет.

```bash
python -m app.cli.image still couple2.jpg --mode line
python -m app.cli.image still couple2.jpg --mode simple
python -m app.cli.image anim bad_apple.mp4 --mode line
```

| Команда | Очередь | Результат |
|---------|---------|-----------|
| `still --mode line` | `pic_still` | PNG. По умолчанию `data/export/<stem>/pik/line_s<scale>.png`. `cells` и `scale` уходят в заявку (320 и 2) |
| `still --mode simple` | `pic_still` | Текстовая сетка в stdout. `cells` и `scale` в заявку не попадают |
| `anim --mode line` | `pic_video` | GIF. По умолчанию `data/export/<stem>/pik/line_s<scale>.gif`. `--out` не с суффиксом `.gif` переписывается в `.gif` |
| `anim --mode simple` | — | Код выхода 1, anim только line |

Для GIF нужны `ffmpeg` и `ffprobe`. PNG и GIF воркер рисует с upscale в коде (флага `--upscale` нет): ширина файла = `cells * 8` (320 -> 2560). `--scale` задаёт, сколько пикселей источника приходится на квадрант smile, и при этом upscale размер кадра не меняет. На CUDA кадры анимации идут батчем 4 (`core/image/render.py`, `GPU_BATCH`); на CPU — по одному.

В Discord `line` и `simple` — текст на 31 колонку, не PNG. Файл даёт только `cli`: PNG для картинки, GIF для gif или видео. Тот же путь, что CLI `--mode line`.

## Тесты

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m app.cli.image still --help
python -m app.cli.image anim --help
```
