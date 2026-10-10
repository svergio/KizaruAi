# Раскладка `data/`

| Путь | Назначение |
|------|------------|
| `import/` | Картинки и видео для pic CLI (gitignore) |
| `export/<stem>/pik/` | PNG от `still --mode line\|render\|details` и GIF от `anim` (gitignore). `still --mode simple` пишет файл только с `--out`, иначе сетка идёт в stdout. Остальные папки в `export/` — выгрузки исследовательских скриптов из `notebooks/research/` |
| `corpus/` | **gitignore:** `kizaru_lyrics.txt`, `genius_*.json`, legacy HF, `messages_fallback.csv` — создаётся после `prepare_data` / orchestrator |
| `dataset/` | `lyrics.jsonl` и сплиты от `python -m training.pack_lyrics_dataset` (gitignore, кроме `dataset/README.md`) |
| `corpus/poetry/` | стихи и стиховые датасеты со схемой рифмы (`download_poetry`, `import_verse_schema`); см. [docs/training.md](../docs/training.md) |
| `rlaif/jef/` | датасет классификатора рифмы JEF (`build_jef_dataset`) |
| `embeddings/navec_hudlit/` | Natasha Navec (HF): word-словарь + веса 300d для whitelist / инициализации; см. [docs/research/embeddings-vocab.md](../docs/research/embeddings-vocab.md) |
| `logs/` | Логи обучения (gitignore) |
| `bot.db` | SQLite-шина бота (gitignore); см. [database/README.md](../database/README.md) |

Логи процессов бота удобнее класть в [var/logs/](../var/README.md), не в corpus.

В git: этот файл, `data/.gitkeep`, `data/embeddings/` (без HF `.cache/`), при необходимости `checkpoints/*.pt`. Корпус, import/export, `bot.db` — только локально.

### Корпус (Genius)

Пул загрузки — файлы `training/artist_pools/*.txt` (если папки нет — встроенный список в `training/prepare_data.py`). Параллельно: `python -m training.genius_orchestrator --workers 5`. Один поток: `python -m training.prepare_data --source genius`.

Актуально после сборки **2026-10-05**: `data/corpus/kizaru_lyrics.txt` (~70M символов), фильтр `LM_STREAM` (пустых строк нет). Shared splits: `experiments/shared/train.txt` / `test.txt`. Текстовые чекпоинты на этом корпусе уже переобучены (word vocab 99868).

Загрузка: публичный API Genius и HTML страниц песен (около 1 запроса в секунду). При `GENIUS_ACCESS_TOKEN` и установленном `lyricsgenius` поиск идёт через токен, текст всё равно со страниц. Пропуски (404, артист не найден) пишутся в stderr.

Фильтр: `strip_genius_annotations` → символы → `core/corpus/clean.py` (`LM_STREAM`); см. [../docs/corpus-cleaning.md](../docs/corpus-cleaning.md).

Кэш с песнями — файлы `genius_<Slug>.json`. Имена slug смотри в `training/artist_pools/*.txt`.

Лирика не под MIT — см. [LICENSE](../LICENSE). На HF и Kaggle её не выкладывать.

Словарь line-режима: `core/image/assets/line_patterns.json`.

### Pic (дефолты CLI)

Pic — символьная сетка, не пиксельный фильтр. CLI только ставит заявку.

- `still --mode line` → PNG в `export/<stem>/pik/line_s<scale>.png`. Upscale включён по умолчанию, выключается `--no-upscale`. Ширина = `cells * 8`. Дефолт `--cells 320` даёт 2560 px. `--scale` (дефолт 2) — сколько px источника на квадрант smile.
- `still --mode simple` → текст в stdout (с `--out` — PNG).
- `anim --mode line|render|details` → GIF в `export/<stem>/pik/`. `anim --mode simple` завершается с кодом 1.
- Отдельных флагов `--audio`, `--mp4`, `--gif` нет.

Пайплайн бота и размеры — в [docs/runbook.md](../docs/runbook.md); обзор — [README.md](../README.md).
