# Раскладка `data/`

| Путь | Назначение |
|------|------------|
| `import/` | Картинки и видео для pic CLI (gitignore) |
| `export/<stem>/pik/` | PNG от `still --mode line` и GIF от `anim --mode line` (gitignore). `still --mode simple` файл не пишет, сетка идёт в stdout |
| `corpus/` | **gitignore:** `kizaru_lyrics.txt`, `genius_*.json`, legacy HF, `messages_fallback.csv` — создаётся после `prepare_data` / orchestrator |
| `dataset/` | `lyrics.jsonl` и сплиты от `pack_lyrics_dataset.py` (gitignore, кроме `dataset/README.md`, если он есть) |
| `embeddings/navec_hudlit/` | Natasha Navec (HF) word vocab + 300d weights for Word LSTM whitelist / init; see [docs/research/embeddings-vocab.md](../docs/research/embeddings-vocab.md) |
| `logs/` | Логи обучения (gitignore) |
| `bot.db` | SQLite-шина бота (gitignore); см. [database/README.md](../database/README.md) |

Логи процессов бота удобнее класть в [var/logs/](../var/README.md), не в corpus.

В git: этот файл, `data/.gitkeep`, `data/embeddings/` (без HF `.cache/`), при необходимости `checkpoints/*.pt`. Корпус, import/export, `bot.db` — только локально.

### Корпус (Genius)

Пул загрузки — `GENIUS_ARTISTS` в `prepare_data.py` (98 slug). Параллельно: `python genius_orchestrator.py --workers 5` (максимум пять воркеров, один артист на воркер, свой `genius_<slug>.json`, корпус пишет оркестратор в конце). Один поток: `python prepare_data.py --source genius`. Пока прогон не закончился, `kizaru_lyrics.txt` ещё не этот пул. `dataset/lyrics.jsonl` сам от этого не обновляется: его собирает отдельный `pack_lyrics_dataset.py`, и файл может пока быть старой HF-сборкой. Чекпоинты на этих 98 артистах не обучены. Обучение не запускать, пока датасет не подготовлен.

Загрузка: публичный API Genius и HTML страниц песен (около 1 запроса в секунду). При `GENIUS_ACCESS_TOKEN` и установленном `lyricsgenius` поиск идёт через токен, текст всё равно со страниц. Пропуски (404, артист не найден) пишутся в stderr.

Фильтр: `strip_genius_annotations` → символы → `core/corpus/clean.py` (`LM_STREAM`); см. [../docs/corpus-cleaning.md](../docs/corpus-cleaning.md).

Кэш с песнями — файлы `genius_<Slug>.json`. Имена slug смотри в `GENIUS_ARTISTS`, не в чекпоинтах.

Лирика не под MIT — см. [LICENSE](../LICENSE). На HF и Kaggle её не выкладывать.

Словарь line-режима: `kizaru/assets/line_patterns.json`.

### Pic (дефолты CLI)

Pic — символьная сетка, не пиксельный фильтр. CLI только ставит заявку.

- `still --mode line` → PNG в `export/<stem>/pik/line_s<scale>.png`. Воркер рисует с upscale в коде (флага `--upscale` нет). Ширина = `cells * 8`. Дефолт `--cells 320` даёт 2560 px. `--scale` (дефолт 2) — сколько px источника на квадрант smile.
- `still --mode simple` → текст в stdout. `--cells` и `--scale` на эту заявку не влияют.
- `anim --mode line` → GIF в `export/<stem>/pik/`. `anim --mode simple` завершается с кодом 1.
- Отдельных флагов `--audio`, `--mp4`, `--gif` нет.

Пайплайн бота и размеры — в [docs/runbook.md](../docs/runbook.md); обзор — [README.md](../README.md).
