# Лицензии — что можно публиковать

## Код проекта (MIT)

Код приложения, конфиги и доки в этом репозитории — [MIT](../LICENSE) (Copyright 2026 KizaruAi), если файл не говорит иное.

Можно публиковать, форкать и распространять **код** по условиям MIT (сохрани copyright notice).

## Только локально — не выкладывать

| Материал | Где | Почему |
|----------|----------|-----|
| Лирика Genius / артистов | `data/corpus/` (`genius_*.json`, `kizaru_lyrics.txt`, legacy HF) | Чужой копирайт; ToS [Genius](https://genius.com); проекту не принадлежит |
| Fallback-экспорт чата | `data/corpus/messages_fallback.csv` | Чужой текст |
| Замороженные train/test | `experiments/shared/train.txt`, `test.txt` | Производные от лирики; пересобери локально после корпуса |
| Рантайм DB / очереди | `data/bot.db` (+ WAL/SHM) | Локальное состояние, может быть user content |
| Pic in/out | `data/import/`, `data/export/` | Пользовательский/медиа контент |
| Логи / temp | `logs/`, `data/logs/`, `var/` (логи, статусы прогонов) | Операционный шум, возможны секреты |
| Секреты | `.env` | Токены (`DISCORD_TOKEN`, `GENIUS_ACCESS_TOKEN`, …) |
| Стиховые корпуса | `data/corpus/poetry/` | Чужие датасеты со своими лицензиями (RIFMA — MIT, ArsPoetica — CC-BY-NC-4.0, PoetryCorpus — Apache-2.0, stihi_ru — см. карточку HF); см. [research/verse-schema-datasets.md](research/verse-schema-datasets.md) |
| Датасеты JEF | `data/rlaif/` | Производные от стихов и лирики |

См. также раздел **Third-party content** в [LICENSE](../LICENSE).

## Можно публиковать (если есть)

| Материал | Где | Лицензия / заметки |
|----------|----------|-----------------|
| Чекпоинты | `checkpoints/**/*.pt` | Веса на локальном корпусе — **ты** отвечаешь, чтобы в `.pt` не уехала лирика текстом; сами веса — артефакт проекта |
| Пак Navec | `data/embeddings/navec_hudlit/` (без `.cache/`) | **MIT** — `data/embeddings/navec_hudlit/LICENSE.txt`; апстрим [BorisTM/…](https://huggingface.co/BorisTM/natasha_navec_hudlit_v1_12B_500K_300d_100q) |
| Метаданные | `experiments/shared/README.md`, `split_ids.json`, `seeds.json` | Короткие сиды в `seeds.json` — фикстуры проекта; полные lyric-файлы остаются локально |

## Genius API / скрейп

Загрузка лирики — для **локального ресёрча и трейна** на своей машине. Репо не даёт прав перепубликовать тексты на Hugging Face, Kaggle или в публичных датасетах. `training.prepare_data` / оркестратор — только с токенами и лимитами, которые тебе разрешены.

## Vendor в админке

| Файл | Версия | Лицензия | SHA-256 |
|------|--------|----------|---------|
| `app/web/static/vendor/d3.v7.min.js` | D3 7.9.0 | ISC (Mike Bostock / Observable) | `f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539` |

Источник: https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js (в репозитории лежит локальная копия; CDN в рантайме не используется).

## Коротко

- **Публикуй:** MIT-исходники, доки, опционально `.pt`, файлы Navec под `data/embeddings/` (без HF `.cache/`), vendor D3 под ISC.
- **Оставляй локально:** корпус, сплиты из лирики, DB, import/export, логи, `.env`.
