# Обучение

Все скрипты обучения и подготовки данных лежат в `training/`. Запуск везде из корня проекта через `python -m ...`. Требования к стенду (GPU, LLM-сервер, MLflow, доп. пакеты) — в [install.md](install.md#требования-к-стенду).

План переобучения (микс → SFT → JEF → RL) с критериями приёмки — [training-plan.md](training-plan.md).

Два уровня:

- **Основные слоты бота** (`lstm`, `lstm-word`, `transformer`, `gpt`, `deepseek`, `bert-gen`, `mood`) — обычные next-token LM (у `bert-gen` — MLM), `CrossEntropy` + Adam. Скрипты `train.py`, `train_lm.py`, `train_mood.py`.
- **Экспериментальный пайплайн JEF → RL** — классификатор рифмы на Qwen (LoRA) и RL-дообучение MiniDeepSeek с этим классификатором как наградой. Скрипты `scripts/build_jef_dataset.py`, `train_jef.py`, `train_rl_deepseek.py`. Описание идеи — [research/jef-rhyme-classifier.md](research/jef-rhyme-classifier.md).

## Данные

| Скрипт | Что делает | Запуск |
|--------|------------|--------|
| `training/prepare_data.py` | Собирает корпус `data/corpus/kizaru_lyrics.txt`. `--source`: `genius` (качать Genius), `genius-cache` (пересобрать из локального кэша без сети), `cis` (старый HF), `kizaru`, `fallback`, `auto` (по умолчанию: кэш Genius, иначе fallback). Ещё `--artists`, `--refresh`, `--out`. | `python -m training.prepare_data --source genius-cache` |
| `training/genius_orchestrator.py` | Параллельная загрузка Genius по пулу артистов (до 25 воркеров, адаптивно; минимум 2), потом склейка корпуса. Флаги `--workers`, `--pools`, `--download-only`, `--partial`, `--cache`, `--out`. При HTTP 429 ставь `--workers 2` или `--workers 3`, иначе банят IP. | `python -m training.genius_orchestrator --workers 3` |
| `training/prepare_subset.py` | Корпус из части пулов (`training/artist_pools_resume/`) + локальные стихи → `data/corpus/kizaru_lyrics_ready.txt`. `--all-caches` берёт все полные `genius_*.json` с дедупом по slug; `--poetry` / `--no-poetry`. | `python -m training.prepare_subset --all-caches` |
| `training/pack_lyrics_dataset.py` | Упаковка кэшей из `data/corpus/` в `data/dataset/lyrics.jsonl` + сплиты (98/2, seed 1). Сеть не нужна. | `python -m training.pack_lyrics_dataset` |
| `training/prep_embedding_vocab.py` | Whitelist слов из Navec для word-моделей (`seed_whitelist.txt` / `seed_missing.txt`). | см. [research/embeddings-vocab.md](research/embeddings-vocab.md) |
| `training/scripts/download_poetry.py` | Качает классические русские стихи (PoetryCorpus `all.xml`, сверка sha256) → `data/corpus/poetry/classical_poetry.txt`; если установлен `datasets`, добавляет выборку `IlyaGusev/stihi_ru` (до 50 000 стихов) → `stihi_ru_sample.txt`. Нужен интернет. | `python -m training.scripts.download_poetry` |
| `training/scripts/import_verse_schema.py` | Переводит стиховые корпуса с разметкой рифмы (RIFMA, ArsPoetica, англ. Chicago rhymedata, poetry_form) в JSONL со схемой куплета `data/corpus/poetry/schema/verse_schema.jsonl` и плоский текст `verse_schema_lm.txt`. `--skip-ars-nc` — без NC-лицензии, `--skip-en` — без английского. | `python -m training.scripts.import_verse_schema` |
| `training/scripts/build_jef_dataset.py` | Датасет для классификатора рифмы JEF: «золото» RIFMA/Chicago + негативы (перемешанные строки) + ArsPoetica/Genius без разметки → `data/rlaif/jef/{train,valid,test}.jsonl`, заметки в `var/jef_data_notes.md`. `--with-base-jeff` дополнительно размечает схемы базовой Qwen (`--jeff-model`, по умолчанию `Qwen/Qwen3-4B-Instruct-2507`) — нужны GPU и `transformers`. | `python -m training.scripts.build_jef_dataset` |

### Проверка корпуса

`training/check_dataset.py` — проверки корпуса в одном CLI. Ничего не меняет в данных: отчёт печатается в stdout, с `--out` дополнительно пишется в markdown (с `--dry-run` файл не пишется).

| Проверка | Что считает | Источник |
|----------|-------------|----------|
| `noise` | Шум Genius (теги секций, кредиты, Contributors, UI-строки, утечки JSON, любые `[..]`) в песнях до и после очистки `prepare_corpus_text` из `core/corpus/` | `genius_*.json` в `--corpus-dir`, случайная выборка `--sample` файлов (по умолчанию 25, `0` — все) |
| `brackets` | Оставшиеся `[теги]`, строки с JSON и строки длиннее 300 символов, топ тегов | готовый корпус `--lyrics` |
| `rhyme` | Совпадение рифменной зоны у соседних строк (сдвиги +1…+4) против случайного порядка, доля рифмы в следующих трёх строках | `--lyrics` |
| `dupes` | Дубли кэша `genius_X-.json` / `genius_X.json` с одинаковым нормализованным slug (как в `prepare_subset`), какой файл оставить; файлы только с дефисом. Ничего не удаляет | `--corpus-dir` |
| `all` | Все четыре | — |

Флаги: `--corpus-dir` (по умолчанию `data/corpus`), `--lyrics` (по умолчанию `data/corpus/kizaru_lyrics.txt`), `--sample`, `--seed` (1), `--out`, `--dry-run`. Если файла `--lyrics` нет, `brackets`/`rhyme` завершаются с кодом 2.

```bash
python -m training.check_dataset all
python -m training.check_dataset noise --sample 25 --seed 1
python -m training.check_dataset dupes --corpus-dir data/corpus
python -m training.check_dataset brackets rhyme --out var/check_dataset.md
```

Новые проверки корпуса добавляются сюда, а не отдельными скриптами.

### Микс для обучения (`build_mix`)

`training/build_mix.py` собирает обучающий микс из сырых источников по рецепту `training/mixes/<name>.yaml`. План, где он используется, — [training-plan.md](training-plan.md).

```bash
python -m training.build_mix --mix ru_rap_v1 --dry-run   # только таблица пулов и план, ничего не пишет
python -m training.build_mix --mix ru_rap_v1             # -> data/mixes/ru_rap_v1/
```

Флаги: `--mix` (имя рецепта или путь к yaml), `--seed` (перекрывает seed рецепта), `--out` (по умолчанию `data/mixes/<name>`), `--dry-run`, `--allow-short`, `--corpus-dir`, `--poetry-dir`, `--stihi full|sample` (по умолчанию `full`), `--rebuild-cache`, `--workers N`.

Кэши Genius с `"complete": false` (их ещё пишет работающий загрузчик) и битые/пустые JSON (`genius_*`, `huggingartists_*`) пропускаются с `WARN`; счётчики `skipped_incomplete` / `skipped_broken` печатаются в выводе (и в `--dry-run`) и пишутся в `report.json`. Старые кэши без поля `complete` считаются полными.

Источники: `data/corpus/genius_*.json`, `data/corpus/huggingartists_*.json`, `data/corpus/poetry/classical_poetry.txt` (кусками по 16 строк), полный `data/corpus/poetry/raw/stihi_ru.jsonl.zst`, английские `poetry/schema/en/poetry_form` и `rhymedata/english_raw`. Текст чистится той же `prepare_corpus_text`, что и в `pack.py`.

**stihi_ru целиком.** Архив читается потоково (`zstandard`, в память целиком не грузится): каждый стих чистится и режется на куски по 16 строк, куски короче 40 символов отбрасываются. Разбор параллельный (`--workers`, по умолчанию `min(8, CPU-1)`), результат кешируется в `data/corpus/poetry/cache/stihi_ru_chunks16_v1.jsonl.zst` + `.meta.json`. Кеш пересобирается сам, если поменялся архив (размер/mtime), длина куска или версия кеша; принудительно — `--rebuild-cache`. Кеш пишется и при `--dry-run` (это не микс): `--dry-run --stihi full` (full — по умолчанию) при отсутствии или устаревании кеша сначала собирает его — первый раз ≈4,5 мин разбора на 12 процессах и ~2,2 ГБ в `data/corpus/poetry/cache/`; с готовым кешем dry-run идёт ≈4 мин. Без сборки кеша — `--stihi sample`. Дальше куски проходят те же фильтры письма и общий дедуп; сплит — по id стиха (все куски стиха в одном сплите). В памяти держатся только числа по кускам, тексты выбранных кусков подтягиваются из кеша при записи. Если архива или пакета `zstandard` нет — с предупреждением берётся `stihi_ru_sample.txt`; явно — `--stihi sample`. Артистов в `ru_poetry` считаем по авторам stihi_ru; правило «>70% латиницы → англ. пул» к stihi_ru не применяется (только пословные фильтры песни).

Что делает по шагам:

1. **Пул песни.** По спискам артистов: `artist_pools/00_existing.txt`, `ru_expand.txt`, `artist_pools_solo_retry/06_solo_retry.txt`, `ru_rap_extra.txt` (~220 mid/underground Genius slug) и все `huggingartists` → `ru_rap`; `ru_pop_rock_chanson.txt` → `ru_other` (жанры там не размечены, поэтому на pop/rock не делится; при пересечении рэп-списки важнее — в нём есть хвосты дублей вроде `Oxxxymiron-2`). Ключ артиста — `pool_key` из `training/pools.py` (кириллица сохраняется, снимается только суффикс дубля `-2`; пустой ключ не мапится); `artist_pools_poets/poets.txt` и стихи → `ru_poetry`; `en_rap.txt` → `en_rap`; англ. стихи → `en_poetry`. Артисты из рэп-списков, которые на деле не русский рэп, перечислены в `training/artist_pools/ru_rap_exclude.txt` (`slug<TAB>pool`, `#` — комментарий). Они не выкидываются, а уходят в указанный пул (`ru_pop`, `ru_rock`, `uk_other`, `kz_uz_other`…); такие пулы создаются по ходу и их можно ставить в рецепт. Читается через `load_ru_rap_exclude_pools()` из `training/pools.py`; строка без пула → `ru_other`; пустой или отсутствующий файл — предупреждение и сборка без исключений. Ключ артиста — `pool_key` оттуда же: снимает только суффикс дубля `-2`/`-3`, `25-17` не трогает. Артист не из списков: кириллица → `ru_other`, латиница → `other`.
2. **Письмо** (`line_script` из `check_dataset`): у артиста >70% латинских строк → англ. пул (`ru_rap` → `en_rap`, `ru_poetry` → `en_poetry`, `ru_other` → `other`). Песни с арабским письмом выкидываются; из ru-пулов — песни с >50% латинских строк (одиночные английские строки остаются); из en-пулов — песни с >50% кириллицы. Короче 40 символов — тоже вон.
3. **Дедупликация** по `text_hash` из `check_dataset` (нормализованный текст) по всем пулам сразу; остаётся первая копия (Genius раньше huggingartists).
4. **Сплит** train/val/test внутри пула, до смешивания, по нормализованному хешу текста (`text_hash` после удаления `[..]` и `(..)` — теги, `(Live)`, `(Remix)`, `(Acoustic)`, эдлибы, в том числе отдельной строкой; без регистра, пунктуации и пробелов). От seed не зависит; одинаковый текст (фит, live, ремикс) всегда в одном сплите. Куски stihi_ru делятся по id стиха.
5. **Выборка** по долям токенов из рецепта. Данные не повторяются: если пулу не хватает, без `--allow-short` сборка падает с кодом 3, с ним — берёт всё, что есть, и предупреждает. `total_tokens: max` — максимум, который собирается в заданных долях без повторов.

Токены считаются приблизительно (слова + пунктуация + переносы строк), как у словной модели.

Рецепты в `training/mixes/`:

```yaml
# ru_rap_v1 / ru_v2_finetune
name: ru_rap_v1
seed: 1
total_tokens: max
splits: {val: 0.02, test: 0.02}
pools: {ru_rap: 85, ru_poetry: 10, en_rap: 5}

# ru_v2_pretrain — весь ru_rap + поэзия/other/pop/rock, en_rap ≤15%
name: ru_v2_pretrain
fill_all: [ru_rap]
max_pool_share: {en_rap: 0.15}
pools: {ru_rap: 40, ru_poetry: 35, en_rap: 15, ru_other: 4, ru_pop: 3, ru_rock: 3}
```

Сплит по хешу нормализованного текста общий для всех миксов: песня из val/test одного микса не попадает в train другого.

Выход (`stihi_ru` в поле `source`, если кусок из полного архива): `{train,val,test}.jsonl` (id, pool, split, hash, source, artist, title, tokens, text), `{train,val,test}.txt` (для `train_lm --corpus`), `{split}_{pool}.txt` (например `val_ru_rap.txt` для `--val-focus`) и `report.json`.

Пул артистов Genius — файлы `training/artist_pools/*.txt` (по одному slug на строку, `#` — комментарий). Если папки нет, берётся встроенный список в `prepare_data.py`. Очистка корпуса — [corpus-cleaning.md](corpus-cleaning.md).

## Обучение моделей

| Скрипт | Что учит | Пример |
|--------|----------|--------|
| `training/train.py` | LSTM (`TextRNN`): `--level char` или `--level word`. Дефолт 256/256×2, `--epochs 12000`. Флаги Navec: `--whitelist`, `--init-navec`, `--navec-tokenizer`, `--navec-weights`. | `python -m training.train --level char --out checkpoints/kizaru.pt` |
| `training/train_lm.py` | Transformer / GPT / DeepSeek / BERT-gen. Уровни: `char` / `word` / `bpe`. Val: `--val-corpus` / `--val-focus` (дефолт `val.txt` + `val_ru_rap.txt`), на `--eval-every` — `val_focus`, `val_all`, `bpc`, `val_unk_rate`. Лучший по `--best-metric` (`val_focus`\|`val_all`) → `<out>_best.pt`, последний → `<out>_last.pt` (финал best не затирает); `--patience N`. Рядом с каждым `_best`/`_last` пишется сайдкар `<name>.pt.meta.json` (имя, токенизатор, vocab_size, step, val_ru_rap, bpc, date) для админки без `torch.load`. BPE: `--vocab-size 32000` (учится только на train), сохраняется как `<stem>.tokenizer.json` рядом с чекпоинтом; `--vocab-from` / `--init-from`; tie weights; `--init-navec` для BPE запрещён. Word+Navec как раньше. | см. двухэтапную схему ниже |
| `training/ckpt_meta.py` | Постоянный CLI для существующих чекпоинтов: читает `.pt` (`weights_only=True`, при необходимости `add_safe_globals`; `weights_only=False` только здесь для локальных файлов) и пишет `<name>.pt.meta.json`. | `python -m training.ckpt_meta --all checkpoints/` |

### Двухэтапная схема GPT v2 (BPE)

1. Миксы: `ru_v2_pretrain`, `ru_v2_finetune` (`python -m training.build_mix --mix … --dry-run` → потом без dry-run).
2. Претрейн (1–2 эпохи, patience по `val_all`): `--out checkpoints/gpt_ru_v2_pre.pt` → `gpt_ru_v2_pre_best.pt`.
3. Дообучение с `--init-from …_pre_best.pt`, меньший `--lr`, `--best-metric val_focus`, `--val-focus …/val_ru_rap.txt` → `gpt_ru_v2_ft_best.pt`.
4. Сравнение с prod (`gpt_ru_rap_v1-prod.pt`): bpc на одном val ru_rap + 20 сэмплов с фиксированным сидом. Prod менять только отдельно.

MLflow-метрики: `loss`, `val_focus`, `val_all`, `bpc`, `val_unk_rate`, `best_val_focus` / `best_val_all`.
| `training/train_mood.py` | RNN настроения (две оси) → `checkpoints/mood.pt`. `--epochs 300`, `--seed 2`. | `python -m training.train_mood` |
| `training/train_jef.py` | LoRA-классификатор рифмы (JEF) на Qwen (`--model`, по умолчанию `Qwen/Qwen3-4B-Instruct-2507`), 4-bit по умолчанию (`--no-4bit` — выключить). Данные `data/rlaif/jef/`, выход `checkpoints/jef_rhyme_lora/`. Гейт на тесте: F1 ≥ 0.80, ECE ≤ 0.10 (`--gate-f1`, `--gate-ece`). Ещё `--eval-only`, `--resume`, `--gpu-max-mem`, `--cpu-max-mem`. | `python -m training.train_jef` |
| `training/train_rl_deepseek.py` | RL-дообучение MiniDeepSeek: политика из SFT-чекпоинта (`--policy`, по умолчанию `checkpoints/deepseek_v2.pt`), награда — вероятность рифмы от JEF (`--jeff checkpoints/jef_rhyme_lora`), алгоритм `--algo grpo` (или `reinforce`) с KL к замороженной SFT-копии. Выход `checkpoints/deepseek_rl_v1.pt`, статус `var/deepseek_rl_status.json`. | `python -m training.train_rl_deepseek --steps 400` |
| `training/mlflow_track.py` | Не скрипт, а обёртка над MLflow для всех `train*.py` (см. ниже). | — |

Пока идёт обучение, не поднимай текстовые воркеры на той же GPU: один процесс обучения на GPU за раз.

Готовые сплиты для честного сравнения моделей: `experiments/shared/` (см. `experiments/shared/README.md`), held-out CE: `python experiments/bench.py <checkpoint.pt>`.

## MLflow (необязательно)

MLflow нужен только для графиков метрик обучения. Сервер в проект не входит и поднимается отдельно (по умолчанию ожидается на `http://127.0.0.1:5000`). Без него обучение работает.

| Что | Как |
|-----|-----|
| Адрес | `MLFLOW_TRACKING_URI`, по умолчанию `http://127.0.0.1:5000` |
| Выключить | `KIZARU_MLFLOW=0` (или `train_lm --no-mlflow`) |
| Продолжить существующий ран | `MLFLOW_RUN_ID=<id>` |
| Эксперименты | `kizaru-lm` (`train_lm`), `kizaru-legacy` (`train`, `train_mood`), `kizaru-jef` (`train_jef`), `kizaru-rl` (`train_rl_deepseek`) |

`mlflow_track.py` ловит все ошибки MLflow и печатает `mlflow_warn ...`, обучение при этом не падает. Если сервер не запущен, клиент MLflow может долго ретраить подключение — тогда проще выставить `KIZARU_MLFLOW=0`.

## Ops-скрипты в `var/`

Служебные скрипты под конкретные ночные прогоны. Писались под конкретные прогоны. Python берут из `sys.executable` / PATH, корень проекта — от расположения скрипта. PID процесса SFT задаётся через `KIZARU_SFT_PID` (`pipeline_monitor.py`, `deepseek_rl_waiter.py`), ID MLflow-рана SFT для `deepseek_cutover_15k_rl.py` — через `KIZARU_SFT_MLFLOW_RUN`: переменная обязательна, значения по умолчанию нет. Без неё скрипт сразу завершается с кодом 2 и понятной ошибкой в stderr, ничего не запуская.

| Скрипт | Что делает |
|--------|------------|
| `var/night_train_chain.ps1` | Ночная цепочка: char LSTM → `checkpoints/kizaru_v2.pt`, затем word GPT → `checkpoints/gpt_v2.pt`; лог `var/night_train_chain.log`. |
| `var/night_train_babysit.py` | Нянька цепочки: следит за логом, процессами и `nvidia-smi`, при зависании или OOM убивает и перезапускает обучение, пишет утренний отчёт `var/night_train_morning.md`. |
| `var/night_train_babysit_light.py` | Облегчённая нянька: если фаза 2 (GPT) умерла, перезапускает `train_lm`. |
| `var/night_train_monitor.py` | Разовый статус цепочки: loss из лога, процессы, GPU, мета чекпоинтов. |
| `var/deepseek_cutover_mlflow.py` | Ждёт, пока SFT DeepSeek дойдёт до шага `KIZARU_CUTOVER_STEP` (5500), останавливает его, сохраняет снапшот `checkpoints/deepseek_v2_step{N}.pt`, дозаливает loss в MLflow и перезапускает `train_lm` уже с MLflow. |
| `var/mlflow_backfill_deepseek_v2.py` | Разово создаёт ран в `kizaru-lm` и заливает в него loss из лога обучения DeepSeek (`--cutover-step` обязателен). |
| `var/deepseek_cutover_15k_rl.py` | Ждёт шаг 15000 у SFT DeepSeek, останавливает только `train_lm`, делает снапшот, закрывает MLflow-ран и запускает RL. |
| `var/deepseek_rl_waiter.py` | Отсоединённый «ждун»: когда SFT DeepSeek закончится и освободит GPU, запускает `training.train_rl_deepseek`. |
| `var/pipeline_monitor.py` | Следит за пайплайном SFT → RL: маркер `var/deepseek_sft_done.json`, затем чекпоинт `checkpoints/deepseek_rl_v1.pt`; лимит 12 ч, лог `var/pipeline_monitor.log`. |

Логи и JSON-статусы этих прогонов (`var/*.log`, `var/*_status.json`, `var/*_pid.txt`) — артефакты конкретных запусков, не конфигурация.

## RLAIF

Цикл с LLM-учителем/судьёй пока только подготовлен на бумаге, запускать нечем: [research/rlaif-song-loop.md](research/rlaif-song-loop.md).
