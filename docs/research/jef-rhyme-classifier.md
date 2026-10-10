# JEF — классификатор рифмы на Qwen (а не генеративный судья)

Чуть не пошли неправильным путём: большая LLM как «судья» для предпочтений в RL.
Правильный путь — **классификатор**: на входе текст куплета, на выходе — есть ли рифма и какая схема.

**JEF** — дообученная LoRA на Qwen с классификационной головой для рифмы. Используется как награда для RL MiniDeepSeek (`training/train_rl_deepseek.py`).

## Задача

| Режим | Метки | Зачем |
|-------|-------|-------|
| binary | `rhyme` / `no_rhyme` | жёсткий гейт награды / KPI LoRA |
| scheme (мягко) | `ABAB`, `ABBA`, `AABB`, `A-A`, `other`, `free` | вспомогательная разметка от **базовой** Qwen zero-shot, не единственный KPI |

Вход — тело строфы (2–16 строк), в том же формате, что выдаёт политика.

## Разметка

1. **Золото** RIFMA + Chicago → жёсткие `label_bin` + `label_scheme` (`soft=false`).
2. **Негативы** — перемешанные строки рифмованного золота → `no_rhyme` / `free`.
3. **ArsPoetica** (NC) + куски Genius со сленгом → мягкая `label_scheme` от базовой Qwen, если включено (`soft=true`); иначе строки идут без схемы с честным `soft=false`.
4. Сплиты `train` / `valid` / `test`, seed 8.

## Данные (локально)

- `data/corpus/poetry/schema/rifma_dataset.json` — RU, `rhyme_scheme` (MIT)
- `data/corpus/poetry/schema/en/rhymedata/english_raw` — EN Chicago (public domain)
- `data/corpus/poetry/schema/arspoetica.json` — RU, без золотой схемы (CC-BY-NC-4.0)
- `data/corpus/genius_*.json` — сленговые строки для мягкой схемы / регистра

Сборка (из корня проекта):

```bash
python -m training.scripts.build_jef_dataset
# опционально, когда свободны GPU и модель:
python -m training.scripts.build_jef_dataset --with-base-jeff
# -> data/rlaif/jef/{train,valid,test}.jsonl
# заметки: var/jef_data_notes.md
```

Строка (золото):

```json
{"text":"line1\nline2\n...","label_bin":"rhyme","label_scheme":"ABAB","lang":"ru","source":"rifma","soft":false}
```

Строка (Ars/Genius до разметки базовой моделью):

```json
{"text":"...","lang":"ru","source":"arspoetica","soft":false}
```

## Модель (влезает в 10 GB VRAM)

- База: `Qwen/Qwen3-4B-Instruct-2507` (по умолчанию в `train_jef.py`), LoRA + бинарная голова, грузится через `transformers` / `peft`.
- QLoRA 4-bit по умолчанию (`--no-4bit` выключает), короткая последовательность (`--max-len 256`).
- Мягкие схемы — от базовой модели zero-shot, а не обучение только на схемах с нуля.
- Не генеративный чат-судья.

## Обучение

1. Собрать JSONL (`build_jef_dataset`).
2. `python -m training.train_jef` → `checkpoints/jef_rhyme_lora/`.
3. Гейт на тесте: F1 ≥ 0.80 и ECE ≤ 0.10.
4. Схемы — только вспомогательная цель (`--scheme-aux-weight`).
5. Дальше SFT DeepSeek → RL, где JEF оценивает пары кандидатов (`training/train_rl_deepseek.py`).

Флаги и запуск — [../training.md](../training.md).

## Вне рамок

- Полный RLAIF-цикл с LLM-судьёй — [rlaif-song-loop.md](rlaif-song-loop.md) (пока только подготовка).
- Схема как единственный KPI.
