# Ноутбуки и зонды

Папка для экспериментов и разбора кода, не для боевого кода. Боевой код pic — `core/image/`, текстовых моделей — `core/text/`.

| Файл | О чём |
|------|--------|
| `text_rnn.ipynb` | Char-level LSTM по слоям (код — `core/text/rnn.py`) |
| `experiments/text_lstm_word.ipynb` | Word-level LSTM: токенизация слов, словарь, зонд-трейн, связка с `lstm-word` |
| `text_transformer.ipynb` | Causal transformer (post-norm/ReLU), затем mini-gpt (pre-norm/GELU); в коде — `core/text/lm.py` |
| `text_deepseek.ipynb` | MiniDeepSeek: MLA + RMSNorm/SwiGLU (без MoE); код — `core/text/deepseek.py` |
| `text_bert_gen.ipynb` | BERT MLM + Gibbs decode; код — `core/text/bert_gen.py` |
| `определенее настроение).ipynb` | RNN mood: часть 1 — pos/neg; часть 2 — 2D-вектор → `core/mood/model.py` |
| `исследование пикчи.ipynb` | Сетка sticker/pic, пробные рендеры |

## 1. Текстовые модели (lstm / transformer / gpt / deepseek / bert-gen)

**lstm** — char-LM. Файл `checkpoints/kizaru.pt`, `arch=rnn`, vocab ~2613. В slash / CLI имя **lstm**.

**lstm-word** — тот же `TextRNN`, токены = слова, `checkpoints/kizaru_word.pt`, vocab ~99868 (общий word-словарь).

**transformer** / **gpt** / **deepseek** / **bert-gen** — после retrain 2026-10-05 все **word** на том же vocab ~99868, emb 300:

- transformer — post-norm, ReLU (`core/text/lm.py`)
- gpt — pre-norm, GELU
- deepseek — MLA + RMSNorm + SwiGLU (`core/text/deepseek.py`) — чекпоинт **есть**
- bert-gen — MLM + Gibbs (`core/text/bert_gen.py`) — чекпоинт **есть**

Учат через `python -m training.train` / `training.train_lm` ([../docs/training.md](../docs/training.md)). Train CE и params: [../docs/methods.md](../docs/methods.md).

Модели в одном репо — сравнить одну лирику на RNN / transformer / mini-gpt / DeepSeek-block / BERT-gen. Имена = `/ai text model=…`.

## 2. Как эволюционировал pic (simple / line / render / details)

**simple** — два глифа dark/light (по умолчанию ⬛/⬜) по яркости ячейки. В боте — текст в чат.

**line** — ячейка smile 2×2 (`A B / C D`); classic NN (shade + measured), без spread-gate. Зонды: `probe_unicode_quads` / `pack_pic_dicts_from_probe`; честный MSE — `pic_luma_bench.py`. Контур — shrink-squares. В боте `line` тоже текст.

**render** (в боте) и CLI `--mode line|render` — файл, не чат-сетка: PNG (`pic_still`) / GIF (`pic_video`). Глиф = тайл 8×8. `render` = OKLab L* + color edges, classic NN; `details` (mode 4) — единственный с soft mid-band gate. CLI `anim` с `--mode simple` не работает.

Подробности — корневой README и `исследование пикчи.ipynb`.

## 3. GPU

`core/text/checkpoint.get_device()` → cuda, если `torch.cuda.is_available()`, иначе cpu. Его используют скрипты обучения в `training/`.

Анимация pic: `core/image/anim.py` + `render.py` — CUDA adaptive batch (`pick_gpu_batch`), CPU ProcessPool для `line_mask`, overlap с GPU.

## Скрипты-зонды

`research/rank_line_glyphs.py` и `research/rank_shade_glyphs.py` ранжируют кандидатов по локальным выгрузкам зондов (сейчас удалены, пересоздаются запуском зондов). В приложение уходит только `core/image/assets/line_patterns.json`.

`research/probe_unicode_quads.py` — обход symbol/geometry блоков Unicode, 2×2 яркость, классы flat/soft/line; выход — локальная папка зонда (прошлые выгрузки удалены 2026-10-09). Разбор: [docs/research/unicode-details-glyphs.md](../docs/research/unicode-details-glyphs.md).

`research/pic_luma_bench.py` — цвет в энкодер, Rec.601 gray ref, line/render/details + свип `LEVEL_EPS` / `details_thr`. Отчёт: [docs/research/pic-luma-bench.md](../docs/research/pic-luma-bench.md).

`research/pack_pic_dicts_from_probe.py` — uneven shade ranges + details dedup → `core/image/assets/shade_ramp.json`.

Из корня репозитория:

```bash
python notebooks/research/probe_unicode_quads.py
python notebooks/research/pack_pic_dicts_from_probe.py
python notebooks/research/pic_luma_bench.py
```
