# Методы — text, mood, pic

Краткий обзор с живыми цифрами и inline-математикой — [../README.md](../README.md) (секция **Математика / алгоритмы**). Архитектура процессов — [architecture.md](architecture.md). История night-contest (старая эра) — [research/night-contest.md](research/night-contest.md).

Слоты бота — **next-token language model** (bert-gen — MLM). Экспериментальное RL-дообучение DeepSeek с наградой от классификатора рифмы описано отдельно — [training.md](training.md); в бот оно не подключено. Корпус режется на символы или слова; модель учится предсказывать токен; на инференсе семплирует продолжение от сида.

**Актуально (retrain 2026-10-05):** корпус `LM_STREAM` (пустых строк нет), word vocab **99868**, emb **300**. Char остался только у `lstm`. Deepseek / bert-gen — **есть** на диске, не pending.

Метрика в таблицах ниже — **train `mean_loss` из meta `.pt`**. Held-out test CE (`experiments/bench.py`) на этом vocab ещё не перегоняли.

## Next-token / CrossEntropy

Для последовательности токенов \(x_1,\ldots,x_T\) модель выдаёт \(p_\theta(x_t\mid x_{<t})\). Лосс:

\[
\mathcal{L} = -\log p_\theta(x_t\mid x_{<t})
\]

В коде — `torch.nn.CrossEntropyLoss`. Оптимизатор Adam.

Char CE и word CE — **разные шкалы** (vocab 2613 vs 99868). Среди word causal сравнивать только word↔word.

## LSTM char

Discord: `/ai text model:lstm`.

- Checkpoint: `checkpoints/kizaru.pt` (05.10).
- Конфиг: embedding/hidden **768**, **2** слоя LSTM.
- Vocab: символы (**2613**). Params ~**13.5M**.
- Train CE: **1.616**.
- Единственный char-слот после retrain — сиды без `<unk>`.

Sketch: \(h_t=\mathrm{LSTM}(e(x_t),h_{t-1})\), \(p=\mathrm{softmax}(W h_t)\).

## LSTM word

Discord: `/ai text model:lstm-word`.

- Checkpoint: `checkpoints/kizaru_word.pt` (05.10).
- Конфиг: emb **300** / hidden **512**, **2** слоя.
- Vocab слов **99868** (`core/text/tokenize.py`). Params ~**85M**.
- Train CE: **5.201** (не сравнивать с char).
- OOV → `<unk>`: на сленговых Discord-сидах сыпется чаще char.

Embeddings / Navec: [research/embeddings-vocab.md](research/embeddings-vocab.md).

## Transformer

Discord: `/ai text model:transformer`.

- Checkpoint: `checkpoints/transformer.pt` (05.10) — **word**.
- Конфиг: `d_model=384`, `n_layers=8`, `n_heads=8`, `block_size=128`, emb **300**.
- Post-norm Vaswani-style (`VaswaniBlock` в `core/text/lm.py`: ReLU FFN, LN после residual).
- Vocab **99868**, params ~**82.7M**.
- Train CE: **4.479**.

## GPT

Discord: `/ai text model:gpt`.

- Checkpoint: `checkpoints/gpt.pt` (05.10) — **word**.
- Та же ёмкость: 384/8/8, block 128, emb 300.
- Pre-norm GPT-block (`GPTBlock`: GELU FFN, LN перед attn/ffn).
- Train CE: **4.394** (лучший среди текущих causal word по train loss).
- Params ~**82.7M**.

## DeepSeek (MiniDeepSeek)

Discord: `/ai text model:deepseek`.

- Checkpoint: `checkpoints/deepseek.pt` (05.10) — **word**, не pending.
- Код: `core/text/deepseek.py` (`MiniDeepSeek`).
- **MLA** (Multi-Head Latent Attention): Q/KV через low-rank latent (`kv_lora_rank` / `q_lora_rank`, дефолт `d_model // 4`), затем expand в heads + causal mask.
- **RMSNorm + SwiGLU**, pre-norm residual — без MoE, без RoPE.
- Конфиг: 384/8/8, block 128, emb 300. Params ~**80.6M**.
- Train CE: **4.509**.
- Обучение: `python -m training.train_lm --arch deepseek --level word ...`
- Ноутбук: [../notebooks/text_deepseek.ipynb](../notebooks/text_deepseek.ipynb).

## BERT-gen (MLM)

Discord: `/ai text model:bert-gen`. Историческая генерация из BERT: Wang & Cho 2019, Gibbs по MLM, не causal GPT.

- Checkpoint: `checkpoints/bert_gen.pt` (05.10) — **word**, не pending.
- Код: `core/text/bert_gen.py` (`BertMLM`).
- Двусторонний Transformer (без causal mask), post-norm, GELU.
- В словарь добавлен `[MASK]` (`mask_id`).
- Обучение: 15% позиций, 80/10/10, CrossEntropy только на масках (`ignore_index=-100`).
- Инференс: сид слева, хвост `[MASK]`, unmask + `n_iters` Gibbs. Word runtime: окно ≤64 / `block_size`.
- Конфиг: 384/8/8, block 128, emb 300. Params ~**82.7M**.
- Train CE (MLM): **6.243**.
- Ноутбук: [../notebooks/text_bert_gen.ipynb](../notebooks/text_bert_gen.ipynb).

## Декод / семплирование

На каждом шаге логиты делятся на температуру \(T\): \(p_i=\mathrm{softmax}(z_i/T)\). Стоп — шесть непустых строк либо лимит длины.

| Модель | level | temp | length (runtime) |
|--------|-------|------|------------------|
| lstm | char | 0.4 | 360 симв. |
| lstm-word | word | 0.8 | 72 токена |
| transformer | word | 0.9 | 72 токена |
| gpt | word | 1.0 | 72 токена |
| deepseek | word | 0.9 | 72 токена |
| bert-gen | word | 1.0 | окно ≤64, 32 Gibbs |

У word-LM хендлер (`core/bus/handlers.py`) при пустом `length` форсит 72 токена (у bert — min(64, block)). Источник дефолтов temp/lines — `core/text/params.py`; клампы до семплера.

Жёсткого запрета повтора строки нет: один повтор хука — припев.

Замеры лупов на char-эре: [research/generation-notes.md](research/generation-notes.md) (до вечернего word-retrain — цифры temp→луп устарели для attention-слотов).

## Mood

Discord: `/ai mood`.

Не генерация. Маленькая RNN (`core/mood/model.py`, `checkpoints/mood.pt`, `kind=vector2`).

Две оси (sigmoid-вероятности):

| Ось | 0 | 1 |
|-----|---|---|
| 0 | злой | добрый |
| 1 | грустный | веселый |

Порог 0.5 даёт имя оси в ответе. Пример: вход `я злой и весёлый` → `злой 0.00 | веселый 1.00`, `vector=(0.00, 1.00)`.

Обучение: `python -m training.train_mood`. Отдельного CLI нет — только `/ai mood` или очередь `kind=mood`.

## Pic

Discord: `/ai pic`. Не нейросеть — яркость кадра → глифы (`core/image/ascii.py`). Демо анимации: [Bad Apple на YouTube](https://youtu.be/yNeDeC8j7tY).

| Mode | Discord / CLI | Суть |
|------|---------------|------|
| **simple** | текст; `cli`/PNG | ровно 2 глифа `dark`/`light` (⬛/⬜), квант 0\|1 |
| **line** | текст; `cli`/PNG | classic NN shade+measured, Rec.601; shade-рампа из `shade_ramp.json` |
| **render** | PNG/GIF | classic NN + OKLab L* + color edges |
| **details** | текст; `cli`/PNG | soft `DETAILS_PATTERNS` mid-band (отдельный режим) |

CLI: `python -m app.cli.image still|anim --mode simple|line|render|details`.  
Discord: те же 4 режима + флаг `cli` → ответ файлом.

`line`/`render` — classic NN без hard `LINE_THR` gate. Soft details только в mode `details`. Регрессия: [research/pic-quality-regression.md](research/pic-quality-regression.md).

### Алгоритм и размер (`line` / `render` / `details`)

Константы: `OUTPUT_SMILE_PX = 8`, `CLI_DEFAULT_SCALE = 2`, `CLI_DEFAULT_CELLS = 320`.

1. Tone: `line`/`details` = Rec.601; `render` = OKLab L* (`render_line_opts`).
2. Mask: `line`/`details` = luma quadtree; `render` = `structure_field` (color edges).
3. Сетка `cells`; окно `(2×scale)²` → глиф; в файле тайл 8×8.

`W_out = cells × 8`, `W_need = cells × 2 × scale`. Discord `render` режет файлы > 8 MB — снижай `cells`.

| Ширина ≈ | scale | cells | W_need | W_out |
|----------|-------|-------|--------|-------|
| 480 | 1–2 | 80–120 | 160–480 | 640–960 |
| 1080 | 2 | 160–200 | 640–800 | 1280–1600 |
| 1800 | 3 | 300 | 1800 | 2400 |
| 1800 | 3 | 320 | 1920 | 2560 |
| 1920 | 2 | 320 | 1280 | 2560 |

Практический обзор тех же правил — в [README](../README.md#render-scale-cells-upscale).

## Сводка train CE (чекпоинты 2026-10-05)

| Слот | level | Train CE | Params | Checkpoint |
|------|-------|----------|--------|------------|
| lstm char | char | **1.616** | ~13.5M | `checkpoints/kizaru.pt` |
| lstm-word | word | **5.201** | ~85M | `checkpoints/kizaru_word.pt` |
| transformer | word | **4.479** | ~82.7M | `checkpoints/transformer.pt` |
| gpt | word | **4.394** | ~82.7M | `checkpoints/gpt.pt` |
| deepseek | word | **4.509** | ~80.6M | `checkpoints/deepseek.pt` |
| bert-gen | word | **6.243** (MLM) | ~82.7M | `checkpoints/bert_gen.pt` |

История team-сравнений (char/~57k эра): [research/night-contest.md](research/night-contest.md), [../experiments/RESULTS.md](../experiments/RESULTS.md). RPS: [services.md](services.md).
