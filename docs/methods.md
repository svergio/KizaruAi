# Methods — text, mood, pic

Краткий обзор с живыми цифрами и inline-математикой — [../README.md](../README.md) (секция **Математика / алгоритмы**). Архитектура процессов — [architecture.md](architecture.md). История отбора чекпоинтов (research) — [research/night-contest.md](research/night-contest.md).

Это **next-token language model**, не RL и не RLHF. Корпус режется на символы или слова; модель учится предсказывать следующий токен; на инференсе семплирует продолжение от сида.

## Next-token / CrossEntropy

Для последовательности токенов \(x_1,\ldots,x_T\) модель выдаёт распределение \(p_\theta(x_t\mid x_{<t})\) (в коде часто сдвиг \(x_{t+1}\mid x_{\le t}\)). Лосс на шаге:

\[
\mathcal{L} = -\log p_\theta(x_t\mid x_{<t})
\]

В коде — `torch.nn.CrossEntropyLoss` по логитам словаря. Оптимизатор Adam. Метрика после обучения — **test CE** на замороженном `experiments/shared/test.txt` (`experiments/bench.py`).

Char-модели сравнимы между собой (один vocab ~814). Word — отдельная шкала (vocab ~57k, другой токен).

## LSTM char

Discord: `/ai text model:lstm`.

- Checkpoint: `checkpoints/kizaru.pt` (текущий после обучения).
- Конфиг: embedding/hidden **768**, **2** слоя LSTM, BPTT 128, seq 512.
- Vocab: символы (~814). Params ~10.7M.
- Test CE: **1.335**.
- Читает слева направо, держит скрытое состояние — удобно для локального ритма строки.

Sketch: \(h_t=\mathrm{LSTM}(e(x_t),h_{t-1})\), \(p=\mathrm{softmax}(W h_t)\).

Прогон / цифры (research): [research/night-contest.md](research/night-contest.md), [../experiments/RESULTS.md](../experiments/RESULTS.md).

## LSTM word

Discord: `/ai text model:lstm-word`.

- Checkpoint: `checkpoints/kizaru_word.pt` (текущий после обучения).
- Конфиг: embedding/hidden **512**, **2** слоя, seq 128, bptt 64, `min_freq=5`.
- Vocab слов ~57k (токенizer `core/text/tokenize.py`). Params ~63M.
- Test CE: **3.705** (не сравнивать с char CE).
- OOV → `<unk>`: на сленговых Discord-сидах word сыпется чаще char-моделей.

Embeddings / Navec: [research/embeddings-vocab.md](research/embeddings-vocab.md).

## Transformer

Discord: `/ai text model:transformer`.

- Checkpoint: `checkpoints/transformer.pt` (текущий после обучения).
- Конфиг: `d_model=384`, `n_layers=8`, `n_heads=8`, `block_size=384`, 45k steps.
- Post-norm Vaswani-style (`VaswaniBlock` в `core/text/lm.py`: ReLU FFN, LN после residual).
- Causal self-attention на окне `block_size` — видит контекст целиком, не только через hidden.
- Test CE: **1.166** (лучший среди текущих char-чекпоинтов).
- Vocab char ~814, params ~15M.

## GPT

Discord: `/ai text model:gpt`.

- Checkpoint: `checkpoints/gpt.pt` (текущий после обучения).
- Та же ёмкость, что у transformer: 384/8/8, block 384, 45k steps.
- Pre-norm GPT-block (`GPTBlock`: GELU FFN, LN перед attn/ffn).
- Test CE: **1.171**.
- Тот же next-token setup; разница с transformer — нормализация и активация блока.

## DeepSeek (MiniDeepSeek)

Discord: `/ai text model:deepseek` (после появления чекпоинта).

- Checkpoint: `checkpoints/deepseek.pt` — **pending** (архитектура готова, полный train на новом корпусе ещё не запускали).
- Код: `core/text/deepseek.py` (`MiniDeepSeek`).
- **MLA** (Multi-Head Latent Attention): Q/KV через low-rank latent (`kv_lora_rank` / `q_lora_rank`, дефолт `d_model // 4`), затем expand в heads + causal mask.
- **RMSNorm + SwiGLU**, pre-norm residual — паттерн блока из open DeepSeek / Llama-подобных, без MoE.
- Не DeepSeek-V3 MoE: для pet-scale (~15–50M, один GPU) dense FFN.
- Обучение: `python -m training.train_lm --arch deepseek ...` (тот же loop, что transformer/gpt).
- Ноутбук: [../notebooks/text_deepseek.ipynb](../notebooks/text_deepseek.ipynb).

## BERT-gen (MLM)

Discord: `/ai text model:bert-gen` (после чекпоинта). Историческая генерация из BERT: Wang & Cho 2019, Gibbs по MLM, не causal GPT.

- Код: `core/text/bert_gen.py` (`BertMLM`).
- Двусторонний Transformer (без causal mask), post-norm, GELU.
- В словарь добавлен `[MASK]` (`mask_id = vocab`).
- Обучение: 15% позиций, 80/10/10, CrossEntropy только на масках (`ignore_index=-100`).
- Инференс: сид слева, хвост `[MASK]` до `seq_len` (бот: 128), последовательный unmask, затем `n_iters` Gibbs.
- Train (после корпуса): `python -m training.train_lm --arch bert_gen --level char --d-model 384 --n-layers 8 --n-heads 8 --block-size 256 --steps 45000 --out checkpoints/bert_gen.pt`
- Ноутбук: [../notebooks/text_bert_gen.ipynb](../notebooks/text_bert_gen.ipynb).

## Декод / семплирование

На каждом шаге логиты делятся на температуру \(T\) и семплируются: \(p_i=\mathrm{softmax}(z_i/T)\). Ниже \(T\) — ближе к argmax и к повторам, выше — больше редких токенов и битых слов. Стоп — шесть непустых строк либо лимит длины.

| Модель | temp | Лимит | Почему так |
|--------|------|-------|------------|
| lstm | 0.4 | 360 симв. | повтор строки не ловился ни на 0.4, ни на 0.8 |
| lstm-word | 0.8 | 72 токена | лупа нет, узкое место — OOV |
| transformer | 0.9 | 360 симв. | колено: луп 13% → 3% при потере 2.8 п.п. реальных слов |
| gpt | 1.0 | 360 симв. | колено: повтор 50% → 17% |
| deepseek | 0.9 | 360 симв. | как transformer, до своего чекпоинта |
| bert-gen | 1.0 | окно 128, 32 Gibbs | не causal: длина фиксирована, качество растёт с числом проходов |

Это только дефолты. Пользователь переопределяет их в `/ai text` (`temp`, `lines`, `length`, `iters`) и в `python -m app.cli.generate` теми же флагами. Таблица диапазонов — в [README](../README.md#ai-text). Источник правды — `core/text/params.py`: там же клампы, и подрезка идёт до вызова семплера, так что воркер не может получить `temp=50`.

Жёсткого запрета повтора строки нет специально: один повтор хука — это припев, а не дефект.

Замеры и кривая «луп против мусора»: [research/generation-notes.md](research/generation-notes.md).

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

| Mode | Discord | Суть |
|------|---------|------|
| **line** | текст, 31 колонка | глифы из `line_patterns.json` (паттерны 2×2 яркости) |
| **simple** | текст, 31 колонка | 11-тоновая рампа блочных символов |
| **cli** | файл PNG/GIF | тайл 8×8; `scale` / `cells` |

CLI ставит ту же очередь: `python -m app.cli.image`.

### Алгоритм и размер (`cli` / `--mode line`)

Константы: `OUTPUT_SMILE_PX = 8`, `CLI_DEFAULT_SCALE = 2`, `CLI_DEFAULT_CELLS = 320`.

Пайплайн кратко:

1. Кадр → luma + stretch контраста.
2. Сетка `cells` колонок; высота по aspect.
3. Ячейка читает окно `(2 × scale) × (2 × scale)` px (четыре квадранта `scale × scale`), выбирает ближайший глиф по паттерну 2×2.
4. В PNG/GIF глиф рисуется тайлом 8×8.

Две величины: ширина выхода `W_out = cells × 8`; «сытая» ширина источника без растягивания `W_need = cells × 2 × scale`.

`upscale` (API, по умолчанию `True`): при `W_src < W_need` либо nearest-up (сетка `cells` сохраняется), либо при `False` — сжатие `cols` до `W_src // (2 × scale)`. Discord и CLI-воркеры всегда передают `upscale=True`; отдельного флага в slash/CLI нет.

Правило: `cells × 2 × scale ≈ ширина кадра`. Для **~1800 px** без upscale: **`scale=3`, `cells=300`** (`W_need=1800`). Дефолт `cells=320` при `scale=3` даёт `W_need=1920` — лёгкий nearest-up. Discord `cli` режет файлы > 8 MB — при `cells=320` часто снижай `cells`.

| Ширина ≈ | scale | cells | W_need | W_out |
|----------|-------|-------|--------|-------|
| 480 | 1–2 | 80–120 | 160–480 | 640–960 |
| 1080 | 2 | 160–200 | 640–800 | 1280–1600 |
| 1800 | 3 | 300 | 1800 | 2400 |
| 1800 | 3 | 320 | 1920 | 2560 |
| 1920 | 2 | 320 | 1280 | 2560 |

Практический обзор тех же правил — в [README](../README.md#pic-scale-cells-upscale).

## Сводка CE (после обучения / текущие чекпоинты)

| Слот | test CE | Checkpoint |
|------|---------|------------|
| lstm char | 1.335 | `checkpoints/kizaru.pt` |
| lstm-word | 3.705 | `checkpoints/kizaru_word.pt` |
| transformer | 1.166 | `checkpoints/transformer.pt` |
| gpt | 1.171 | `checkpoints/gpt.pt` |
| deepseek | — (pending) | `checkpoints/deepseek.pt` |

История team-сравнений (research): [research/night-contest.md](research/night-contest.md), [../experiments/RESULTS.md](../experiments/RESULTS.md). RPS / замеры: [services.md](services.md).
