# Ноутбуки и зонды

Папка для экспериментов и разбора кода, не для production API. Стоп-кадр и анимация в проде — `pic.py` и `kizaru/`. Черновиков бота с `legacy/` здесь нет.

| Файл | О чём |
|------|--------|
| `text_rnn.ipynb` | Char-level LSTM по слоям (та же идея, что `kizaru/model.py`) |
| `experiments/text_lstm_word.ipynb` | Word-level LSTM: токенизация слов, словарь, зонд-трейн, связка с `lstm-word` |
| `text_transformer.ipynb` | Causal transformer (post-norm/ReLU), затем mini-gpt (pre-norm/GELU); в коде — `kizaru/lm.py` |
| `text_deepseek.ipynb` | MiniDeepSeek: MLA + RMSNorm/SwiGLU (без MoE); код — `core/text/deepseek.py` |
| `text_bert_gen.ipynb` | BERT MLM + Gibbs decode; код — `core/text/bert_gen.py` |
| `определенее настроение).ipynb` | RNN mood: часть 1 — pos/neg; часть 2 — 2D-вектор → `kizaru/mood.py` |
| `исследование пикчи.ipynb` | Сетка sticker/pic, пробные рендеры |

## 1. Текстовые модели (lstm / transformer / gpt / deepseek)

**lstm** — char-LM из `text_rnn.ipynb` и `kizaru/model.py`. Файл `checkpoints/kizaru.pt`, `arch=rnn`. В slash / CLI имя **lstm**.

**lstm-word** — тот же `TextRNN`, но токены = слова (`kizaru/tokenize.py`), чекпоинт `checkpoints/kizaru_word.pt`, воркер `worker_lstm_word.py`. Transformer/gpt/deepseek тоже умеют `--level word` на том же словаре.

**transformer** — causal LM: post-norm, ReLU. **gpt** — тот же task, pre-norm и GELU. Оба учатся через `train_lm.py` (дефолт около 800 шагов).

**deepseek** — MLA + RMSNorm + SwiGLU (`text_deepseek.ipynb`, `core/text/deepseek.py`). Чекпоинт `checkpoints/deepseek.pt` — после следующего retrain; пока не учить.

Загруженные чекпоинты сняты со старого корпуса HF/CIS, не с пула Genius. У того прогона у lstm средний loss около 1.53, у transformer/gpt около 2.64–2.66. Текст слабый. `train.py`, `train_lm.py` и `train_mood.py` не запускать, пока датасет не подготовлен (загрузка Genius ещё может идти).

Модели в одном репо нужны, чтобы сравнить одну и ту же лирику на RNN, transformer, mini-gpt и DeepSeek-block. Имена совпадают с ботом: `/ai text`, параметр `model`.

## 2. Как эволюционировал pic (line / simple / cli)

**simple** — сетка из эмодзи по яркости (одиннадцать ступеней). В боте это текст в чат, 31 колонка.

**line** — ячейка smile 2×2 (`A B / C D`); line-символы из измеренных паттернов. Зонды: `research/rank_line_glyphs.py` и `research/rank_shade_glyphs.py` по локальным `data/export/*_probe/` (gitignore) → `kizaru/assets/line_patterns.json`. Контур — shrink-squares по цветным рёбрам (quadtree). В боте `line` тоже текст, 31 колонка.

**cli** (в боте) и `--mode line` в CLI — файл, не чат-сетка: PNG для картинки (`pic_still`), GIF для gif/видео (`pic_video`). Каждый глиф штампуется в тайл 8×8 (`indexes_to_image`, `render_fast`). JSON только выбирает символ. CLI `anim` с `--mode simple` не работает. Флагов `--audio`, `--mp4`, `--gif` у `pic.py` нет.

Подробности — корневой README и `исследование пикчи.ipynb`.

## 3. GPU

`kizaru/checkpoint.get_device()` → cuda, если `torch.cuda.is_available()`, иначе cpu.

`train.py`, `train_lm.py` и `train_mood.py` берут этот device. Сейчас их не запускать: датасет ещё не готов, текущие `.pt` трогать не нужно.

Анимация pic: `kizaru/render_fast.py` — на CUDA батч 4 (`GPU_BATCH`), на CPU по одному кадру.

## Скрипты-зонды

`research/rank_line_glyphs.py` и `research/rank_shade_glyphs.py` ранжируют кандидатов по локальным `data/export/*_probe/`. В приложение уходит только `kizaru/assets/line_patterns.json`.

Из корня репозитория:

```bash
python notebooks/research/rank_line_glyphs.py
python notebooks/research/rank_shade_glyphs.py
```
