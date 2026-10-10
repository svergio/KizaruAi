# Word-эмбеддинги / vocab для Word LSTM

**Прод (вечер 2026-10-05):** word-чекпоинты на диске делят vocab **99868**, emb **300** (`lstm-word`, `transformer`, `gpt`, `deepseek`, `bert-gen`). Цифры coverage ниже (65k / min_freq=5) — срез до финального vocab; актуальные params/CE — в [../methods.md](../methods.md) и [../../experiments/RESULTS.md](../../experiments/RESULTS.md).

## Выбранный ассет

| Поле | Значение |
|-------|-------|
| HF repo | [`BorisTM/natasha_navec_hudlit_v1_12B_500K_300d_100q`](https://huggingface.co/BorisTM/natasha_navec_hudlit_v1_12B_500K_300d_100q) |
| Апстрим | Natasha [Navec](https://github.com/natasha/navec) `hudlit_v1` (литературный RU, ~500k слов, 300d) |
| Лицензия | MIT (`LICENSE.txt` в локальной папке) |
| Путь | `data/embeddings/navec_hudlit/` |

Почему не news FastText / Word2Vec:

- Большой word-vocab по-русски, в т.ч. «сленговые» литературные формы (`братва`, `бабки`, `шлюха`, …).
- MIT, бесплатно, офлайн после скачивания.
- Те же векторы годятся для init матрицы эмбеддингов (не только список слов).
- Литературный `hudlit` ближе к рэп/сленгу, чем чистый news-пак.

Рассматривались: `BorisTM/natasha_navec_news_v1_…` (меньше, новости), `Shu-vi/Russian-News-FastText-Skip-gram-100D` (крошечный 100d gensim), `SherifAnar/russian-news-word2vec-300d` (news gensim).

## Локальные файлы

```
data/embeddings/navec_hudlit/
  tokenizer.json      # WordLevel vocab (~500k) — источник whitelist
  model.safetensors   # embedding.weight [500004, 300]
  LICENSE.txt
  README.md
  seed_whitelist.txt  # генерит prep-скрипт
  seed_missing.txt    # токены сидов, которых нет в Navec
```

Перескачать:

```bash
hf download BorisTM/natasha_navec_hudlit_v1_12B_500K_300d_100q ^
  --local-dir data/embeddings/navec_hudlit ^
  --include "tokenizer.json" --include "model.safetensors" ^
  --include "LICENSE.txt" --include "README.md" ^
  --include "modules.json" --include "config_sentence_transformers.json"
```

Опционально для init весов: `pip install safetensors`.

## Как это стыкуется с word-трейном

Проблема: vocab Word LSTM строится с `min_freq` (`core/text/tokenize.py`). Редкий Discord-сленг уходит в `<unk>`, даже если слово нормальное русское.

**Не** сваливай весь список Navec (500k) в корпус. Мержи **маленький whitelist** (сиды / сленг) в vocab и при желании копируй строки Navec в `nn.Embedding`.

### 1) Whitelist из сидов турнира

```bash
python -m training.prep_embedding_vocab ^
  --seeds experiments/shared/seeds.json ^
  --check-ckpt checkpoints/kizaru_word.pt
```

Пишет `seed_whitelist.txt` (токены есть в Navec) и `seed_missing.txt` (например бренды вроде `зиртек`).

### 2) Следующий word finetune / retrain

```bash
python -m training.train --level word ^
  --whitelist data/embeddings/navec_hudlit/seed_whitelist.txt ^
  --init-navec --embedding-size 300
```

- `--whitelist` — оставить перечисленные токены даже при freq `< min_freq` и дописать seed-only.
- `--init-navec` — скопировать совпавшие строки из `model.safetensors` в `model.encoder` (обрежет/паддит, если dim ≠ 300). Лучше `--embedding-size 300`, чтобы взять полные векторы Navec.
- Хелперы: `core/text/pretrained_vocab.py` (`load_wordlevel_vocab`, `merge_whitelist`, `init_embedding_from_navec`).

### 3) Word-level transformer / GPT / DeepSeek / BERT-gen LM

Char LM не умеет строки Navec (нет word id). Нужны `--level word` и те же флаги whitelist/Navec на `training.train_lm`. Токены в `model.tok`; опциональный `--embedding-size 300` сохраняет полные векторы Navec и проецирует в `d_model` через `model.emb_proj`, если размеры разные (можно оставить winner-ish `d_model=384` / 8 голов).

```bash
python -m training.train_lm --arch transformer --level word ^
  --whitelist data/embeddings/navec_hudlit/seed_whitelist.txt ^
  --init-navec --embedding-size 300 --d-model 384 --n-heads 8
```

Тот же паттерн для `--arch gpt|deepseek|bert_gen`. Воркеры читают `level=word` и сэмплят через word detokenize (`evaluate_lm_words` / word `generate_mlm`).

Finetune со старого `kizaru_word.pt` требует совместимого размера vocab (рост vocab меняет `Embedding` / `fc`) — для спасения сленга чище свежий word-train с whitelist + Navec init; либо руками расширять тензоры чекпоинта позже.

## Coverage на текущем корпусе (2026-10-05)

`data/corpus/kizaru_lyrics.txt`, 5.89M word-токенов, 208 865 типов. Vocab Navec — 500 004.

| Срез | Доля |
|-------|-------|
| Токены, чьё слово есть в Navec | 92.2% |
| Типы в Navec | 63.1% (131 776 / 208 865) |
| Model vocab (`freq >= 5`, 65 164 типов) с вектором Navec | 80.6% |
| Токены → `<unk>` при `min_freq=5` | 4.2% |
| То же, если редкие слова из Navec в whitelist | 1.7% |

Частые слова, которых **нет** в Navec: в основном Genius page JSON (`apipath`, `lyricsstate`, `annotatable`) плюс сленг и имена (`skrrt`, `feduk`, `lil`). JSON уходит на следующем `prepare_data --source genius-cache`; сленг остаётся со случайным init.

Navec инициализирует смысл. **Рифме он не учит:** у вектора нет конца строки. На очищенных барах (25 артистов, 3 032 песни, 163k строк) зона рифмы соседних строк совпадает **3.9%** времени — **×40** к shuffle-baseline; **5.9%** строк рифмуются с одной из следующих трёх. Сигнал реальный и уже в рэп-дате. Классические стихи поднимут rate и сменят регистр; это **не** замена недостающему объёму рэпа.

## Smoke

- Покрытие сидов в Navec: обычный сленг/литформы — да; имена собственные / бренды — могут остаться `<unk>` или в `--extra-words` со случайным init.
- Не считай Navec корпусом лирики; лицензия Genius отдельно (см. `LICENSE` проекта).
