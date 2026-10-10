# Результаты турнира / чекпоинтов

## Сейчас на диске (retrain 2026-10-05)

Новый Genius-корпус (`LM_STREAM`, пустых строк 0) + word vocab **99868** (emb 300). Char только у `lstm` (vocab **2613**).

Цифры — **train `mean_loss`** из meta `.pt`, не held-out bench.

| Слот | level | Train CE | Params | Checkpoint | Заметки |
|------|-------|----------|--------|------------|-------|
| lstm | char | 1.616 | ~13.5M | `checkpoints/kizaru.pt` | 768/2, epoch 50k |
| lstm-word | word | 5.201 | ~85M | `checkpoints/kizaru_word.pt` | emb300/hid512, epoch ~25k |
| transformer | word | 4.479 | ~82.7M | `checkpoints/transformer.pt` | 384/8/8 block128, 60k |
| gpt | word | **4.394** | ~82.7M | `checkpoints/gpt.pt` | лучший causal word train CE |
| deepseek | word | 4.509 | ~80.6M | `checkpoints/deepseek.pt` | MLA, 60k |
| bert-gen | word | 6.243 | ~82.7M | `checkpoints/bert_gen.pt` | MLM CE, 60k |

### Шкала CE: char ≠ word

- **~1.1–1.7** — норма для **char** (vocab сотни–тысячи): каждый токен = символ, предсказывать легче.
- **~3.5–5.5** — нормальный порядок для **word** casual LM на рэп-корпусе (vocab ~50k–100k): следующий **слово** из огромного словаря. **Не надо** гнать word-transformer к CE≈1 — это была бы другая задача (почти запомнить корпус) или char-модель.
- Исторический night-contest transformer/gpt с CE ~1.16–1.17 были **char**, vocab ~814, не word.

Held-out: `python -m experiments.bench --model checkpoints/<name>.pt` на замороженном `experiments/shared/test.txt` — дописать сюда.

---

## Архив night-contest (эпоха char / word ~57k)

Старый турнир команд A/B (утро 05.10 и раньше). Чекпоинты с тех прогонов **перезаписаны** вечерним retrain. Цифры ниже — архив, не текущий диск.

### lstm_char
Победитель: Team A (CE 1.334966 vs Team B 1.377270). Тогда: vocab ~814, ~10.7M.

### lstm_word
Победитель: Team B (CE 3.705130 vs Team A 3.739446). Тогда: vocab ~57k, ~63M.

### transformer
Победитель: Team B (CE 1.166438 vs Team A best 1.188683). Тогда: **char**, vocab ~814, ~15M.

### gpt
Победитель: Team B (CE 1.171173 vs Team A 1.445534). Тогда: **char**, vocab ~814, ~15M.
