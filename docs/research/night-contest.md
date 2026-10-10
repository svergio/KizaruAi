# Ночной турнир: 4 модели × 2 команды

Дата прогона: 2026-10-05 (утро / день). Одна GPU (RTX 3080). Mood/pic/video в турнир не входили.

**Статус:** архив. Вечером 05.10 все текстовые `.pt` переобучены на новом корпусе + word vocab 99868 (см. [../../experiments/RESULTS.md](../../experiments/RESULTS.md) — секция Current). Цифры ниже не описывают текущий диск.

## Формат

- 4 слота Discord-текста тогда: `lstm` char, `lstm-word`, `transformer` **char**, `gpt` **char**.
- На модель: команда A до ~90 мин, затем B до ~90 мин, затем общий бенч. Дуэль = **90+90**.
- Hard resource: **одна GPU**; два train параллельно запрещены.
- Общий замороженный train/test: `experiments/shared/` (song-level Genius, seed `20261005`).
- Метрика: test cross-entropy (`experiments/bench.py`).
- Tiebreak при `gap < 0.02`: меньше повторов / больше кириллицы.
- Промпт-паттерны: [agent-pool-training.md](agent-pool-training.md).

## Результаты (победители той ночи)

| Слот | Winner | test CE | Тогдашний конфиг |
|------|--------|---------|------------------|
| lstm char | A | ~1.335 | ~10.7M / vocab 814 |
| lstm-word | B | ~3.705 | ~63M / ~57k |
| transformer | B | ~1.166 | ~15M / char 814 |
| gpt | B | ~1.171 | ~15M / char 814 |

## Что стало после

Вечерний retrain 05.10:

- корпус `LM_STREAM` (0 пустых строк)
- word vocab **99868**, emb 300 (Navec-размер)
- `transformer` / `gpt` / `deepseek` / `bert-gen` → **word**
- char остался только у `lstm` (vocab **2613**)

Актуальные train CE и params — [../../experiments/RESULTS.md](../../experiments/RESULTS.md), [../methods.md](../methods.md).

## Артефакты турнира (лог команд)

```
experiments/<model>/team_{a,b}/{config.json,model.pt,train.log,score.json,...}
experiments/RESULTS.md   # сверху — Current; снизу — этот архив
```
