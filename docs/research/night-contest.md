# Ночной турнир: 4 модели × 2 команды

Дата прогона: 2026-10-05. Одна GPU (RTX 3080). Mood/pic/video в турнир не входили.

## Формат

- 4 слота Discord-текста: `lstm` char, `lstm-word`, `transformer` char, `gpt` char.
- На модель: команда A до ~90 мин, затем B до ~90 мин, затем общий бенч. Дуэль = **90+90**, не «90 на двоих».
- Hard resource: **одна GPU**; два train параллельно запрещены. После wall — handoff/sleep → coordinator (workers до RESULTS не поднимать).
- Общий замороженный train/test: `experiments/shared/` (song-level Genius, seed `20261005`). `test.txt` команды не трогают.
- Метрика: test cross-entropy (`experiments/bench.py`). Победитель → `checkpoints/*.pt` (схема: 2 teams × N → bench → checkpoints).
- Tiebreak при `gap < 0.02`: меньше повторов / больше кириллицы; при полном равенстве — команда A (дефолты).
- Промпт-паттерны (outcome+absence, роли vs mega-agent, «как по пулу», Plan→approve→Build): см. [agent-pool-training.md](agent-pool-training.md).

## Результаты (победители)

| Слот | Winner | test CE | Checkpoint | Params / vocab |
|------|--------|---------|------------|----------------|
| lstm char | A | ~1.335 | `checkpoints/kizaru.pt` | ~10.7M / 814 |
| lstm-word | B | ~3.705 | `checkpoints/kizaru_word.pt` | ~63M / ~57k |
| transformer | B | ~1.166 | `checkpoints/transformer.pt` | ~15M / 814 |
| gpt | B | ~1.171 | `checkpoints/gpt.pt` | ~15M / 814 |

Точные цифры и пути: [experiments/RESULTS.md](../../experiments/RESULTS.md).

Шкала CE у word ≠ char: сравнивать word только с word.

## Сэмпл-проверка

Сиды в `experiments/shared/seeds.json` (в т.ч. сленговая фраза). На сленге word-модель сыпется в `<unk>` (словарный OOV). На той же фразе GPT выглядел лучше остальных char-моделей.

## Следующий шаг (не сделано в ту ночь)

Для word: не заливать «толковый словарь» как корпус. Имеет смысл **slang whitelist finetune** — расширить vocab целевым сленгом/токенами под Discord-сиды, не раздувать train общим словарём.

## Артефакты

```
experiments/<model>/team_{a,b}/{config.json,model.pt,train.log,score.json,...}
experiments/RESULTS.md
checkpoints/{kizaru,kizaru_word,transformer,gpt}.pt
checkpoints/prev/   # бэкап до копирования победителей
```
