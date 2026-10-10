# Сервисы — smoke, CE, RPS

**Train CE** — `mean_loss` из meta чекпоинтов после retrain **2026-10-05** (word vocab 99868 / char 2613).
**Smoke / RPS** — старый прогон (char-эра, до word-retrain); после вечера 05.10 не переснимали.
Held-out CE: `python experiments/bench.py <checkpoint.pt>` на `experiments/shared/test.txt` — на новом vocab ещё не гоняли.

Метод RPS (для перезамера): 8 заявок на сервис, на GPU одна модель за раз; `rps = 8 / wall_seconds` от первой постановки до последнего `done`. Сид для текста: `Я`.

## Текстовые воркеры

| Сервис | Чекпоинт | level | Train CE | Smoke s† | RPS† | 8 заявок, s† | Smoke† |
|--------|----------|-------|----------|----------|------|--------------|--------|
| lstm (char) | `checkpoints/kizaru.pt` | char | 1.616 | 1.036 | 3.377 | 2.369 | PASS |
| lstm-word | `checkpoints/kizaru_word.pt` | word | 5.201 | 0.894 | 3.587 | 2.230 | PASS |
| transformer | `checkpoints/transformer.pt` | word | 4.479 | 1.484 | 1.402 | 5.706 | PASS |
| gpt | `checkpoints/gpt.pt` | word | 4.394 | 1.214 | 1.583 | 5.053 | PASS |
| deepseek | `checkpoints/deepseek.pt` | word | 4.509 | — | — | — | загружен |
| bert-gen | `checkpoints/bert_gen.pt` | word | 6.243 | — | — | — | загружен |

† Smoke/RPS сняты на char-весах; сейчас воркеры грузят word-веса (vocab 99868). Нужен перезамер.

Smoke-сид: `Я купил фенизипам для ворка на вск 4`.
`worker_kind`: `text_lstm`, `text_lstm_word`, `text_transformer`, `text_gpt`, `text_deepseek`, `text_bert_gen`.

## Mood / pic (без CE)

| Сервис | Чекпоинт | Вход | Smoke s | RPS (8 заявок) | 8 заявок, s | Smoke |
|--------|----------|------|---------|----------------|-------------|-------|
| mood | `checkpoints/mood.pt` | `я злой и весёлый` | 0.539 | 3.830 | 2.089 | PASS |
| pic still (line) | — | PNG 32×32 | 0.577 | 5.968 | 1.340 | PASS |

Остальные проверки того прогона (без RPS): pic still в других режимах — PASS, pic video — PASS, админка `GET /` — HTTP 200.

## Условия замера

- Очередь: SQLite `data/bot.db`, живые оркестратор и воркеры (`python -m app.*`).
- Текст: модели по очереди, без одновременной нагрузки на GPU.
- Mood / pic still: после текста, CPU.
