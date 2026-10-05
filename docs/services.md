# Services — smoke, CE, RPS

Live queue metrics after functional smokes (local only: dumps under `var/tmp/` at runtime, not in repo).  
Test CE from [experiments/RESULTS.md](../experiments/RESULTS.md) (requires local `experiments/shared/test.txt`).  
RPS: 8 jobs per service, one model at a time on GPU; `rps = 8 / wall_seconds` from first enqueue to last `done`. Text seed for RPS: `Я`.

## Text workers

| Service | Checkpoint | Test CE | Smoke seed | Smoke s | RPS (8 jobs) | 8-job wall s | Smoke |
|---------|------------|---------|------------|---------|--------------|--------------|-------|
| lstm (char) | `checkpoints/kizaru.pt` | 1.335 | Я купил фенизипам… | 1.036 | 3.377 | 2.369 | PASS |
| lstm-word | `checkpoints/kizaru_word.pt` | 3.705 | Я купил фенизипам… | 0.894 | 3.587 | 2.230 | PASS |
| transformer | `checkpoints/transformer.pt` | 1.166 | Я купил фенизипам… | 1.484 | 1.402 | 5.706 | PASS |
| gpt | `checkpoints/gpt.pt` | 1.171 | Я купил фенизипам… | 1.214 | 1.583 | 5.053 | PASS |

Smoke seed full: `Я купил фенизипам для ворка на вск 4`. Worker kinds: `text_lstm`, `text_lstm_word`, `text_transformer`, `text_gpt`.

## Mood / pic (no CE)

| Service | Checkpoint | Smoke input | Smoke s | RPS (8 jobs) | 8-job wall s | Smoke |
|---------|------------|-------------|---------|--------------|--------------|-------|
| mood | `checkpoints/mood.pt` | `я злой и весёлый` | 0.539 | 3.830 | 2.089 | PASS |
| pic still mode1 | — (ascii line) | 32×32 PNG, mode_key `1` | 0.577 | 5.968 | 1.340 | PASS |

Other FuncRest checks (not in RPS table): pic still mode2 PASS, pic cli mode3 PASS, pic video PASS, admin `GET /` HTTP 200 PASS.

## Method

- Queue: SQLite `data/bot.db`, orchestrator + workers live (`python -m app.*`).
- Text RPS: sequential models (no concurrent GPU load across arches).
- Mood / pic still: after text, CPU path.
- Raw RPS dump (local smoke): `var/tmp/rps_metrics.json` at runtime.
