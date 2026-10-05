# KizaruAi — links hub

| What | Where |
|------|--------|
| Bot README (showcase, demos, commands) | [../README.md](../README.md) |
| Математика / алгоритмы (CE, LSTM, attention, mood, pic) | [methods.md](methods.md) |
| First install (fresh clone) | [install.md](install.md) |
| Licensing (publish vs local) | [licensing.md](licensing.md) |
| Runbook: launch, Genius, train, Discord, pic | [runbook.md](runbook.md) |
| Architecture: tree, worker bus | [architecture.md](architecture.md) |
| Live services: CE, smoke, RPS | [services.md](services.md) |
| Checkpoint CE after training (historical RESULTS) | [../experiments/RESULTS.md](../experiments/RESULTS.md) |
| Training selection notes (research) | [research/night-contest.md](research/night-contest.md) |
| Agent-pool training method | [research/agent-pool-training.md](research/agent-pool-training.md) |
| Word embeddings / Navec vocab | [research/embeddings-vocab.md](research/embeddings-vocab.md) |
| Generation audit: loops, corpus echo, temp | [research/generation-notes.md](research/generation-notes.md) |
| Shared frozen train/test | [../experiments/shared/](../experiments/shared/) |
| Per-team runs | [../experiments/](../experiments/) (`lstm_char`, `lstm_word`, `transformer`, `gpt`) |
| Current checkpoints | [../checkpoints/](../checkpoints/) (`kizaru.pt`, `kizaru_word.pt`, `transformer.pt`, `gpt.pt`) |
| Prev backups | [../checkpoints/prev/](../checkpoints/prev/) |
| Pic / video demos (export, gitignore) | `data/export/` locally (gwyndolin, emoji_sticker, bad_apple, …) |
| Bad Apple (YouTube) | https://youtu.be/yNeDeC8j7tY |
| Notebooks | [../notebooks/README.md](../notebooks/README.md) |
| Data layout | [../data/README.md](../data/README.md) |
| SQLite bus (bot.db) | [../database/README.md](../database/README.md) |
| Runtime var/ (logs, tmp; not in repo) | [../var/README.md](../var/README.md) |
| Test fixtures note | [../data/fixtures/README.md](../data/fixtures/README.md) |
| Core lib | `core/text/`, `core/mood/`, `core/image/`, `core/corpus/`, `core/bus/` |
| Runtime stack | `app/discord/`, `app/orchestrator/`, `app/workers/`, `app/web/`, `app/cli/` |
| Training | `training/` (`python -m training.train`, …) |
| KB dossier (same research) | `E:\.knowledgebase\Проекты\KizaruAi\` |
| Admin panel | `python -m app.web` → http://127.0.0.1:8765/ |
