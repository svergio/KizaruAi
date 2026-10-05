# Licensing — what you can publish

## Project code (MIT)

Application code, configs, and docs in this repository are under [MIT](../LICENSE) (Copyright 2026 KizaruAi), unless a file states otherwise.

You may publish, fork, and redistribute the **code** under MIT terms (keep the copyright notice).

## Local-only — do not redistribute

| Material | Location | Why |
|----------|----------|-----|
| Genius / artist lyrics | `data/corpus/` (`genius_*.json`, `kizaru_lyrics.txt`, legacy HF dumps) | Third-party copyright; [Genius](https://genius.com) ToS; not owned by this project |
| Fallback chat export | `data/corpus/messages_fallback.csv` | Third-party text |
| Frozen train/test splits | `experiments/shared/train.txt`, `experiments/shared/test.txt` | Derived from lyrics; rebuild locally after you have corpus |
| Runtime DB / queues | `data/bot.db` (+ WAL/SHM) | Local state, may contain user content |
| Pic inputs/outputs | `data/import/`, `data/export/` | User/media content |
| Logs / temp | `logs/`, `var/logs/`, `var/tmp/` | Operational noise, may contain secrets |
| Secrets | `.env` | Tokens (`DISCORD_TOKEN`, `GENIUS_ACCESS_TOKEN`, etc.) |

See also the **Third-party content** section in [LICENSE](../LICENSE).

## OK to publish (when present)

| Material | Location | License / notes |
|----------|----------|-----------------|
| Checkpoints | `checkpoints/**/*.pt` | Weights trained on local corpus — **you** are responsible for not shipping lyric text inside checkpoints; weights themselves are project artifacts |
| Navec embeddings pack | `data/embeddings/navec_hudlit/` (minus `.cache/`) | **MIT** — see `data/embeddings/navec_hudlit/LICENSE.txt`; upstream [BorisTM/natasha_navec_hudlit_v1_12B_500K_300d_100q](https://huggingface.co/BorisTM/natasha_navec_hudlit_v1_12B_500K_300d_100q) |
| Metadata | `experiments/shared/README.md`, `split_ids.json`, `seeds.json` | Short generation seeds in `seeds.json` are project fixtures; full lyric files stay local |

## Genius API / scraping

Fetching lyrics is for **local research and training** on your machine. This repo does not grant rights to republish lyrics on Hugging Face, Kaggle, or public datasets. Use `training.prepare_data` / orchestrator only with tokens and rate limits you are allowed to use.

## Summary

- **Publish:** MIT source, docs, optional `.pt` checkpoints, Navec files under `data/embeddings/` (without HF `.cache/`).
- **Keep local:** corpus, lyric-derived splits, DB, imports/exports, logs, `.env`.
