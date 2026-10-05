# Fixtures / test data

Большие и lyric-данные **не в git** — создайте локально (см. [docs/install.md](../../docs/install.md)).

| Path | Use |
|------|-----|
| `data/corpus/messages_fallback.csv` | fallback корпус / `tests/test_textrnn.py` (local) |
| `data/import/` | pic CLI / Discord media smoke (gitignore) |
| `experiments/shared/train.txt`, `test.txt` | frozen splits (gitignore; metadata `split_ids.json` / `seeds.json` в git) |

Сюда можно класть маленькие синтетические файлы для тестов, если не хочется трогать runtime `corpus/` / `import/`.
