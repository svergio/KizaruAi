# Общий сплит (заморожен)

Источник: только `data/corpus/genius_*.json`. Huggingartists не входит.

- `train.txt` — пул для обучения (gitignore; пересоберите локально из корпуса).
- `test.txt` — общий hold-out (gitignore). **Не редактировать** копию у себя после сборки.
- `split_ids.json` — id песен, seed `20261005`, 95/5 по песням со стратификацией по артисту.
- `seeds.json` — фиксированные старты генерации для `experiments/bench.py`.
