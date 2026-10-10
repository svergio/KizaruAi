# Фикстуры / тестовые данные

Большие и lyric-данные **не в git** — создайте локально (см. [docs/install.md](../../docs/install.md)).

| Путь | Для чего |
|------|-----|
| `data/corpus/messages_fallback.csv` | fallback-корпус для `prepare_data` (локально) |
| `data/import/` | pic CLI / Discord media smoke (gitignore) |
| `experiments/shared/train.txt`, `test.txt` | замороженные сплиты (gitignore; метаданные `split_ids.json` / `seeds.json` в git) |

Сюда можно класть маленькие синтетические файлы для тестов, если не хочется трогать runtime `corpus/` / `import/`.
