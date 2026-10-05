# Runtime temp (`var/`)

Каталог создаётся при работе бота/смoke-тестов. В git только этот README; `logs/` и `tmp/` — gitignore.

| Путь | Назначение |
|------|------------|
| `var/logs/` | Логи процессов бота/воркеров |
| `var/tmp/` | Временные дампы (FuncText/FuncRest, RPS JSON, …) |

Корпус, `bot.db` и pic import/export остаются в `data/`. Старые логи в корневом `logs/` можно перенести сюда вручную.
