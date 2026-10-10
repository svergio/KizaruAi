# Рантайм (`var/`)

Логи, временные файлы и служебные скрипты ночных прогонов. Корпус, `bot.db` и pic import/export лежат в `data/`.

| Путь | Назначение |
|------|------------|
| `var/logs/` | Логи процессов бота и воркеров (`start_all.ps1` пишет сюда `<имя>.out.log` и `start_all_pids.txt`) |
| JSON-строки сервисов | stdout/stderr процессов с `core/logging_setup.py` (`service`, `worker_kind`, `job_id`); генерация — одна строка `kizaru.generation` после complete/fail |
| `var/tmp/` | Временные дампы (проверки корпуса — `python -m training.check_dataset`, не скрипты здесь) |
| `var/*.log`, `var/*_status.json`, `var/*_pid.txt`, `var/*_mlflow_run.json` | Логи и статусы конкретных прогонов обучения (артефакты, не конфигурация) |
| `var/jef_data_notes.md` | Заметки, которые пишет `training/scripts/build_jef_dataset.py` |
| `var/night_train_*`, `var/deepseek_*.py`, `var/mlflow_backfill_deepseek_v2.py`, `var/pipeline_monitor.py` | Ops-скрипты ночных прогонов (цепочка обучения, няньки, переход SFT → RL, дозаливка в MLflow) |

Что делает каждый ops-скрипт — [docs/training.md](../docs/training.md#ops-скрипты-в-var). Скрипты писались под конкретные прогоны; перед повторным запуском задайте `KIZARU_SFT_PID`, а для `deepseek_cutover_15k_rl.py` обязательно `KIZARU_SFT_MLFLOW_RUN` — значения по умолчанию у него нет, и без неё скрипт сразу выходит с кодом 2 и ошибкой в stderr (см. там же).
