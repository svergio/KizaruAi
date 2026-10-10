# Метод: agent-pool training (команды агентов)

Коротко: **мультиагентные команды на стене времени бьют одиночное «самообучение» одним агентом** на том же железе и корпусе. Пользователь явно предпочитает этот режим.

## Роли

| Роль | Задача |
|------|--------|
| Coordinator (pool) | GPU-mutex, порядок раундов, бенч, копирование победителя, RESULTS. Не один mega-agent «сделай всё». |
| DatasetFixer | Один раз: freeze train/test, sanity (нет overlap, не JSON-хвосты). |
| Team agent | Архитектура/гиперпараметры в рамках слота, срез train. |
| Lab runner | Гоняет `train.py` / `train_lm.py`, рестартит внутри своих 90 мин. |

Координатор = **пул оркестрации**, а не единственный агент с полным контекстом всех логов.

## Что сработало в промптах

1. **GPU mutex** — один train-процесс на GPU; текстовые Discord-воркеры не поднимать до RESULTS (бот + orchestrator можно оставить).
2. **Команды сами крутят гиперпараметры** внутри wall clock; якоря из README — старт, не клетка.
3. **Бенч общий и замороженный** — CE + фиксированные сиды; tiebreak `gap < 0.02` → кириллица/повторы.
4. **Не спамить `notify_on_output`** на длинных трейнах — иначе координатор тонет в шуме; статус через лог/score файлы.
5. **Слоты жёсткие** — char остаётся char, word остаётся word; transformer/gpt этой ночью без `--level word`.

## Промпт-паттерны (чеклист)

Цитаты/формулировки, которые стоит копировать в новые ночные промпты:

1. **Outcome + absence (sleep → coordinator)** — исход: слот закрыт / score готов; отсутствие: не поднимать Discord text-workers и не «доделывать всё самому». После wall clock — **sleep / handoff → coordinator**, не mega-followup.
2. **Hard resource first (1 GPU)** — в начале промпта: «одна GPU, два train параллельно запрещены»; mutex раньше архитектуры.
3. **Decompose roles; forbid mega-agent** — Coordinator / DatasetFixer / Team / Lab; явно: «не один агент „сделай всё“».
4. **Freeze shared test** — `experiments/shared/test.txt` заморожен до раунда 1; команды режут только свой train-срез.
5. **2 teams × N → bench → checkpoints** — на модель: A, B, общий бенч, winner → `checkpoints/*.pt`.
6. **90 per team / 90+90 duel** — дуэль = **90+90**, не «90 на двоих»; каждый агент ест свой бюджет рестартами.
7. **Autonomy inside wall** — внутри своих 90 мин команда сама крутит lr/batch/arch-якоря; координатор не микроменеджит каждый рестарт.
8. **Plan → Build approve beats** — сначала Plan (формат/роли/mutex), апрув пользователя, потом Build/ночь. Апрувленный план бьёт спонтанный mega-run.
9. **Correct mega-agent drift («как по пулу»)** — если агент снова тянет весь лог/все слоты в один контекст: поправка **«как по пулу»** — вернуть оркестрацию ролей, не «один герой».
10. **User preference** — пул команд на стене времени предпочтительнее одиночного self-training на том же железе; так и оставлять для тюнинга текстовых чекпоинтов.

## Почему лучше self-training

- Две независимые стратегии на одну модель → разведка гиперпараметров параллельна по смыслу (последовательно по GPU).
- Координатор не переписывает train-код «под эксперимент» и не держит весь лог в одном контексте.
- Победитель отбирается объективным test CE, а не «агенту показалось, что loss упал».
- Предпочтение пользователя: **агенты/команды обучают лучше, чем один self-трейн**.

Self-training (один агент крутит одну модель всю ночь) оставлять для разведки архитектуры; для слотов бота — **дуэли команд**.

## Cursor: пул vs design loop

Расширение для пайплайнов (не только ночные дуэли): роли A0–A4, GPU mutex, gate Jeff → SFT → RL, и цикл User↔Agent (уточнения → план → approve → фазы) — в [cursor-training-workflows.md](cursor-training-workflows.md).

Паттерны выше (mutex, decompose roles, Plan → Build approve, no mega-agent) остаются каноном; sibling doc привязывает их к Jev/DeepSeek-фазам и к сессионному design loop.

## Как поднимать снова

См. [night-contest.md](night-contest.md). Команды обучения и бенча (из корня проекта):

```bash
python -m training.train --level char --corpus experiments/shared/train.txt --out checkpoints/kizaru.pt
python -m training.train_lm --arch transformer --level word --corpus experiments/shared/train.txt --out checkpoints/transformer.pt
python experiments/bench.py checkpoints/transformer.pt
```

Все флаги — [../training.md](../training.md).
