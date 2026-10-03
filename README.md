# KizaruAi

Char-level LSTM, который дописывает следующую строку в стиле Кизару.

В 2021 это был стрим «Пишем нейронку»: свой `TextRNN` на PyTorch, корпус реп-текстов, предсказание следующего символа. В git тогда попал другой черновик — Discord-бот и классификатор настроения. Этот репозиторий восстанавливает генератор со стрима и собирает его в рабочий проект.

## Что здесь

- `kizaru_rnn/` — модель со стрима: Embedding + LSTM(2x128) + Linear
- `prepare_data.py` — скачивает тексты `huggingartists/kizaru`, иначе берёт старый `legacy/messages.csv`
- `train.py` / `generate.py` — обучение и сэмплинг следующей строки
- `bot.py` — команда `AiTalk` / `!talk`
- `legacy/` — исходные файлы 2022 года без секретов

Тексты песен чужие. Скрипт качает их локально и не кладёт в git. Только для личного эксперимента.

## Запуск

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python train.py --epochs 2000 --checkpoint checkpoints/kizaru.pt
python generate.py --start "Я" --lines 4 --temp 0.4
```

На стриме стояло 50 000 эпох, `SEQ_LEN=256`, batch 16, Adam `lr=1e-2`, `amsgrad=True`. Для проверки хватает пары тысяч шагов на CPU.

Бот:

```bash
export DISCORD_TOKEN=...
python bot.py
```

Потом в Discord: `AiTalk Ночью` или `!talk cousin`.

## Тесты

```bash
python -m unittest tests/test_textrnn.py
```
