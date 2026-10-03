# KizaruAi

Восстановленный проект 2021 года. Основные модули к тому моменту уже были сделаны: генерация текста Кизару, определение настроения и Discord-бот. В git в 2022 попал только неполный черновик. Здесь эти модули снова собраны и запускаются.

## Что уже было

- `kizaru/model.py` — char-level LSTM `TextRNN` со стрима «Пишем нейронку» (июль 2021): следующий символ, потом строка
- `kizaru/mood.py` — RNN настроения из ноутбука `определенее настроение).ipynb`: позитив / негатив, в том числе с «не»
- `bot.py` — две команды, которые и задумывались: дописать трек и по реплаю сказать настроение
- `notebooks/` — учебные ноутбуки, на которых эти модели собирались. Их не переписывали заново

`legacy/` — то, что реально лежало в репозитории в феврале 2022: каркас бота, пустой парсер и `messages.csv`.

## Запуск

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python train.py --out checkpoints/kizaru.pt --epochs 2000
python train_mood.py --out checkpoints/mood.pt
python generate.py checkpoints/kizaru.pt --start "Я" --lines 4
```

Обучение по-прежнему пишет готовый файл через `torch.save`. Боту отдаются эти файлы, он сам не учится.

```bash
export DISCORD_TOKEN=...
python bot.py checkpoints/kizaru.pt --mood checkpoints/mood.pt
```

В Discord:

- `AiTalk Ночью` или `!talk cousin` — дописывает трек
- реплай на сообщение и `AiMood` или `!mood` — настроение того сообщения

## Тесты

```bash
python -m unittest tests/test_textrnn.py tests/test_mood.py
```

Тексты песен качаются локально и в git не попадают. Если Hugging Face недоступен, берётся `legacy/messages.csv`.
