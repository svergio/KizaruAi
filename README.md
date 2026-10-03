# KizaruAi

Восстановленный проект 2021 года. Основные модули к тому моменту уже были сделаны: генерация текста Кизару, определение настроения и Discord-бот. В git в 2022 попал только неполный черновик. Здесь эти модули снова собраны и запускаются.

## Что уже было

- `kizaru/model.py` — char-level LSTM `TextRNN` со стрима «Пишем нейронку» (июль 2021): следующий символ, потом строка
- `kizaru/mood.py` — RNN настроения из ноутбука `определенее настроение).ipynb`: позитив учится на позитиве, негатив на негативе
- `bot.py` — команды, которые и задумывались: дописать трек, по реплаю сказать настроение, собрать пикчу
- `kizaru/pic.py` — PNG в стикеры. Режим 1: линия, квадрат, серая градация, наложение, блок 2x2 на стикер. Режим 2: блок 4x4, средний цвет против среднего цвета стикера из словаря
- `notebooks/` — учебные ноутбуки. В `определенее настроение).ipynb` настроение считает свой `LSTM2`. В `исследование пикчи.ipynb` рисуются тестовые картинки для обоих режимов стикеров

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
- `AiPic 1` или `AiPic 2` с картинкой — пикча. 1 это лайн, 2 это простой режим как на старом скрине. Заголовок `Пикча:`

```bash
python pic.py image.png --mode 2 --cells 36
python pic.py image.png --mode 1 --cells 32
```

## Тесты

```bash
python -m unittest tests/test_textrnn.py tests/test_mood.py tests/test_pic.py
```

Тексты песен качаются локально и в git не попадают. Если Hugging Face недоступен, берётся `legacy/messages.csv`.
