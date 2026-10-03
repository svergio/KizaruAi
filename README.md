# KizaruAi

Char-level LSTM со стрима 2021. Обучение пишет файл модели, бот этот файл только читает.

`notebooks/` — ноутбуки, на которых учился предсказывать. Их не трогать как учебный след.

## Ноутбуки

- `notebooks/определенее настроение).ipynb` — первый numpy-RNN, бинарное настроение
- `notebooks/text_rnn.ipynb` — char-LSTM, следующая строка, в конце `torch.save` в `.pt`

## Модель файлом

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python train.py --out checkpoints/kizaru.pt --epochs 2000
python generate.py checkpoints/kizaru.pt --start "Я" --lines 4
```

`train.py` делает `torch.save(...)` в один `.pt`: веса, словарь символов, размер LSTM. Этот файл можно отдать боту как есть.

```bash
export DISCORD_TOKEN=...
python bot.py checkpoints/kizaru.pt
```

В Discord две команды:

- `AiTalk Ночью` или `!talk cousin` — дописывает трек
- реплай на сообщение и `AiMood` или `!mood` — настроение того сообщения

Настроение — тот же бинарный RNN из ноутбука (`позитив` / `негатив`). Файл отдельно:

```bash
python train_mood.py --out checkpoints/mood.pt
python bot.py checkpoints/kizaru.pt --mood checkpoints/mood.pt
```

## Тесты

```bash
python -m unittest tests/test_textrnn.py
```

Тексты песен качаются локально и в git не попадают. `legacy/messages.csv` — старый форумный корпус, если Hugging Face недоступен.
