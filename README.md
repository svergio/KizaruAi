# KizaruAi

Discord-бот по CIS-rap: **text** (LSTM / Transformer / GPT / DeepSeek), **mood** (две оси) и **pic** (символьная мозаика из картинок и видео).

Лицензия: код — [LICENSE](LICENSE) (MIT); что можно выкладывать публично — [docs/licensing.md](docs/licensing.md). Корпус (`data/corpus/`) в репозиторий не входит.

## Первый запуск

Свежий clone — без корпуса и часто без `.pt`. Полный чеклист: **[docs/install.md](docs/install.md)**. Кратко: venv → `prepare_data` (или Genius) → чекпоинты → процессы ниже.

```bash
python -m pip install -r requirements.txt
python -m training.prepare_data

set DISCORD_TOKEN=...
python -m app.orchestrator --db data/bot.db
python -m app.workers.lstm checkpoints/kizaru.pt --db data/bot.db
python -m app.workers.lstm_word checkpoints/kizaru_word.pt --db data/bot.db
python -m app.workers.transformer checkpoints/transformer.pt --db data/bot.db
python -m app.workers.gpt checkpoints/gpt.pt --db data/bot.db
python -m app.workers.deepseek checkpoints/deepseek.pt --db data/bot.db
python -m app.workers.mood checkpoints/mood.pt --db data/bot.db
python -m app.workers.still --db data/bot.db
python -m app.workers.video --db data/bot.db
python -m app.discord --db data/bot.db
```

Админка: `python -m app.web` → http://127.0.0.1:8765/

Полный деплой, Genius, train, флаги, инвайт Discord: **[docs/runbook.md](docs/runbook.md)**.

CLI без Discord (та же очередь):

```bash
python -m app.cli.generate --arch lstm --start "Я" --db data/bot.db
python -m app.cli.image still couple2.jpg --mode line
```

## Команды `/ai`

Префикса нет — только slash. Бот синкает команды на гильдии. Исходник: `app/discord/__main__.py`.

### `/ai text`

Продолжение текста (next-token LM). Воркер должен быть запущен, иначе бот отвечает сразу «модель не запущена».

| Параметр | Обязательный | Значения / диапазон | По умолчанию | Смысл |
|----------|--------------|---------------------|--------------|--------|
| `model` | да | `lstm` \| `lstm-word` \| `transformer` \| `gpt` \| `deepseek` \| `bert-gen` | — | Какой checkpoint/воркер семплирует |
| `start` | нет | любая строка | `Я` | Сид: модель продолжает от этой строки |
| `temp` | нет | 0.1…2.0 | по модели | Температура софтмакса |
| `lines` | нет | 1…12 | `6` | Сколько непустых строк вывести |
| `length` | нет | 40…1200 символов, у `lstm-word` 8…200 слов | по модели | Бюджет генерации; у `bert-gen` — длина окна |
| `iters` | нет | 1…200 | `32` | Только `bert-gen`: число проходов Gibbs |

- `lstm` — char LSTM (`checkpoints/kizaru.pt`)
- `lstm-word` — word LSTM; OOV → `<unk>` (`checkpoints/kizaru_word.pt`)
- `transformer` — causal Transformer, post-norm (`checkpoints/transformer.pt`)
- `gpt` — causal GPT-block, pre-norm (`checkpoints/gpt.pt`)
- `deepseek` — MLA + RMSNorm/SwiGLU (`checkpoints/deepseek.pt`, pending retrain)
- `bert-gen` — двусторонний MLM + Gibbs (`checkpoints/bert_gen.pt`, pending retrain)

Пустые `temp` / `lines` / `length` берут дефолт модели, значения вне диапазона подрезаются. Дефолтные температуры разные, потому что модели по-разному лупятся:

| `model` | `temp` | `length` | Что будет, если крутить |
|---------|--------|----------|--------------------------|
| `lstm` | `0.4` | `360` символов | повтор строки не появляется ни на 0.4, ни на 0.8 |
| `lstm-word` | `0.8` | `72` слова | узкое место не температура, а OOV → `<unk>` |
| `transformer` | `0.9` | `360` символов | `0.7` даёт повтор строки в 53% ответов, `1.1` ломает каждое пятое слово |
| `gpt` | `1.0` | `360` символов | `0.8` повторял строку в половине ответов |

Правило: **ниже температура — больше повторов строк, выше — больше нерусских слов.** Хочешь ровный куплет без рисков — оставь дефолт; хочешь дичь — `temp:1.2`. Замеры: [docs/research/generation-notes.md](docs/research/generation-notes.md).

CLI-аналог: `python -m app.cli.generate --arch gpt --start "Доброе утро Америка" --temp 0.9 --lines 8 --length 500`.

### `/ai mood`

Классификация настроения, не генерация. Нужен `app.workers.mood`.

| Параметр | Обязательный | Значения | По умолчанию | Смысл |
|----------|--------------|----------|--------------|--------|
| `text` | нет* | любая строка | `""` | Вход для RNN `vector2` |

\* Если `text` пустой — берётся текст из reply на сообщение. Без text и без reply бот просит указать источник.

Выход: две оси (sigmoid 0…1) — злой↔добрый и грустный↔веселый, плюс `vector=(x, y)`.

### `/ai pic`

Символьная картинка из вложения. Не нейросеть: яркость → глифы. Нужны still/video-воркеры по режиму.

| Параметр | Обязательный | Значения / диапазон | По умолчанию | Смысл |
|----------|--------------|---------------------|--------------|--------|
| `mode` | да | `line` \| `simple` \| `cli` | — | Куда и как рендерить |
| `file` | да | PNG / JPG / GIF / видео | — | Вложение Discord |
| `scale` | нет | 1…32 | `2` | Только `cli`: px источника на квадрант smile |
| `cells` | нет | 8…320 | `320` | Только `cli`: ширина сетки в символах |

| `mode` | Ответ | Что делает | `scale` / `cells` |
|--------|-------|------------|-------------------|
| `line` | текст, 31 колонка | глифы из `line_patterns.json` (паттерны 2×2 яркости) | игнорируются |
| `simple` | текст, 31 колонка | 11-тоновая блок-рампа | игнорируются |
| `cli` | файл PNG/GIF | тайл 8×8, line-пайплайн | используются |

CLI-аналог: `python -m app.cli.image still <path> --mode line --scale 2 --cells 320`.

### `/ai help`

Краткая справка в чате (без параметров).

## Pic: scale, cells, upscale

Код: `core/image/ascii.py`. Константы: `OUTPUT_SMILE_PX = 8`, `CLI_DEFAULT_SCALE = 2`, `CLI_DEFAULT_CELLS = 320`.

### Как работает алгоритм

1. Кадр → яркость (luma), лёгкий stretch контраста.
2. Строится сетка: `cells` колонок; число строк по aspect ratio кадра.
3. Каждая ячейка («smile») смотрит на окно источника размером `(2 × scale) × (2 × scale)` пикселей — четыре квадранта по `scale × scale`. По средним яркостям квадрантов выбирается ближайший глиф из словаря line-паттернов (матрица 2×2).
4. В PNG/GIF глиф рисуется тайлом **8×8** px.

Две полезные величины:

- ширина выхода: `W_out = cells × 8`
- ширина источника, при которой на каждый квадрант хватает «живых» пикселей без растягивания: `W_need = cells × 2 × scale`

Дефолт CLI (`cells=320`, `scale=2`): `W_need = 1280`, `W_out = 2560`.

### Upscale

В API (`render_png` / `resolve_frame`) флаг `upscale` по умолчанию `True`. Discord `/ai pic mode:cli` и CLI-воркеры **всегда** вызывают рендер с `upscale=True`; отдельного slash-параметра / `--upscale` в CLI нет.

| `upscale` | Если `W_src < W_need` |
|-----------|------------------------|
| `True` | сетка `cells` сохраняется; источник nearest-up до растра smile |
| `False` | `cols` сжимается до `W_src // (2 × scale)` |

Практически: если источник уже ≈ `W_need`, пиксели не изобретаются. Если сильно меньше — лучше снизить `cells` или `scale`, а не надеяться на nearest-up.

### Подбор под разрешение

Правило большого пальца: `cells × 2 × scale ≈ ширина кадра`.

| Ширина кадра ≈ | scale | cells | W_need | W_out | Заметка |
|----------------|-------|-------|--------|-------|---------|
| 480 | 1–2 | 80–120 | 160–480 | 640–960 | не гони `cells=320` |
| 1080 | 2 | 160–200 | 640–800 | 1280–1600 | удобно для Discord `cli` |
| **1800** | **3** | **300** | **1800** | **2400** | scale 3 без upscale: `cells = W / (2×scale)` |
| 1800 | 3 | 320 | 1920 | 2560 | дефолтные cells; лёгкий nearest-up |
| 1920 | 2 | 320 | 1280 | 2560 | дефолт CLI |
| 1920 | 3 | 320 | 1920 | 2560 | максимум деталей под Full HD |

Кратко для **~1800p**: бери **`scale=3`, `cells=300`** — тогда `W_need` совпадает с шириной кадра и upscale не нужен. Если оставить дефолт `cells=320` при `scale=3`, `W_need=1920` — Discord/CLI чуть подтянут кадр nearest-up.

Discord `cli` режет файлы > **8 MB**. При `cells=320` (`W_out=2560`) PNG часто не влезает — снижай `cells` (например 160).

Глубже (маска line, quadtree, simple-рампа): [docs/methods.md#pic](docs/methods.md#pic).

## Модели (архитектура)

Текущие чекпоинты после обучения; цифры из checkpoints / [docs/methods.md](docs/methods.md) / [experiments/RESULTS.md](experiments/RESULTS.md).

| Слот | Тип | Размер / конфиг | Vocab | Params | Test CE | Checkpoint |
|------|-----|-----------------|-------|--------|---------|------------|
| **lstm** | Char LSTM | emb/hidden **768**, **2** слоя, BPTT 128, seq 512 | ~814 символов | ~10.7M | **1.335** | `checkpoints/kizaru.pt` |
| **lstm-word** | Word LSTM | emb/hidden **512**, **2** слоя, seq 128, bptt 64, `min_freq=5` | ~57k слов | ~63M | **3.705**\* | `checkpoints/kizaru_word.pt` |
| **transformer** | Causal Transformer | `d_model=384`, `n_layers=8`, `n_heads=8`, `block_size=384`; post-norm Vaswani (ReLU) | ~814 char | ~**15M** | **1.166** | `checkpoints/transformer.pt` |
| **gpt** | Causal GPT | та же ёмкость **384/8/8**, block 384; pre-norm GPT-block (GELU) | ~814 char | ~**15M** | **1.171** | `checkpoints/gpt.pt` |
| **deepseek** | MiniDeepSeek (MLA) | ориентир **384/8/8**, block 384; MLA + RMSNorm + SwiGLU, без MoE | ~814 char | ~15–50M | — (pending) | `checkpoints/deepseek.pt` |
| **bert-gen** | BERT MLM + Gibbs | bidirectional, `[MASK]`, block 256; не causal | ~814 + mask | ~15M | — (pending) | `checkpoints/bert_gen.pt` |
| **mood** | RNN → 2 оси | `kind=vector2`, sigmoid | — | малая | — | `checkpoints/mood.pt` |

\* Word CE на другой шкале — с char не сравнивать. Char CE сравнимы между собой.

Обучение текстовых моделей: next-token LM, лосс CrossEntropy, Adam. Не RL / не RLHF. Как учить и куда класть `.pt` — [docs/runbook.md](docs/runbook.md).

## Математика / алгоритмы

Полный разбор: **[docs/methods.md](docs/methods.md)**. Ниже — ядро text-слотов коротко.

### Next-token и CrossEntropy

Корпус режется на токены. На каждом шаге модель учится предсказывать следующий: вероятность \(p_\theta(x_t \mid x_{<t})\). Лосс — минус лог этой вероятности для правильного токена (в коде `CrossEntropyLoss` по логитам словаря). Метрика после обучения — **test CE** на замороженном `experiments/shared/test.txt`. Меньше CE → лучше на этом тесте.

### Char vs word

| Режим | Токен | Vocab | Следствие |
|-------|-------|-------|-----------|
| **lstm / transformer / gpt / deepseek** | символ | ~814 | CE сравнимы; сиды «как есть» |
| **lstm-word** | слово | ~57k (`min_freq=5`) | своя шкала CE; OOV → `<unk>` |

### LSTM

Char и word — одна схема, разный токенайзер: эмбеддинг → LSTM → softmax по словарю. Скрытое состояние несёт контекст слева направо; BPTT режет длинную последовательность на окна.

### Transformer / GPT / DeepSeek

Transformer и GPT — causal self-attention на окне `block_size` (только прошлое). Attention: \(\mathrm{softmax}(QK^\top / \sqrt{d_k} + M)\,V\), где \(M\) — causal-маска.

- **transformer** — post-norm Vaswani-block (ReLU FFN)
- **gpt** — pre-norm GPT-block (GELU FFN)
- **deepseek** — MLA (compressed Q/KV) + RMSNorm + SwiGLU; без MoE ([docs/methods.md#deepseek-minideepseek](docs/methods.md#deepseek-minideepseek))

Тот же next-token + CE; меняется блок attention / нормализации / активации.

### Сводка режимов и CE

| Режим | Идея | Test CE | Полный разбор |
|-------|------|---------|---------------|
| **lstm** | Char LSTM; ~10.7M (768/768×2) | **1.335** | [docs/methods.md#lstm-char](docs/methods.md#lstm-char) |
| **lstm-word** | Word LSTM; ~63M; OOV → `<unk>` | **3.705** (своя шкала) | [docs/methods.md#lstm-word](docs/methods.md#lstm-word) |
| **transformer** | Causal attention, post-norm; 384/8/8, ~15M | **1.166** | [docs/methods.md#transformer](docs/methods.md#transformer) |
| **gpt** | Pre-norm GPT-block; 384/8/8, ~15M | **1.171** | [docs/methods.md#gpt](docs/methods.md#gpt) |
| **deepseek** | MLA + RMSNorm/SwiGLU; pending checkpoint | — | [docs/methods.md#deepseek-minideepseek](docs/methods.md#deepseek-minideepseek) |
| **mood** | RNN → 2 sigmoid-оси | — | [docs/methods.md#mood](docs/methods.md#mood) |
| **pic** | Яркость → глифы; cli = PNG/GIF | — | [docs/methods.md#pic](docs/methods.md#pic) |

История отбора чекпоинтов: [experiments/RESULTS.md](experiments/RESULTS.md), [docs/research/night-contest.md](docs/research/night-contest.md).

## Демо

### Bad Apple (pic / line)

[![Bad Apple — символьная сетка](https://img.youtube.com/vi/yNeDeC8j7tY/hqdefault.jpg)](https://youtu.be/yNeDeC8j7tY)

[Смотреть на YouTube](https://youtu.be/yNeDeC8j7tY)

### Text — живые прогоны CLI

`python -m app.cli.generate --arch <model> --start "…" --db data/bot.db`  
(оркестратор + text-воркеры). Word-модель честно сыплет `<unk>` на OOV.

#### Сид: `Я ебал игру как будто это шлюха`

```text
# lstm
Я ебал игру как будто это шлюха (Ха)

Меня в комнате не стоит по первый гангстер

Я снова стану тебя на концерте, как солнце (Ага)

Я не знаю, что в этом сердце только стоит (Да)

Я был под ногами в моих детей (Эй)

Я поднял тебя и не понимаю (А-а)
```

```text
# lstm-word
я ебал игру как будто это шлюха

я сияю всё. брат всё или это точно?

но это не так, так так, чтоб я люблю тебя

я не собираюсь спать, никогда не могу соврать

я всегда в дыму по пути, быстро

моя жизнь не перечитать, а я- я
```

```text
# transformer
Я ебал игру как будто это шлюха

И пусть имени жить в дом

Никто не примет дом (Дом)

По мальчикам твоим придёт (Воу-воу-воу)

Ты всё придёт имени жить в дом

Никто не примет дом
```

```text
# gpt
Я ебал игру как будто это шлюха (У-у)

Она хочет всегда такие расстанутся (Да), всех бывают ночь (Ха-ха)

Я не достал бы просто не знал себе "Спляшу голые сердце (Ха-ха-ха)

Я не достал бы просто не знал себе "Спляшу голые сердце (Цети)

Она хочет всегда такие расстанутся (Да)

Я не достал бы просто не знал себе (Себе)
```

#### Сид: `Денег дохуя будто я альфонс`

```text
# lstm
Денег дохуя будто я альфонский белый

Мне нужен был на ней, но не поменял свой стиль

Я не стану себя со мной на моём сердце в колени

Я не спрашиваю в этом темноте

Я не знаю, я не должен мне в тебе не было

Мой брат старших под колени, но не ставлю тебя
```

```text
# lstm-word
денег дохуя будто я альфонс- чушь

я просто здесь— <unk> и <unk>, ведь я на хуй не <unk>

я пробил её руку, как я продал <unk>

я вас не должен, не знаю, чё будет с нами

от этого головою все ветер, ведь это мания

я не лежу из снов, в стае, в этой суке, я люблю,
```

```text
# transformer
Денег дохуя будто я альфонсол

Мои губы — вы мои руки — пока не поймёт

Мимо последних мы с тобой уйдём дынём

Я увидел в шторм у меня есть в невидежной альфонс, а, а

Сделал в шторм хаос, что легче так близко

Вы ссытый мутят пленный так, слушать придётся нам
```

```text
# gpt
Денег дохуя будто я альфонсирую тропил

В моей лучшей жизни размер, мой кузон (А-а-а)

Мой ночи или я строк бывал на этот странный стайл

Когда написал и коль им я в списке, а ты мой выстрел? (Wha'?)

Прошу, сейчас со мной доверяет с лучшим

В моей лучшей жизни размер, мой выстрел все прошлые
```

#### Сид: `Доброе утро Америка`

```text
# lstm
Доброе утро Американский сон

Меня не видит с последний раз таких больше не верит

Всё так от кого не показалось в сторону

В кармане хотят странных девятой столица

Мой парень — это мой брат, ты не смог понять

Ты моя девочка на полной мамах
```

```text
# lstm-word
доброе утро америка

я помогу, я не хотел дышать( не хочу)

я не понимаю, что я могу( м- м)

останусь я один, но я не собираюсь( не знаю, чё со мной)

я в городе не сказал, что я не смог

и я видел это так, чтобы я снова tags-
```

```text
# transformer
Доброе утро Америка

Так заработал мимо без подруг — это работал кик

Вот и так и самая ночь

Ты так самая ночь

Ты так самая ночь

Ты так самая ночь
```

```text
# gpt
Доброе утро Американ

Чтоб мне не держать по кабелям

Но снова стоит с женой собственной сторонам

И на ней карикале

И на ней карикале

И на ней карикале
```

### Mood

Вход: `я злой и весёлый`

```text
# /ai mood text:я злой и весёлый
Настроение: злой 0.00 | веселый 1.00
vector=(0.00, 1.00)
я злой и весёлый
```

## Архитектура процессов (кратко)

Discord только enqueue; считает оркестратор + воркеры. Шина — SQLite `data/bot.db` (без Redis): `queued` → `assigned` → `running` → `done` / `error`.

```
app/discord  -->  data/bot.db  <--  app/orchestrator
                      ^
                      |
              app/workers/{lstm,lstm_word,transformer,gpt,deepseek,mood,still,video}
```

Полное дерево, `worker_kind`, модули: **[docs/architecture.md](docs/architecture.md)**.

## Сервисы и метрики (кратко)

RPS (8 jobs, одна модель на GPU):

| Сервис | Test CE | Smoke s | RPS |
|--------|---------|---------|-----|
| lstm | 1.335 | 1.04 | 3.38 |
| lstm-word | 3.705 | 0.89 | 3.59 |
| transformer | 1.166 | 1.48 | 1.40 |
| gpt | 1.171 | 1.21 | 1.58 |
| mood | — | 0.54 | 3.83 |
| pic still | — | 0.58 | 5.97 |

Таблицы, метод замера, raw dump: **[docs/services.md](docs/services.md)**.

## Research / Genius / embeddings

Хаб всех ссылок: **[docs/LINKS.md](docs/LINKS.md)**

- Genius / корпус / обучение моделей: [docs/runbook.md](docs/runbook.md)
- История отбора чекпоинтов (research): [docs/research/night-contest.md](docs/research/night-contest.md)
- Agent-pool training: [docs/research/agent-pool-training.md](docs/research/agent-pool-training.md)
- Word embeddings / Navec: [docs/research/embeddings-vocab.md](docs/research/embeddings-vocab.md)
- Разбор генерации (лупы, температура, сверка с корпусом): [docs/research/generation-notes.md](docs/research/generation-notes.md)
- Данные: [data/README.md](data/README.md)
- Ноутбуки / зонды глифов: [notebooks/README.md](notebooks/README.md)
