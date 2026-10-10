# Зонд Unicode 2×2 — мягкие глифы для details

Дата прогона: 2026-10-09. Скрипт: [`notebooks/research/probe_unicode_quads.py`](../../notebooks/research/probe_unicode_quads.py).  
Артефакты прогона удалены 2026-10-09 (пересоздаются запуском скрипта).  
Таблица блоков: [symbl Unicode table](https://symbl.cc/ru/unicode-table/#undefined-45).

## Зачем

В pic-ячейке 8×8 логика — матрица `[a][b] / [c][d]`. Сейчас:

- **simple / shade** — один mean по окну → мало ступеней серого;
- **line** — резкий паттерн (ink vs empty), `LINE_CONTRAST≈0.55`.

Мелкий перепад (типа 10% vs 40%) не линия и не плоский фон — детали сливаются. Нужен словарь **soft 2×2**: любые `[a,b,c,d]` со средним spread, не только шахматка.

## Метод

1. Скан assigned-блоков (не весь Han/emoji): Box Drawing, Block Elements, Geometric Shapes (+ Extended), Misc Symbols, Dingbats, Braille, Misc Arrows, Supplemental Arrows-C, Legacy Computing.
2. Шрифт: первый, у которого маска не пустая и **не совпадает с `.notdef`** (fingerprint на `U+FFFE` / `U+FFFF` / `U+1FFFE`).
3. Рендер 64×64 → crop ink bbox → mean TL/TR/BL/BR.
4. Класс по `spread = max-min`:
   - `flat` — spread &lt; 0.08
   - `soft` — 0.08 … 0.45
   - `line` — ≥ 0.45 или сильный одиночный выброс
5. Soft дедуп L2 &lt; 0.08; паттерны рядом с текущим [`line_patterns.json`](../../core/image/assets/line_patterns.json) в soft не тащим.

## Результат прогона

Шрифт почти всего: **Segoe UI Symbol** (`seguisym.ttf`). Unifont / Noto Symbols 2 не нашлись ни среди системных шрифтов, ни в `data/fonts/`.

| Метрика | Число |
|---------|-------|
| codepoints scanned | 1856 |
| measured (real ink) | 1464 |
| skipped tofu (.notdef) | 392 |
| flat / soft / line | 484 / 826 / 154 |
| soft after dedup | **255** |
| line-like kept | 124 |
| existing line_patterns | 111 |

Soft shapes (после дедупа):

| shape | n |
|-------|---|
| half_h | 70 |
| L | 68 |
| half_v | 39 |
| checker | 28 |
| single_quad | 28 |
| other | 22 |

Примеры soft (codepoint → `[tl,tr,bl,br]`, spread):

| cp | pat (округл.) | spread | смысл |
|----|---------------|--------|--------|
| U+2BC6 | 0.69 / 0.69 / 0.24 / 0.28 | 0.45 | верх светлее низа |
| U+2710 | 0.11 / 0.53 / 0.54 / 0.10 | 0.45 | диагональный бленд |
| U+251C | 0.46 / 0.04 / 0.48 / 0.08 | 0.44 | half_v (box tee) |
| U+266A | 0.10 / 0.41 / 0.52 / 0.11 | 0.42 | «нота» как градиент 2×2 |

Файлы: `soft_patterns.json`, `line_like.json`, `ranked.csv`, `atlas_soft.png`, `atlas_line.png`, `summary.json`.

## Выводы

1. **Soft-словарь живой** — 255 уникальных 2×2 после дедупа; half/L/checker покрыты. Для режима `details` базы хватает.
2. **Серого всё ещё мало в flat** — shade-рампа отдельная проблема; soft закрывает *пространственный* бленд внутри ячейки, не число плоских ступеней.
3. **Дыра: Legacy Computing (`1FB00–1FBFF`)** — пиксельные четверти/восьмушки идеальны под мягкий бленд, но Segoe их почти не рисует (tofu). Имеет смысл кинуть [GNU Unifont](https://unifoundry.com/unifont/) / Noto Sans Symbols 2 в `data/fonts/` и перепрогнать.
4. **Line не заменяем** — 124 line-like + текущие 111 measured; soft рядом с line_patterns отфильтрованы. Резкий край (три квадранта ~10, один ~50) остаётся у line.
5. Следующий шаг (не этот док): режим `details` матчит 2×2 к `soft_patterns.json`; line — при высоком contrast; flat shade — как сейчас.

## Запуск

```bash
python notebooks/research/probe_unicode_quads.py
```
