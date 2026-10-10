# Датасеты стихов со схемой рифмы (до RL)

Цель — усилить SFT на **структурированных строфах** (4–8 строк, метки рифмы) до любого RL. RL и LLM-сервер — потом.

## Единица (целевая)

```text
lang / form / lines / rhyme / mood / seed
body: N строк
```

## Источники

| Датасет | Размер | Подходит под схему | Лицензия | Локальный путь |
|---------|--------|--------------------|----------|----------------|
| **Koziev RIFMA** | 5121 строфа | `poem_text` + **`rhyme_scheme`** + ударения | MIT | `data/corpus/poetry/schema/rifma_dataset.json` |
| **inkoziev/ArsPoetica** | 8498 стихов | текст + ударения; рифма не размечена | **CC-BY-NC-4.0** | `data/corpus/poetry/schema/arspoetica.json` |
| Chicago rhymedata (EN) | — | схема рифмы | public domain | `data/corpus/poetry/schema/en/rhymedata/english_raw` |
| poetry_form (EN) | — | форма стиха | см. карточку датасета | `data/corpus/poetry/schema/en/poetry_form/` |
| PoetryCorpus | ~17k стихов | только текст | Apache-2.0 | `data/corpus/poetry/classical_poetry.txt` |
| выборка stihi_ru | ~53 МБ | текст, шумно | см. карточку HF | `data/corpus/poetry/stihi_ru_sample.txt` |

**Основной для схемы:** RIFMA (метки совпадают с нашим `rhyme: ABAB|AABB|…`).
**Объём + ударения:** ArsPoetica (NC — только исследования / некоммерческое).
**Уже в LM-корпусе:** классика + выборка stihi (без поля рифмы). Обе качает `training/scripts/download_poetry.py`.

## Импорт

```powershell
python -m training.scripts.import_verse_schema
# только MIT (без NC):
python -m training.scripts.import_verse_schema --skip-ars-nc `
  --out-jsonl data/corpus/poetry/schema/verse_schema_mit.jsonl `
  --out-txt data/corpus/poetry/schema/verse_schema_mit_lm.txt
```

Выход:

- `verse_schema.jsonl` — записи с `form/lines/rhyme/body/seed/…`
- `verse_schema_lm.txt` — плоские строфы для `prepare_subset --poetry`
- `*_mit*` — без ArsPoetica

## Дальше

1. Пересобрать `kizaru_lyrics_ready.txt` через `python -m training.prepare_subset --poetry ...` с `verse_schema_mit_lm.txt` (и полной схемой, если NC допустимо).
2. Подмешивать в следующий SFT / дообучение, не останавливая текущие прогоны.
3. RL — только после SFT со схемами.

## Ссылки

- https://github.com/Koziev/Rifma
- https://huggingface.co/datasets/inkoziev/ArsPoetica
- https://github.com/IlyaGusev/PoetryCorpus
