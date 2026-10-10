"""Сборка обучающего микса из сырых источников по рецепту training/mixes/<name>.yaml.

Порядок: источники -> чистка -> пул (ru_rap, ru_other, en_rap, ru_poetry, en_poetry, other)
-> глобальная дедупликация по нормализованному хешу текста -> сплит train/val/test
по хешу песни внутри пула (до смешивания) -> выборка по долям токенов рецепта.
Данные никогда не повторяются: если пулу не хватает токенов, сборка падает
(или с --allow-short берёт всё, что есть, и пишет предупреждение).

    python -m training.build_mix --mix ru_rap_v1 --dry-run
    python -m training.build_mix --mix ru_rap_v1 --seed 1 --out data/mixes/ru_rap_v1

Выход: <out>/{train,val,test}.jsonl и .txt (для train_lm --corpus),
плюс <split>_<pool>.txt (например val_ru_rap.txt для --val-focus) + report.json.

ru_poetry берётся из полного stihi_ru (poetry/raw/stihi_ru.jsonl.zst) потоково. Разобранные и
почищенные куски стихов кешируются в data/corpus/poetry/cache/ (пересборка при смене исходника
или версии кеша, либо по --rebuild-cache); кеш пишется и при --dry-run.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import json
import os
import random
import re
import sys
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path

from core.corpus.clean import join_songs
from core.corpus.lyrics import prepare_corpus_text
from training.check_dataset import line_script, text_hash
from training.pools import load_ru_rap_exclude_pools, pool_key

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"
POETRY_DIR = CORPUS_DIR / "poetry"
MIXES_DIR = ROOT / "training" / "mixes"
POOLS_DIR = ROOT / "training" / "artist_pools"
POETS_LIST = ROOT / "training" / "artist_pools_poets" / "poets.txt"
RAP_EXCLUDE_LIST = POOLS_DIR / "ru_rap_exclude.txt"  # "slug<TAB>pool": артист уходит в указанный пул
SOLO_RETRY_LIST = ROOT / "training" / "artist_pools_solo_retry" / "06_solo_retry.txt"
DEFAULT_OUT_ROOT = ROOT / "data" / "mixes"
STIHI_ZST = POETRY_DIR / "raw" / "stihi_ru.jsonl.zst"   # полный IlyaGusev/stihi_ru
STIHI_CACHE_DIR = POETRY_DIR / "cache"
STIHI_CACHE_VERSION = 1
STIHI_BATCH = 2000

POOLS = ("ru_rap", "ru_other", "en_rap", "ru_poetry", "en_poetry", "other")  # + пулы из ru_rap_exclude.txt
SPLITS = ("train", "val", "test")

ARTIST_EN_SHARE = 0.70     # артист с >70% латинских строк считается англоязычным
SONG_LAT_DROP_RU = 0.50    # песня с >50% латиницы выкидывается из ru-пулов
SONG_CYR_DROP_EN = 0.50    # песня с >50% кириллицы выкидывается из en-пулов
MIN_SONG_CHARS = 40
POEM_CHUNK_LINES = 16      # в txt-корпусах поэзии нет границ стихов: режем блоками

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Приближение словного токенизатора: слова + пунктуация + переносы строк."""
    return len(TOKEN_RE.findall(text)) + text.count("\n")


def script_counts(text: str) -> Counter:
    cnt = Counter()
    for ln in text.splitlines():
        s = line_script(ln)
        if s:
            cnt[s] += 1
    return cnt


# ---------- артисты -> пул ----------

def _read_list(path: Path) -> list[str]:
    if not path.is_file():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def load_artist_pools(pools_dir: Path = POOLS_DIR) -> dict[str, str]:
    """pool_key -> пул (pool_key из training/pools.py: кириллица сохраняется, снимается только -2/-3). Приоритет при пересечениях: en_rap > ru_poetry > ru_rap > ru_other
    (не-рэп из рэп-списков выносится через ru_rap_exclude.txt).
    ru_rap_exclude.txt применяется поверх всего (assign_pool)."""
    lists = [
        # поп-список первым: в нём есть хвосты дублей рэп-slug (Oxxxymiron-2, Kizaru-2 в «volume fillers»),
        # после pool_key они совпадают с рэперами — рэп-списки должны их перекрыть
        ("ru_other", pools_dir / "ru_pop_rock_chanson.txt"),
        ("ru_rap", pools_dir / "00_existing.txt"),
        ("ru_rap", pools_dir / "ru_expand.txt"),
        ("ru_rap", SOLO_RETRY_LIST),
        ("ru_rap", pools_dir / "ru_rap_extra.txt"),
        ("ru_poetry", POETS_LIST),
        ("en_rap", pools_dir / "en_rap.txt"),
    ]
    mapping: dict[str, str] = {}
    for pool, path in lists:  # позднее перезаписывает раннее -> приоритет у конца списка
        for slug in _read_list(path):
            key = artist_key(slug)
            if key:  # пустой ключ не мапим никогда
                mapping[key] = pool
    return mapping


def artist_key(slug: str) -> str:
    """Ключ артиста: pool_key из training/pools.py (снимает только суффикс дубля -2/-3, 25-17 не трогает)."""
    return pool_key(slug)


def load_rap_exclude(log=print) -> dict[str, str]:
    """pool_key артиста -> пул для исключённых из ru_rap (ru_pop, ru_rock, uk_other, kz_uz_other...).
    Источник — training/artist_pools/ru_rap_exclude.txt через training.pools.load_ru_rap_exclude_pools
    (пул не указан -> ru_other). Артист не выкидывается, а уходит в этот пул (пулы создаются по ходу)."""
    pools = load_ru_rap_exclude_pools()
    if not pools:
        log(f"WARN: {RAP_EXCLUDE_LIST.name} пуст или отсутствует — исключения из ru_rap не применяются")
    return dict(pools)


def base_pool(source: str, artist_slug: str, artist_map: dict[str, str]) -> str | None:
    """Пул по источнику и спискам артистов (без ru_rap_exclude.txt — он применяется отдельно)."""
    if source == "huggingartists":
        return "ru_rap"
    if source in ("ru_poetry_txt",):
        return "ru_poetry"
    if source in ("en_poetry_form", "en_rhymedata"):
        return "en_poetry"
    key = artist_key(artist_slug)
    return artist_map.get(key) if key else None


def assign_pool(source: str, artist_slug: str, artist_lat_share: float,
                artist_map: dict[str, str], rap_exclude: dict[str, str]) -> str:
    """Итоговый пул артиста. Явный пул из ru_rap_exclude.txt важнее списков и письма."""
    if source in ("genius", "lyricsgenius", "huggingartists"):
        explicit = rap_exclude.get(artist_key(artist_slug))
        if explicit:
            return explicit
    return resolve_pool(base_pool(source, artist_slug, artist_map), artist_lat_share)


def resolve_pool(base: str | None, artist_lat_share: float) -> str:
    """Пул артиста с учётом письма. Неизвестный артист: кириллица -> ru_other, иначе other."""
    en = artist_lat_share > ARTIST_EN_SHARE
    if base is None:
        return "other" if en else "ru_other"
    if base == "ru_rap" and en:
        return "en_rap"
    if base == "ru_poetry" and en:
        return "en_poetry"
    if base == "ru_other" and en:
        return "other"
    return base


def song_filter(pool: str, cnt: Counter) -> str | None:
    """Причина выброса песни или None. Одиночные английские строки в ru-песнях остаются."""
    n = sum(cnt.values())
    if not n:
        return "no_letters"
    if cnt["arab"]:
        return "arabic_script"
    if pool.startswith("ru_") and cnt["lat"] / n > SONG_LAT_DROP_RU:
        return "latin_in_ru"
    if pool.startswith("en_") and cnt["cyr"] / n > SONG_CYR_DROP_EN:
        return "cyrillic_in_en"
    return None


_TAG_RE = re.compile(r"\[[^\]\n]*\]|\([^)\n]*\)")  # [Куплет], (Live), (Remix), (эй) — в т.ч. отдельной строкой


def split_hash(text: str) -> str:
    """Ключ сплита: text_hash (нижний регистр, без пунктуации и пробелов) после удаления [..] и (..).
    Один и тот же текст (фит, live, ремикс с другим тегом) всегда попадает в один сплит."""
    return text_hash(_TAG_RE.sub(" ", text))


def split_of(h: str, val: float, test: float) -> str:
    """Детерминированный сплит по хешу песни (не зависит от seed и состава микса)."""
    x = int(h[:8], 16) / 0xFFFFFFFF
    if x < test:
        return "test"
    if x < test + val:
        return "val"
    return "train"


# ---------- источники ----------

def _warn(msg: str) -> None:
    print(f"WARN: {msg}", file=sys.stderr, flush=True)


def iter_genius(corpus_dir: Path, stats: Counter | None = None):
    """Кэши Genius. Пропускает (с WARN и счётчиком в stats) файлы с "complete": false — их ещё
    пишет загрузчик — и битые/пустые JSON. Старые кэши без поля complete считаются полными."""
    stats = stats if stats is not None else Counter()
    for fp in sorted(corpus_dir.glob("genius_*.json")):
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            stats["skipped_broken"] += 1
            _warn(f"пропускаю битый {fp.name}: {exc}")
            continue
        if not isinstance(payload, dict) or not payload:
            stats["skipped_broken"] += 1
            _warn(f"пропускаю {fp.name}: пустой или не объект JSON")
            continue
        if payload.get("complete") is False:
            stats["skipped_incomplete"] += 1
            _warn(f"пропускаю {fp.name}: complete=false (кэш ещё качается)")
            continue
        slug = payload.get("slug") or fp.stem[len("genius_"):]
        src = payload.get("source") or "genius"
        for i, song in enumerate(payload.get("songs") or []):
            raw = (song.get("text") or "").strip() if isinstance(song, dict) else ""
            if raw:
                yield {"source": src, "artist": slug, "title": song.get("title"),
                       "id": f"{src}:{slug}:{i}", "raw": raw, "genius": True}


def iter_huggingartists(corpus_dir: Path, stats: Counter | None = None):
    stats = stats if stats is not None else Counter()
    for fp in sorted(corpus_dir.glob("huggingartists_*.json")):
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            stats["skipped_broken"] += 1
            _warn(f"пропускаю битый {fp.name}: {exc}")
            continue
        if not isinstance(payload, dict) or not payload:
            stats["skipped_broken"] += 1
            _warn(f"пропускаю {fp.name}: пустой или не объект JSON")
            continue
        songs = payload.get("train") or payload.get("text") or []
        if isinstance(songs, dict):
            songs = list(songs.values())
        slug = fp.stem[len("huggingartists_"):].replace("_", "-")
        for i, item in enumerate(songs):
            raw = str(item).strip()
            if raw:
                yield {"source": "huggingartists", "artist": slug, "title": None,
                       "id": f"huggingartists:{slug}:{i}", "raw": raw, "genius": False}


def iter_ru_poetry(poetry_dir: Path, include_sample: bool = True):
    """classical_poetry.txt (+ stihi_ru_sample.txt, если полный stihi_ru не используется)."""
    files = [poetry_dir / "classical_poetry.txt"]
    if include_sample:
        files.append(poetry_dir / "stihi_ru_sample.txt")
    for fp in files:
        if not fp.is_file():
            continue
        buf: list[str] = []
        k = 0
        with fp.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.strip():
                    buf.append(line.rstrip("\n"))
                if len(buf) >= POEM_CHUNK_LINES:
                    yield {"source": "ru_poetry_txt", "artist": fp.stem, "title": None,
                           "id": f"{fp.stem}:{k}", "raw": "\n".join(buf), "genius": False}
                    buf, k = [], k + 1
        if buf:
            yield {"source": "ru_poetry_txt", "artist": fp.stem, "title": None,
                   "id": f"{fp.stem}:{k}", "raw": "\n".join(buf), "genius": False}


def iter_en_poetry(poetry_dir: Path):
    form_dir = poetry_dir / "schema" / "en" / "poetry_form" / "data"
    if form_dir.is_dir():
        try:
            import pyarrow.parquet as pq
        except ImportError:
            pq = None
        if pq is not None:
            for fp in sorted(form_dir.glob("*.parquet")):
                for row in pq.read_table(fp, columns=["id", "poem", "title", "author"]).to_pylist():
                    if (row.get("poem") or "").strip():
                        yield {"source": "en_poetry_form", "artist": row.get("author") or "unknown",
                               "title": row.get("title"), "id": f"poetry_form:{row['id']}",
                               "raw": row["poem"], "genius": False}
    raw_dir = poetry_dir / "schema" / "en" / "rhymedata" / "english_raw"
    for fp in sorted(raw_dir.glob("*.txt")) if raw_dir.is_dir() else []:
        author, title, buf, k = fp.stem, None, [], 0
        for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("AUTHOR "):
                author = line[7:].strip()
            elif line.startswith("TITLE "):
                if buf:
                    yield {"source": "en_rhymedata", "artist": author, "title": title,
                           "id": f"rhymedata:{fp.stem}:{k}", "raw": "\n".join(buf), "genius": False}
                    k += 1
                title, buf = line[6:].strip(), []
            elif line.startswith("RHYME "):
                continue
            elif line.strip():
                buf.append(line.strip())
        if buf:
            yield {"source": "en_rhymedata", "artist": author, "title": title,
                   "id": f"rhymedata:{fp.stem}:{k}", "raw": "\n".join(buf), "genius": False}


def iter_sources(corpus_dir: Path, poetry_dir: Path, include_stihi_sample: bool = True,
                 stats: Counter | None = None):
    # порядок задаёт, какая копия дубля остаётся: Genius (есть названия) раньше huggingartists
    yield from iter_genius(corpus_dir, stats)
    yield from iter_huggingartists(corpus_dir, stats)
    yield from iter_ru_poetry(poetry_dir, include_sample=include_stihi_sample)
    yield from iter_en_poetry(poetry_dir)


# ---------- полный stihi_ru: потоковый разбор и кеш ----------

def stihi_poem_chunks(poem: dict) -> tuple[list[dict], int]:
    """Стих -> куски по POEM_CHUNK_LINES строк после чистки. Возвращает (куски, число коротких)."""
    text = prepare_corpus_text(poem.get("text") or "", genius=False).strip()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    pid = str(poem.get("id") or "")
    out, short = [], 0
    for k in range(0, len(lines), POEM_CHUNK_LINES):
        body = "\n".join(lines[k:k + POEM_CHUNK_LINES])
        if len(body) < MIN_SONG_CHARS:
            short += 1
            continue
        cnt = script_counts(body)
        out.append({"id": f"stihi:{pid}:{k // POEM_CHUNK_LINES}", "poem": pid,
                    "artist": poem.get("author") or "unknown", "title": poem.get("title"),
                    "hash": text_hash(body), "tokens": count_tokens(body), "lines": body.count("\n") + 1,
                    "cyr": cnt["cyr"], "lat": cnt["lat"], "arab": cnt["arab"], "text": body})
    return out, short


def _stihi_batch(raw_lines: list[str]) -> tuple[list[dict], int, int]:
    chunks, short, poems = [], 0, 0
    for raw in raw_lines:
        try:
            poem = json.loads(raw)
        except json.JSONDecodeError:
            continue
        poems += 1
        c, sh = stihi_poem_chunks(poem)
        chunks.extend(c)
        short += sh
    return chunks, short, poems


def _iter_zst_lines(path: Path, batch: int = STIHI_BATCH):
    import zstandard

    with path.open("rb") as fh:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        buf = []
        for line in reader:
            if line.strip():
                buf.append(line)
            if len(buf) >= batch:
                yield buf
                buf = []
        if buf:
            yield buf


def stihi_cache_paths(src: Path = STIHI_ZST, cache_dir: Path = STIHI_CACHE_DIR) -> tuple[Path, Path]:
    stem = f"{src.name.split('.')[0]}_chunks{POEM_CHUNK_LINES}_v{STIHI_CACHE_VERSION}"
    return cache_dir / f"{stem}.jsonl.zst", cache_dir / f"{stem}.meta.json"


def _stihi_fingerprint(src: Path) -> dict:
    st = src.stat()
    return {"source": src.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "version": STIHI_CACHE_VERSION, "chunk_lines": POEM_CHUNK_LINES, "min_chars": MIN_SONG_CHARS}


def stihi_cache_valid(src: Path, data_path: Path, meta_path: Path) -> bool:
    if not (data_path.is_file() and meta_path.is_file()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("fingerprint") == _stihi_fingerprint(src)


def build_stihi_cache(src: Path, data_path: Path, meta_path: Path, workers: int, log=print) -> dict:
    """Потоково разбирает stihi_ru.jsonl.zst (без загрузки целиком) в кеш кусков. Порядок детерминирован."""
    import multiprocessing as mp

    import zstandard

    data_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = data_path.with_name(data_path.name + ".tmp")
    t0 = time.time()
    poems = chunks = short = 0
    with tmp.open("wb") as raw_out:
        writer = zstandard.ZstdCompressor(level=3).stream_writer(raw_out)
        out = io.TextIOWrapper(writer, encoding="utf-8")

        def consume(results):
            nonlocal poems, chunks, short
            for batch_chunks, sh, np_ in results:
                for c in batch_chunks:
                    out.write(json.dumps(c, ensure_ascii=False) + "\n")
                chunks += len(batch_chunks)
                short += sh
                poems += np_
                if poems and poems % 200000 < STIHI_BATCH:
                    log(f"  stihi: {poems} стихов, {chunks} кусков ({time.time() - t0:.0f}s)")

        if workers > 1:
            with mp.Pool(workers) as pool:
                consume(pool.imap(_stihi_batch, _iter_zst_lines(src), chunksize=1))
        else:
            consume(map(_stihi_batch, _iter_zst_lines(src)))
        out.flush()
        writer.flush(zstandard.FLUSH_FRAME)
        out.close()
    tmp.replace(data_path)
    meta = {"fingerprint": _stihi_fingerprint(src), "poems": poems, "chunks": chunks,
            "drop_short": short, "build_sec": round(time.time() - t0, 1)}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  stihi cache: {chunks} кусков из {poems} стихов за {meta['build_sec']}s -> {data_path.name}")
    return meta


def iter_stihi_cache(data_path: Path):
    import zstandard

    with data_path.open("rb") as fh:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for line in reader:
            if line.strip():
                yield json.loads(line)


class BigPool:
    """Компактное хранение миллионов кусков stihi_ru: только числа, тексты остаются в кеше."""

    def __init__(self, pool: str, data_path: Path):
        self.pool = pool
        self.data_path = data_path
        self.h64 = array("Q")
        self.tokens = array("I")
        self.lines = array("I")
        self.lat = array("I")
        self.script = array("I")
        self.split = array("b")
        self.idx = array("I")  # номер записи в кеше
        self.authors: set[str] = set()

    def __len__(self) -> int:
        return len(self.idx)

    def add(self, rec_idx: int, h64: int, rec: dict, split: str) -> None:
        self.h64.append(h64)
        self.tokens.append(rec["tokens"])
        self.lines.append(rec["lines"])
        self.lat.append(rec["lat"])
        self.script.append(rec["cyr"] + rec["lat"] + rec["arab"])
        self.split.append(SPLITS.index(split))
        self.idx.append(rec_idx)
        self.authors.add(rec["artist"])


# ---------- сборка пулов ----------

def h64_of(h: str) -> int:
    return int(h[:16], 16)


def collect_songs(corpus_dir: Path, poetry_dir: Path, artist_map: dict[str, str],
                  val: float, test: float, keep_text: bool = True, log=print,
                  rap_exclude: dict[str, str] | None = None, stihi_path: Path | None = None):
    """Возвращает (songs, stats, big). Песня: id, pool, split, hash, tokens, lines, source, artist, title[, text].
    stihi_path — кеш кусков полного stihi_ru (тогда stihi_ru_sample.txt не читается); big — BigPool или None."""
    rap_exclude = rap_exclude or {}
    stats = Counter()
    staged = []  # (row, text, counts) — пул артиста считается после прохода по всем его песням
    artist_scripts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    t0 = time.time()
    stats["skipped_incomplete"] = 0
    stats["skipped_broken"] = 0
    for row in iter_sources(corpus_dir, poetry_dir, include_stihi_sample=stihi_path is None, stats=stats):
        stats["raw_songs"] += 1
        text = prepare_corpus_text(row["raw"], genius=row["genius"]).strip()
        if len(text) < MIN_SONG_CHARS:
            stats["drop_short"] += 1
            continue
        cnt = script_counts(text)
        artist_scripts[(row["source"], row["artist"])].update(cnt)
        row.pop("raw")
        staged.append((row, text, cnt))
        if stats["raw_songs"] % 20000 == 0:
            log(f"  read {stats['raw_songs']} songs ({time.time() - t0:.0f}s)")

    seen: set[int] = set()
    songs = []
    for row, text, cnt in staged:
        a = artist_scripts[(row["source"], row["artist"])]
        n = sum(a.values()) or 1
        pool = assign_pool(row["source"], row["artist"], a["lat"] / n, artist_map, rap_exclude)
        if row["source"] != "ru_poetry_txt" and rap_exclude.get(artist_key(row["artist"])) == pool:
            stats[f"excluded_to_{pool}"] += 1
        why = song_filter(pool, cnt)
        if why:
            stats[f"drop_{why}"] += 1
            continue
        h = text_hash(text)
        if h64_of(h) in seen:
            stats["drop_duplicate"] += 1
            continue
        seen.add(h64_of(h))
        song = {
            "id": row["id"], "pool": pool, "split": split_of(split_hash(text), val, test), "hash": h,
            "source": row["source"], "artist": row["artist"], "title": row["title"],
            "tokens": count_tokens(text), "lines": text.count("\n") + 1,
            "lat_lines": cnt["lat"], "script_lines": sum(cnt.values()),
        }
        if keep_text:
            song["text"] = text
        songs.append(song)
    del staged

    big = None
    if stihi_path is not None:
        big = BigPool("ru_poetry", stihi_path)
        t1 = time.time()
        for i, rec in enumerate(iter_stihi_cache(stihi_path)):
            stats["stihi_chunks"] += 1
            cnt = Counter({"cyr": rec["cyr"], "lat": rec["lat"], "arab": rec["arab"]})
            why = song_filter(big.pool, cnt)
            if why:
                stats[f"drop_{why}"] += 1
                continue
            h64 = h64_of(rec["hash"])
            if h64 in seen:
                stats["drop_duplicate"] += 1
                continue
            seen.add(h64)
            # сплит по стиху: все куски одного стиха в одном сплите
            psplit = split_of(hashlib.sha1(f"stihi:{rec['poem']}".encode("utf-8")).hexdigest(), val, test)
            big.add(i, h64, rec, psplit)
            if stats["stihi_chunks"] % 1000000 == 0:
                log(f"  stihi: прочитано {stats['stihi_chunks']} кусков ({time.time() - t1:.0f}s)")
    stats["kept_songs"] = len(songs) + (len(big) if big else 0)
    return songs, stats, big


def pool_table(songs, big: BigPool | None = None) -> dict:
    table: dict[str, dict] = {}
    for s in songs:
        p = table.setdefault(s["pool"], {"songs": 0, "lines": 0, "tokens": 0, "artists": set(),
                                         "lat_lines": 0, "script_lines": 0,
                                         "by_split": {k: 0 for k in SPLITS}})
        p["songs"] += 1
        p["lines"] += s["lines"]
        p["tokens"] += s["tokens"]
        p["artists"].add(s["artist"])
        p["lat_lines"] += s["lat_lines"]
        p["script_lines"] += s["script_lines"]
        p["by_split"][s["split"]] += s["tokens"]
    if big is not None and len(big):
        import numpy as np

        p = table.setdefault(big.pool, {"songs": 0, "lines": 0, "tokens": 0, "artists": set(),
                                        "lat_lines": 0, "script_lines": 0,
                                        "by_split": {k: 0 for k in SPLITS}})
        tok = np.frombuffer(big.tokens, dtype=np.uint32).astype(np.int64)
        spl = np.frombuffer(big.split, dtype=np.int8)
        p["songs"] += len(big)
        p["lines"] += int(np.frombuffer(big.lines, dtype=np.uint32).astype(np.int64).sum())
        p["tokens"] += int(tok.sum())
        p["artists"] |= big.authors
        p["lat_lines"] += int(np.frombuffer(big.lat, dtype=np.uint32).astype(np.int64).sum())
        p["script_lines"] += int(np.frombuffer(big.script, dtype=np.uint32).astype(np.int64).sum())
        for j, name in enumerate(SPLITS):
            p["by_split"][name] += int(tok[spl == j].sum())
    for p in table.values():
        p["artists"] = len(p["artists"])
        p["latin_share"] = round(100.0 * p["lat_lines"] / max(1, p["script_lines"]), 2)
        del p["lat_lines"], p["script_lines"]
    return table


# ---------- рецепт и выборка ----------

def load_recipe(name_or_path: str) -> dict:
    import yaml

    path = Path(name_or_path)
    if not path.suffix:
        path = MIXES_DIR / f"{name_or_path}.yaml"
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    shares = {k: float(v) for k, v in (recipe.get("pools") or {}).items()}
    # пулы не из POOLS допустимы (их создаёт ru_rap_exclude.txt); пустой пул всплывёт как short
    total = sum(shares.values())
    if total <= 0:
        raise ValueError("доли пулов должны быть > 0")
    recipe["pools"] = {k: v / total for k, v in shares.items()}
    sp = recipe.get("splits") or {}
    recipe["splits"] = {"val": float(sp.get("val", 0.02)), "test": float(sp.get("test", 0.02))}
    recipe.setdefault("seed", 1)
    recipe.setdefault("total_tokens", "max")
    recipe["name"] = recipe.get("name") or path.stem
    fill_all = recipe.get("fill_all") or []
    recipe["fill_all"] = [str(p) for p in fill_all]
    caps = recipe.get("max_pool_share") or {}
    recipe["max_pool_share"] = {str(k): float(v) for k, v in caps.items()}
    return recipe


def max_feasible_total(available_train: dict[str, int], shares: dict[str, float]) -> tuple[int, str]:
    """Сколько train-токенов можно собрать в заданных долях без повторов и какой пул ограничивает."""
    best = None
    for pool, share in shares.items():
        if share <= 0:  # пул с долей 0 ничего не ограничивает
            continue
        cap = available_train.get(pool, 0) / share
        if best is None or cap < best[0]:
            best = (cap, pool)
    if best is None:
        return 0, None
    return int(best[0]), best[1]


def plan_targets(recipe: dict, table: dict) -> dict:
    shares = dict(recipe["pools"])
    fill_all = set(recipe.get("fill_all") or [])
    max_share = dict(recipe.get("max_pool_share") or {})
    all_pools = sorted(set(shares) | fill_all | set(max_share))
    for p in all_pools:
        shares.setdefault(p, 0.0)
    # clamp planned shares by max_pool_share, then renormalize
    for pool, cap in max_share.items():
        if pool in shares and shares[pool] > cap:
            shares[pool] = cap
    share_sum = sum(shares.values())
    if share_sum <= 0:
        raise ValueError("доли пулов после max_pool_share должны быть > 0")
    shares = {k: v / share_sum for k, v in shares.items()}

    avail = {s: {p: table.get(p, {}).get("by_split", {}).get(s, 0) for p in shares} for s in SPLITS}
    frac_train = 1.0 - recipe["splits"]["val"] - recipe["splits"]["test"]

    if fill_all:
        # Берём всё доступное по каждому пулу рецепта (fill_all и остальные),
        # затем усекаем пулы с max_pool_share. Мелкие пулы не лимитируют total.
        req = recipe["total_tokens"]
        train_take = {p: int(avail["train"].get(p, 0)) for p in shares}
        # итеративно усечь capped-пулы до max_pool_share от текущего total
        for _ in range(8):
            total = sum(train_take.values())
            changed = False
            for pool, cap in max_share.items():
                if pool not in train_take or cap <= 0:
                    continue
                limit = int(total * cap)
                if train_take[pool] > limit:
                    train_take[pool] = max(0, limit)
                    changed = True
            if not changed:
                break
        total = sum(train_take.values())
        if str(req).lower() != "max":
            want_total = int(req)
            if total > want_total:
                # пропорционально урезать non-fill пулы, fill_all оставить
                fixed = sum(train_take[p] for p in fill_all)
                flex_budget = max(0, want_total - fixed)
                flex_pools = [p for p in train_take if p not in fill_all]
                flex_have = sum(train_take[p] for p in flex_pools) or 1
                for p in flex_pools:
                    train_take[p] = int(train_take[p] * flex_budget / flex_have)
                total = sum(train_take.values())
        feasible = total
        limiting = ",".join(sorted(fill_all)) or None
        targets, short = {}, []
        for split in SPLITS:
            for pool in shares:
                have = avail[split][pool]
                if split == "train":
                    want = train_take.get(pool, 0)
                else:
                    # val/test: пропорционально train_take / train_avail, или всё если fill
                    train_have = avail["train"].get(pool, 0)
                    if train_have <= 0:
                        want = 0
                    elif pool in fill_all or train_take.get(pool, 0) >= train_have:
                        want = have
                    else:
                        want = int(have * train_take.get(pool, 0) / train_have)
                targets[(split, pool)] = min(want, have)
                if want > have:
                    short.append({"split": split, "pool": pool, "want": want, "have": have})
        caps = {p: int(avail["train"][p] / sh) for p, sh in shares.items() if sh > 0}
        return {"total_train_tokens": total, "max_feasible_train_tokens": feasible, "caps": caps,
                "limiting_pool": limiting, "targets": targets, "short": short,
                "fill_all": sorted(fill_all), "max_pool_share": max_share}

    feasible, limiting = max_feasible_total(avail["train"], shares)
    for split in ("val", "test"):  # val/test тоже не должны упираться в пул
        if recipe["splits"][split] <= 0:
            continue
        scale = recipe["splits"][split] / frac_train
        cap, pool = max_feasible_total({p: avail[split][p] / scale for p in shares}, shares)
        if cap < feasible:
            feasible, limiting = cap, f"{pool} ({split})"
    req = recipe["total_tokens"]
    total = feasible if str(req).lower() == "max" else int(req)
    for pool, cap in max_share.items():
        have = avail["train"].get(pool, 0)
        if shares.get(pool, 0) <= 0:
            continue
        # итоговая доля пула не больше cap
        max_total_by_cap = int(have / max(shares[pool], 1e-12))
        # also absolute: pool tokens <= cap * total → total >= pool/cap from have
        max_total_abs = int(have / cap) if cap > 0 else total
        limited = min(max_total_by_cap, max_total_abs)
        if limited < total:
            total, limiting = limited, f"{pool} (max_share={cap})"
    targets, short = {}, []
    for split in SPLITS:
        scale = 1.0 if split == "train" else recipe["splits"][split] / frac_train
        for pool, share in shares.items():
            want = int(total * share * scale)
            if pool in max_share and split == "train":
                want = min(want, int(total * max_share[pool]))
            have = avail[split][pool]
            targets[(split, pool)] = min(want, have)
            if want > have:
                short.append({"split": split, "pool": pool, "want": want, "have": have})
    caps = {p: int(avail["train"][p] / sh) for p, sh in shares.items() if sh > 0}
    return {"total_train_tokens": total, "max_feasible_train_tokens": feasible, "caps": caps,
            "limiting_pool": limiting, "targets": targets, "short": short,
            "fill_all": sorted(fill_all), "max_pool_share": max_share}


_M64 = (1 << 64) - 1


def order_key(h64: int, seed: int) -> int:
    """Детерминированный порядок выборки: splitmix64(h64 ^ seed)."""
    z = (h64 ^ (seed * 0x9E3779B97F4A7C15)) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def _order_keys_np(h64, seed: int):
    import numpy as np

    with np.errstate(over="ignore"):
        z = h64 ^ np.uint64((seed * 0x9E3779B97F4A7C15) & _M64)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def sample(songs, targets: dict, seed: int, big: BigPool | None = None) -> dict[str, list]:
    """Берёт песни в порядке order_key, пока не набран целевой объём токенов. Без повторов.
    Куски BigPool попадают в выборку как заглушки с полем _big (текст подтягивает hydrate_big)."""
    by_key = defaultdict(list)
    for s in songs:
        by_key[(s["split"], s["pool"])].append((order_key(h64_of(s["hash"]), seed), 0, s))
    big_keys = big_split = None
    if big is not None and len(big):
        import numpy as np

        big_keys = _order_keys_np(np.frombuffer(big.h64, dtype=np.uint64), seed)
        big_split = np.frombuffer(big.split, dtype=np.int8)
    out = {k: [] for k in SPLITS}
    for key, target in targets.items():
        split, pool = key
        streams = [iter(sorted(by_key.get(key, []), key=lambda t: t[0]))]
        if big_keys is not None and pool == big.pool:
            import numpy as np

            sel = np.nonzero(big_split == SPLITS.index(split))[0]
            order = sel[np.argsort(big_keys[sel], kind="stable")]
            streams.append(((int(big_keys[j]), 1, int(j)) for j in order))
        got = 0
        for _, kind, item in heapq.merge(*streams, key=lambda t: (t[0], t[1])):
            if got >= target:
                break
            if kind == 0:
                out[split].append(item)
                got += item["tokens"]
            else:
                out[split].append({"_big": int(big.idx[item]), "pool": big.pool, "split": split,
                                   "tokens": int(big.tokens[item])})
                got += int(big.tokens[item])
    for split in SPLITS:  # перемешать пулы внутри сплита, детерминированно
        random.Random(f"{seed}:{split}").shuffle(out[split])
    return out


def hydrate_big(chosen: dict[str, list], big: BigPool, log=print) -> None:
    """Подтягивает тексты выбранных кусков stihi_ru одним проходом по кешу."""
    need = {s["_big"]: s for rows in chosen.values() for s in rows if "_big" in s}
    if not need:
        return
    for i, rec in enumerate(iter_stihi_cache(big.data_path)):
        s = need.get(i)
        if s is not None:
            s.update({"id": rec["id"], "hash": rec["hash"], "source": "stihi_ru", "artist": rec["artist"],
                      "title": rec["title"], "text": rec["text"]})
            del s["_big"]
    log(f"  stihi: подтянуто {len(need)} кусков")


def real_shares(chosen: dict[str, list]) -> dict:
    res = {}
    for split, rows in chosen.items():
        tok = Counter()
        for s in rows:
            tok[s["pool"]] += s["tokens"]
        total = sum(tok.values())
        res[split] = {"songs": len(rows), "tokens": total,
                      "share": {p: round(100.0 * t / total, 2) for p, t in tok.items()} if total else {}}
    return res


def write_outputs(out_dir: Path, chosen: dict[str, list], report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in chosen.items():
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for s in rows:
                rec = {k: s[k] for k in ("id", "pool", "split", "hash", "source", "artist", "title", "tokens", "text")}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        (out_dir / f"{split}.txt").write_text(join_songs([s["text"] for s in rows]), encoding="utf-8")
        by_pool: dict[str, list[str]] = defaultdict(list)
        for s in rows:
            by_pool[s["pool"]].append(s["text"])
        for pool, texts in by_pool.items():
            (out_dir / f"{split}_{pool}.txt").write_text(join_songs(texts), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def render_table(table: dict) -> str:
    rows = [f"{'pool':<12} {'songs':>10} {'lines':>12} {'tokens':>14} {'train_tok':>14} {'artists':>8} {'latin%':>7}"]
    for pool in list(POOLS) + sorted(set(table) - set(POOLS)):
        p = table.get(pool)
        if not p:
            rows.append(f"{pool:<12} {'0':>10}")
            continue
        rows.append(f"{pool:<12} {_fmt(p['songs']):>10} {_fmt(p['lines']):>12} {_fmt(p['tokens']):>14} "
                    f"{_fmt(p['by_split']['train']):>14} {p['artists']:>8} {p['latin_share']:>7}")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Сборка микса из пулов по рецепту training/mixes/<name>.yaml")
    ap.add_argument("--mix", required=True, help="имя рецепта в training/mixes или путь к yaml")
    ap.add_argument("--seed", type=int, default=None, help="перекрывает seed из рецепта")
    ap.add_argument("--out", default=None, help="папка выхода (по умолчанию data/mixes/<name>)")
    ap.add_argument("--dry-run", action="store_true", help="только посчитать пулы и план, ничего не писать")
    ap.add_argument("--allow-short", action="store_true",
                    help="если пулу не хватает токенов — взять всё, что есть (без повторов), вместо ошибки")
    ap.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    ap.add_argument("--poetry-dir", default=str(POETRY_DIR))
    ap.add_argument("--stihi", choices=("full", "sample"), default="full",
                    help="ru_poetry из полного stihi_ru.jsonl.zst (full, по умолчанию) или из stihi_ru_sample.txt")
    ap.add_argument("--rebuild-cache", action="store_true", help="пересобрать кеш разобранного stihi_ru")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)),
                    help="процессов для разбора stihi_ru при сборке кеша")
    args = ap.parse_args(argv)

    recipe = load_recipe(args.mix)
    seed = args.seed if args.seed is not None else int(recipe["seed"])
    print(f"mix={recipe['name']} seed={seed} shares={recipe['pools']} total={recipe['total_tokens']}", flush=True)

    log = lambda m: print(m, flush=True)  # noqa: E731
    t_start = time.time()
    poetry_dir = Path(args.poetry_dir)
    stihi_path, stihi_meta = None, None
    if args.stihi == "full":
        src = poetry_dir / "raw" / STIHI_ZST.name
        try:
            import zstandard  # noqa: F401
        except ImportError:
            src = None
            log("WARN: нет пакета zstandard — ru_poetry из stihi_ru_sample.txt")
        if src is not None and not src.is_file():
            log(f"WARN: нет {src} — ru_poetry из stihi_ru_sample.txt")
            src = None
        if src is not None:
            data_path, meta_path = stihi_cache_paths(src, poetry_dir / "cache")
            if args.rebuild_cache or not stihi_cache_valid(src, data_path, meta_path):
                log(f"stihi: собираю кеш {data_path.name} (workers={args.workers})")
                stihi_meta = build_stihi_cache(src, data_path, meta_path, args.workers, log=log)
            else:
                stihi_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                log(f"stihi: кеш {data_path.name} актуален ({stihi_meta['chunks']} кусков)")
            stihi_path = data_path
    songs, stats, big = collect_songs(Path(args.corpus_dir), poetry_dir, load_artist_pools(),
                                      recipe["splits"]["val"], recipe["splits"]["test"],
                                      keep_text=not args.dry_run, log=log, rap_exclude=load_rap_exclude(log),
                                      stihi_path=stihi_path)
    if stihi_meta:
        stats["stihi_poems"] = stihi_meta["poems"]
        stats["stihi_drop_short"] = stihi_meta["drop_short"]
    table = pool_table(songs, big)
    plan = plan_targets(recipe, table)
    print("\n" + render_table(table))
    print(f"\nskipped files: incomplete={stats['skipped_incomplete']} (complete=false), "
          f"broken={stats['skipped_broken']} (битый/пустой JSON)")
    print("filters: " + ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(stats.items())))
    print(f"\nmax_feasible_train_tokens={_fmt(plan['max_feasible_train_tokens'])} "
          f"(limiting pool: {plan['limiting_pool']}); requested={recipe['total_tokens']}")
    print("train-токенов микса, если бы ограничивал только этот пул: "
          + ", ".join(f"{p}={_fmt(c)}" for p, c in plan["caps"].items()))
    for sh in plan["short"]:
        print(f"WARN short pool {sh['pool']}/{sh['split']}: want {_fmt(sh['want'])}, have {_fmt(sh['have'])}",
              file=sys.stderr)

    chosen = sample(songs, plan["targets"], seed, big)
    shares = real_shares(chosen)
    for split in SPLITS:
        print(f"{split}: songs={_fmt(shares[split]['songs'])} tokens={_fmt(shares[split]['tokens'])} "
              f"share={shares[split]['share']}")

    report = {
        "mix": recipe["name"], "seed": seed, "recipe_shares": recipe["pools"],
        "requested_total_tokens": recipe["total_tokens"], "splits": recipe["splits"],
        "pools": table, "filters": dict(stats),
        "skipped_incomplete": stats["skipped_incomplete"], "skipped_broken": stats["skipped_broken"],
        "max_feasible_train_tokens": plan["max_feasible_train_tokens"],
        "limiting_pool": plan["limiting_pool"], "caps": plan["caps"], "total_train_tokens": plan["total_train_tokens"],
        "short": plan["short"], "real": shares,
        "thresholds": {"artist_en_share": ARTIST_EN_SHARE, "song_lat_drop_ru": SONG_LAT_DROP_RU,
                       "song_cyr_drop_en": SONG_CYR_DROP_EN, "min_song_chars": MIN_SONG_CHARS},
    }
    report["elapsed_sec"] = round(time.time() - t_start, 1)
    report["stihi_cache"] = str(stihi_path.relative_to(ROOT)) if stihi_path and stihi_path.is_relative_to(ROOT) else (
        str(stihi_path) if stihi_path else None)
    print(f"elapsed={report['elapsed_sec']}s")
    if plan["short"] and not args.allow_short:  # и при --dry-run: тот же код, что у настоящей сборки
        print("ERROR: пулам не хватает токенов (см. WARN). Уменьшите total_tokens, поставьте max "
              "или запустите с --allow-short. Данные не повторяются.", file=sys.stderr)
        return 3
    if args.dry_run:
        print("\ndry-run: микс не записан (кеш stihi_ru, если собирался, лежит в data/corpus/poetry/cache)")
        return 0
    if big is not None:
        hydrate_big(chosen, big, log=log)
    out_dir = Path(args.out) if args.out else DEFAULT_OUT_ROOT / recipe["name"]
    write_outputs(out_dir, chosen, report)
    print(f"written {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
