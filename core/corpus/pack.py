import json
import random
import re
from pathlib import Path

from core.corpus.lyrics import _BRACKET_ONLY, _is_genius_section_label, prepare_corpus_text

HF_PREFIX = "huggingartists_"
GENIUS_PREFIX = "genius_"
DEFAULT_CORPUS_DIR = Path("data/corpus")
DEFAULT_OUT_DIR = Path("data/dataset")


def slug_from_hf_path(path: Path) -> str:
    stem = path.stem
    if not stem.startswith(HF_PREFIX):
        raise ValueError(f"not a huggingartists cache: {path}")
    return stem[len(HF_PREFIX) :].replace("_", "-")


def row_has_section_tag_line(text: str) -> bool:
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        bracket = _BRACKET_ONLY.match(line)
        if bracket and _is_genius_section_label(bracket.group(1)):
            return True
        low = line.lower()
        if low in ("embed", "read more") or "you might also like" in low:
            return True
    return False


def _clean_text(text: str, source: str) -> str:
    genius = source in ("genius", "lyricsgenius")
    return prepare_corpus_text(text, genius=genius).strip()


def _row_id(source: str, artist: str, index: int) -> str:
    safe_artist = re.sub(r"[^a-zA-Z0-9._-]+", "_", artist)
    return f"{source}:{safe_artist}:{index}"


def iter_huggingartists_rows(corpus_dir: Path):
    for path in sorted(corpus_dir.glob(f"{HF_PREFIX}*.json")):
        slug = slug_from_hf_path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        songs = payload.get("train") or payload.get("text") or []
        if isinstance(songs, dict):
            songs = list(songs.values())
        for idx, item in enumerate(songs):
            raw = str(item).strip()
            if not raw:
                continue
            text = _clean_text(raw, "huggingartists")
            if not text:
                continue
            yield {
                "id": _row_id("huggingartists", slug, idx),
                "artist": slug,
                "title": None,
                "text": text,
                "source": "huggingartists",
            }


def iter_genius_rows(corpus_dir: Path):
    for path in sorted(corpus_dir.glob(f"{GENIUS_PREFIX}*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        slug = payload.get("slug") or path.stem.replace(GENIUS_PREFIX, "", 1)
        artist = payload.get("name") or slug
        src = payload.get("source") or "genius"
        for idx, song in enumerate(payload.get("songs") or []):
            raw = (song.get("text") or "").strip()
            if not raw:
                continue
            text = _clean_text(raw, src)
            if not text:
                continue
            title = (song.get("title") or "").strip() or None
            yield {
                "id": _row_id(src, artist, idx),
                "artist": artist,
                "title": title,
                "text": text,
                "source": src,
            }


def collect_rows(corpus_dir: Path):
    rows = []
    rows.extend(iter_huggingartists_rows(corpus_dir))
    rows.extend(iter_genius_rows(corpus_dir))
    return rows


def assign_splits(rows, train_ratio=0.98, seed=1):
    if not rows:
        return rows
    order = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(order)
    n_train = max(1, int(len(rows) * train_ratio))
    if len(rows) > 1 and n_train >= len(rows):
        n_train = len(rows) - 1
    train_set = set(order[:n_train])
    out = []
    for i, row in enumerate(rows):
        split = "train" if i in train_set else "valid"
        item = dict(row)
        item["split"] = split
        out.append(item)
    return out


def write_dataset(rows, out_dir: Path, corpus_dir: Path, train_ratio=0.98, seed=1):
    out_dir.mkdir(parents=True, exist_ok=True)
    main_path = out_dir / "lyrics.jsonl"
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_ids = splits_dir / "train_ids.txt"
    valid_ids = splits_dir / "valid_ids.txt"

    with main_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_rows = [r for r in rows if r["split"] == "train"]
    valid_rows = [r for r in rows if r["split"] == "valid"]
    for path, subset in (
        (train_path, train_rows),
        (valid_path, valid_rows),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_ids.write_text("\n".join(r["id"] for r in train_rows) + ("\n" if train_rows else ""), encoding="utf-8")
    valid_ids.write_text("\n".join(r["id"] for r in valid_rows) + ("\n" if valid_rows else ""), encoding="utf-8")

    by_source = {}
    by_artist = {}
    for row in rows:
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1
        key = f"{row['source']}:{row['artist']}"
        by_artist[key] = by_artist.get(key, 0) + 1

    genius_files = list(corpus_dir.glob(f"{GENIUS_PREFIX}*.json"))
    ratio_pct = int(round(train_ratio * 100))
    manifest = {
        "rows_total": len(rows),
        "train": len(train_rows),
        "valid": len(valid_rows),
        "split_ratio": f"{ratio_pct}/{100 - ratio_pct}",
        "split_seed": seed,
        "by_source": by_source,
        "by_artist": by_artist,
        "genius_cache_files": len(genius_files),
        "genius_download_note": (
            "partial_or_absent"
            if len(genius_files) == 0
            else f"{len(genius_files)} genius_*.json in corpus/"
        ),
        "merged_corpus_txt": str(DEFAULT_CORPUS_DIR / "kizaru_lyrics.txt"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return main_path, manifest


def pack_corpus(corpus_dir=None, out_dir=None, train_ratio=0.98, seed=1):
    corpus_dir = Path(corpus_dir or DEFAULT_CORPUS_DIR)
    out_dir = Path(out_dir or DEFAULT_OUT_DIR)
    rows = collect_rows(corpus_dir)
    if not rows:
        raise ValueError(f"No lyric rows found under {corpus_dir}")
    rows = assign_splits(rows, train_ratio=train_ratio, seed=seed)
    main_path, manifest = write_dataset(
        rows, out_dir, corpus_dir, train_ratio=train_ratio, seed=seed
    )
    return main_path, manifest
