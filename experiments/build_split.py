import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.corpus.lyrics import prepare_corpus_text
from core.corpus.genius import lyrics_looks_clean
CORPUS_DIR = ROOT / "data" / "corpus"
OUT_DIR = ROOT / "experiments" / "shared"
SEED = 20261005
TEST_RATIO = 0.05
HUGE_CHARS = 70_000
JUNK_MARKERS = ('"children"', "window.__", "<script")
SEEDS = [
    "Я",
    "я ебал игру будто это шлюха",
    "дай дай дай бабки дай дай шум",
    "Зиртек",
    "Братва",
    "На районе",
]


def _has_page_junk(text):
    if not text:
        return True
    if len(text) >= HUGE_CHARS:
        return True
    if not lyrics_looks_clean(text):
        return True
    low = text.lower()
    return any(marker.lower() in low for marker in JUNK_MARKERS)


def _song_id(slug, idx, song):
    url = (song.get("url") or "").strip()
    if url:
        return url
    title = (song.get("title") or "").strip() or "untitled"
    return f"genius:{slug}:{idx}:{title}"


def load_genius_songs():
    songs = []
    dropped = []
    seen = set()
    for path in sorted(CORPUS_DIR.glob("genius_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        slug = payload.get("slug") or path.stem.replace("genius_", "", 1)
        artist = payload.get("name") or slug
        for idx, song in enumerate(payload.get("songs") or []):
            raw = (song.get("text") or "").strip()
            sid = _song_id(slug, idx, song)
            if not raw:
                dropped.append({"id": sid, "reason": "empty"})
                continue
            if _has_page_junk(raw):
                dropped.append({"id": sid, "reason": "page_junk"})
                continue
            cleaned = prepare_corpus_text(raw, genius=True).strip()
            if not cleaned:
                dropped.append({"id": sid, "reason": "empty_clean"})
                continue
            if _has_page_junk(cleaned):
                dropped.append({"id": sid, "reason": "page_junk_clean"})
                continue
            if sid in seen:
                dropped.append({"id": sid, "reason": "dup"})
                continue
            seen.add(sid)
            songs.append(
                {
                    "id": sid,
                    "artist": artist,
                    "slug": slug,
                    "title": (song.get("title") or "").strip() or None,
                    "text": cleaned,
                    "chars": len(cleaned),
                }
            )
    return songs, dropped


def stratified_split(songs, seed=SEED, test_ratio=TEST_RATIO):
    by_artist = defaultdict(list)
    for song in songs:
        by_artist[song["artist"]].append(song)
    rng = random.Random(seed)
    train, test = [], []
    for artist in sorted(by_artist):
        items = list(by_artist[artist])
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_ratio))
        if n_test >= n:
            n_test = n - 1 if n > 1 else 0
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    if not test and train:
        rng.shuffle(train)
        n_test = max(1, int(round(len(train) * test_ratio)))
        test = train[:n_test]
        train = train[n_test:]
    return train, test


def _join(songs):
    body = "\n\n".join(s["text"] for s in songs)
    if body and not body.endswith("\n"):
        body += "\n"
    return body


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    songs, dropped = load_genius_songs()
    if not songs:
        raise SystemExit("no clean Genius songs")
    train, test = stratified_split(songs)
    train_ids = {s["id"] for s in train}
    test_ids = {s["id"] for s in test}
    overlap = train_ids & test_ids
    if overlap:
        raise SystemExit(f"id overlap: {len(overlap)}")
    if not test:
        raise SystemExit("test split is empty")

    train_text = _join(train)
    test_text = _join(test)
    (OUT_DIR / "train.txt").write_text(train_text, encoding="utf-8")
    (OUT_DIR / "test.txt").write_text(test_text, encoding="utf-8")
    (OUT_DIR / "seeds.json").write_text(
        json.dumps(SEEDS, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    train_chars = [s["chars"] for s in train]
    test_chars = [s["chars"] for s in test]
    typical = sorted(s["chars"] for s in songs)
    median = typical[len(typical) // 2]
    split = {
        "seed": SEED,
        "train_ratio": 1.0 - TEST_RATIO,
        "test_ratio": TEST_RATIO,
        "source": "genius_*.json",
        "train_ids": [s["id"] for s in train],
        "test_ids": [s["id"] for s in test],
        "dropped": dropped,
        "counts": {
            "songs_kept": len(songs),
            "songs_dropped": len(dropped),
            "train_songs": len(train),
            "test_songs": len(test),
            "train_chars": len(train_text),
            "test_chars": len(test_text),
            "artists": len({s["artist"] for s in songs}),
            "median_song_chars": median,
            "max_song_chars": max(typical),
            "max_test_song_chars": max(test_chars) if test_chars else 0,
        },
    }
    (OUT_DIR / "split_ids.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("songs_kept", len(songs))
    print("songs_dropped", len(dropped))
    print("train_songs", len(train), "train_chars", len(train_text))
    print("test_songs", len(test), "test_chars", len(test_text))
    print("artists", split["counts"]["artists"])
    print("median_song_chars", median, "max_song_chars", max(typical))
    print("max_test_song_chars", split["counts"]["max_test_song_chars"])
    print("id_overlap", len(overlap))
    print("test_empty", False)
    junk70 = sum(1 for s in songs if s["chars"] >= HUGE_CHARS)
    print("songs_ge_70k", junk70)
    if median >= HUGE_CHARS:
        raise SystemExit("median song looks like JSON junk")
    if max(test_chars) >= HUGE_CHARS:
        raise SystemExit("test song looks like JSON junk")


if __name__ == "__main__":
    main()
