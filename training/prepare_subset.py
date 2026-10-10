import argparse
import json
import re
import sys
from pathlib import Path

from core.corpus.clean import join_songs
from core.corpus.lyrics import clean_lyrics, prepare_corpus_text

CACHE_DIR = Path("data/corpus")
POOLS_DIR = Path("training/artist_pools_resume")
DEFAULT_POETRY = (
    Path("data/corpus/poetry/classical_poetry.txt"),
    Path("data/corpus/poetry/stihi_ru_sample.txt"),
)


def _norm_slug(slug: str) -> str:
    s = slug.lower().rstrip("-")
    return re.sub(r"[^a-z0-9]+", "", s)


def _load_pool(name: str) -> list[str]:
    path = POOLS_DIR / name
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _index_caches(cache_dir: Path) -> dict[str, Path]:
    return {p.name[len("genius_") : -len(".json")]: p for p in cache_dir.glob("genius_*.json")}


def _payload_weight(path: Path) -> tuple[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    songs = payload.get("songs") or []
    texts = [(s.get("text") or "").strip() for s in songs]
    texts = [t for t in texts if t]
    return len(texts), sum(len(t) for t in texts)


def _by_norm(caches: dict[str, Path]) -> dict[str, list[str]]:
    by_norm: dict[str, list[str]] = {}
    for slug in caches:
        by_norm.setdefault(_norm_slug(slug), []).append(slug)
    return by_norm


def resolve_all_complete(caches: dict[str, Path]) -> tuple[list[str], int]:
    by_norm = _by_norm(caches)
    chosen: dict[str, str] = {}
    skipped = 0
    for key, family in by_norm.items():
        complete_family = []
        for slug in family:
            try:
                payload = json.loads(caches[slug].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                skipped += 1
                continue
            if payload.get("complete") is not True:
                skipped += 1
                continue
            songs = payload.get("songs") or []
            if not any((s.get("text") or "").strip() for s in songs):
                skipped += 1
                continue
            complete_family.append(slug)
        if not complete_family:
            continue
        chosen[key] = max(complete_family, key=lambda s: _payload_weight(caches[s]))
    return sorted(chosen.values()), skipped


def resolve_slugs(wanted: list[str], caches: dict[str, Path]) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    by_norm = _by_norm(caches)

    chosen: dict[str, str] = {}
    aliases: list[tuple[str, str]] = []
    missing: list[str] = []

    for slug in wanted:
        if slug in caches:
            key = _norm_slug(slug)
            prev = chosen.get(key)
            if prev is None:
                chosen[key] = slug
            else:
                w_new = _payload_weight(caches[slug])
                w_old = _payload_weight(caches[prev])
                if w_new > w_old:
                    chosen[key] = slug
            continue
        alts = by_norm.get(_norm_slug(slug), [])
        if not alts:
            missing.append(slug)
            continue
        best = max(alts, key=lambda s: _payload_weight(caches[s]))
        key = _norm_slug(best)
        prev = chosen.get(key)
        if prev is None or _payload_weight(caches[best]) > _payload_weight(caches[prev]):
            chosen[key] = best
        aliases.append((slug, best))

    # drop trailing-hyphen / compact dupes: keep heaviest per norm among all caches that match chosen norms
    for key, slug in list(chosen.items()):
        family = by_norm.get(key, [slug])
        best = max(family, key=lambda s: _payload_weight(caches[s]))
        chosen[key] = best

    return sorted(chosen.values()), missing, aliases


def load_texts(slugs: list[str], caches: dict[str, Path]):
    texts = []
    stats = []
    for slug in slugs:
        payload = json.loads(caches[slug].read_text(encoding="utf-8"))
        songs = payload.get("songs") or []
        parts = [(s.get("text") or "").strip() for s in songs]
        parts = [t for t in parts if t]
        texts.extend(parts)
        stats.append((slug, len(parts), sum(len(t) for t in parts)))
    return texts, stats


def load_poetry_docs(paths: list[Path]) -> list[str]:
    # download_poetry writes poems with a single trailing \\n (no blank separator),
    # so poem boundaries are not recoverable — clean as LM_STREAM bars.
    docs = []
    for path in paths:
        if not path.is_file():
            print(f"skip missing poetry file: {path}", file=sys.stderr)
            continue
        raw = path.read_text(encoding="utf-8")
        cleaned = clean_lyrics(raw).strip()
        if cleaned:
            docs.append(cleaned)
        print(
            f"poetry {path.name}: chars_raw={len(raw)} chars_clean={len(cleaned)}",
            flush=True,
        )
    return docs


def parse_args():
    p = argparse.ArgumentParser(
        description="Clean genius pool subset (+ optional local poetry txt) into a lyrics corpus"
    )
    p.add_argument(
        "--pools",
        nargs="*",
        default=["05_existing.txt", "04_en_rap.txt"],
        help="Pool files under training/artist_pools_resume (no Genius poets)",
    )
    p.add_argument(
        "--all-caches",
        action="store_true",
        help="Use every complete genius_*.json (dedupe by normalized slug); ignore --pools",
    )
    p.add_argument(
        "--poetry",
        nargs="*",
        default=[str(p) for p in DEFAULT_POETRY],
        help="Local poetry txt files (PoetryCorpus / stihi). Empty list to skip.",
    )
    p.add_argument("--no-poetry", action="store_true", help="Do not append local poetry")
    p.add_argument("--out", default="data/corpus/kizaru_lyrics_ready.txt")
    p.add_argument("--cache", default=str(CACHE_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache)
    caches = _index_caches(cache_dir)
    wanted: list[str] = []
    missing: list[str] = []
    aliases: list[tuple[str, str]] = []
    skipped_incomplete = 0

    if args.all_caches:
        slugs, skipped_incomplete = resolve_all_complete(caches)
        wanted = list(slugs)
    else:
        seen = set()
        for pool in args.pools or []:
            for slug in _load_pool(pool):
                if slug not in seen:
                    seen.add(slug)
                    wanted.append(slug)
        slugs, missing, aliases = resolve_slugs(wanted, caches) if wanted else ([], [], [])

    texts, stats = load_texts(slugs, caches) if slugs else ([], [])

    poetry_paths = [] if args.no_poetry else [Path(p) for p in (args.poetry or [])]
    poetry_docs = load_poetry_docs(poetry_paths) if poetry_paths else []

    if not texts and not poetry_docs:
        raise SystemExit("no songs/poetry resolved")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Clean genius songs with genius pipeline; poetry already cleaned via clean_lyrics.
    genius_parts = [prepare_corpus_text(t, genius=True).strip() for t in texts]
    genius_parts = [t for t in genius_parts if t]
    before = sum(len(t) for t in texts) + sum(len(t) for t in poetry_docs)
    merged = join_songs(genius_parts + poetry_docs, separator="\n")
    after = len(merged)
    out.write_text(merged, encoding="utf-8")

    ok = sum(1 for _, n, _ in stats if n > 0)
    source_note = "all_caches" if args.all_caches else f"pools={args.pools}"
    denom = len(caches) if args.all_caches else len(wanted)
    print(
        f"wrote {out} {source_note} genius_artists={ok}/{denom} "
        f"genius_docs={len(genius_parts)} poetry_docs={len(poetry_docs)} "
        f"chars_raw={before} chars_clean={after} "
        f"missing={len(missing)} aliases={len(aliases)} "
        f"skipped_incomplete={skipped_incomplete}",
        flush=True,
    )
    print(f"keep_ratio={after / max(before, 1):.3f}", flush=True)
    if missing[:20]:
        print("missing sample: " + ", ".join(missing[:20]), file=sys.stderr)
    lines = merged.split("\n")
    empty = sum(1 for line in lines if not line.strip())
    bracket = sum(1 for line in lines if line.strip().startswith("[") and "]" in line[:40])
    print(
        f"audit lines={len(lines)} empty={empty} bracketish_lines={bracket} "
        f"size_mb={out.stat().st_size / 1e6:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
