import argparse
import sys
from pathlib import Path

from core.corpus.genius import salvage_caches
from core.corpus.genius_pool import MAX_WORKERS, adopt_legacy_caches, pending_slugs, run_genius_pool
from training.prepare_data import DEFAULT_OUT, GENIUS_ARTISTS, write_corpus_from_genius_cache

DEFAULT_CACHE = Path("data/corpus")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign Genius artists to at most 5 workers, then merge caches"
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--partial",
        nargs="*",
        default=(),
        help="Slugs whose json was cut off; those files are deleted and fetched again",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache)
    out_path = Path(args.out)
    healed = salvage_caches(cache_dir)
    if healed:
        print(f"salvaged dirty caches={len(healed)}", flush=True)
    stamped, dropped = adopt_legacy_caches(cache_dir, args.partial)
    if stamped or dropped:
        print(
            f"legacy cache stamped={len(stamped)} dropped={len(dropped)}",
            flush=True,
        )
    _done, failures = run_genius_pool(GENIUS_ARTISTS, cache_dir, workers=args.workers)
    missing = pending_slugs(GENIUS_ARTISTS, cache_dir)
    if failures or missing:
        print(
            f"corpus not written incomplete={len(missing)} failures={len(failures)}",
            flush=True,
        )
        return 1
    n_docs, before, after, stats = write_corpus_from_genius_cache(cache_dir, out_path)
    ok = ", ".join(name for name, n, _chars in stats if n > 0)
    print(
        f"wrote {out_path} source=genius-pool artists=[{ok}] docs={n_docs} "
        f"chars_raw={before} chars_clean={after}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
