import argparse
import sys
from pathlib import Path

from core.corpus.pack import pack_corpus


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pack local lyrics caches into labeled JSONL (no upload)"
    )
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--out-dir", default="data/dataset")
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        main_path, manifest = pack_corpus(
            corpus_dir=args.corpus_dir,
            out_dir=args.out_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    print(
        f"wrote {main_path} rows={manifest['rows_total']} "
        f"train={manifest['train']} valid={manifest['valid']} "
        f"sources={manifest['by_source']} genius={manifest['genius_download_note']}"
    )


if __name__ == "__main__":
    main()
