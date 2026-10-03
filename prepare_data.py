import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

DEFAULT_OUT = Path("data/kizaru_lyrics.txt")
HF_JSON = "https://huggingface.co/datasets/huggingartists/kizaru/resolve/main/datasets.json"
LEGACY_CSV = Path("legacy/messages.csv")


def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "KizaruAi/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, dest.open("wb") as handle:
        handle.write(response.read())
    return dest


def from_huggingartists(out_path):
    raw_path = Path("data/huggingartists_kizaru.json")
    download(HF_JSON, raw_path)
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    songs = payload.get("train") or payload.get("text") or []
    if isinstance(songs, dict):
        songs = list(songs.values())
    texts = [str(item).strip() for item in songs if str(item).strip()]
    if not texts:
        raise ValueError("huggingartists/kizaru has no texts")
    out_path.write_text("\n\n".join(texts), encoding="utf-8")
    return len(texts), sum(len(text) for text in texts)


def from_legacy_csv(out_path):
    if not LEGACY_CSV.exists():
        raise FileNotFoundError(LEGACY_CSV)
    parts = []
    with LEGACY_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = (row.get("text") or "").strip()
            if text:
                parts.append(text)
    if not parts:
        raise ValueError("legacy messages.csv is empty")
    out_path.write_text("\n\n".join(parts), encoding="utf-8")
    return len(parts), sum(len(text) for text in parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a local lyrics corpus")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--source", choices=("kizaru", "legacy", "auto"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    sources = ["kizaru", "legacy"] if args.source == "auto" else [args.source]
    for source in sources:
        try:
            if source == "kizaru":
                n_docs, n_chars = from_huggingartists(out_path)
            else:
                n_docs, n_chars = from_legacy_csv(out_path)
            print(f"wrote {out_path} source={source} docs={n_docs} chars={n_chars}")
            return
        except Exception as exc:
            errors.append(f"{source}: {exc}")
            print(f"{source} failed: {exc}", file=sys.stderr)
    raise SystemExit("Could not prepare corpus:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
