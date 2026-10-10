"""Download classical RU poetry (PoetryCorpus all.xml) + optional stihi_ru sample."""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OUT_DIR = Path("data/corpus/poetry")
OUT_TXT = OUT_DIR / "classical_poetry.txt"
RAW_XML = OUT_DIR / "raw" / "poetry_corpus.xml"

COMMIT = "7a5f70e6a46b4717f7c903671f9a6a917aee6162"
DOWNLOAD_URL = (
    f"https://github.com/IlyaGusev/PoetryCorpus/raw/{COMMIT}/datasets/corpus/all.xml"
)
FILE_SHA256 = "3f6e43fb066916f3ee32dd7b63488c4831b62a29f72caa41e806a8ef59749c89"


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"GET {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "KizaruAi/1.0"})
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as out:
        total = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            total += len(chunk)
            if total % (1024 * 1024 * 2) < 1024 * 256:
                print(f"  bytes={total}", flush=True)
    digest = h.hexdigest()
    print(f"saved {dest} bytes={dest.stat().st_size} sha256={digest}", flush=True)
    if digest != FILE_SHA256:
        print(f"WARN sha mismatch expected={FILE_SHA256}", flush=True)
    return dest


def poems_from_xml(path: Path):
    # Stream-parse large XML
    for _event, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "item":
            continue
        text_el = elem.find("text")
        text = (text_el.text or "").strip() if text_el is not None else ""
        if text:
            yield text
        elem.clear()


def write_corpus(poems, out_path: Path) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    chars = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for poem in poems:
            body = "\n".join(line.strip() for line in poem.splitlines() if line.strip())
            if not body:
                continue
            handle.write(body)
            handle.write("\n")
            n += 1
            chars += len(body)
            if n % 2000 == 0:
                print(f"  poems={n} chars={chars}", flush=True)
    return n, chars


def try_hf_stihi_sample(limit: int = 50000) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed; skip HF stihi_ru", flush=True)
        return
    print(f"streaming IlyaGusev/stihi_ru limit={limit}", flush=True)
    ds = load_dataset("IlyaGusev/stihi_ru", split="train", streaming=True)
    out = OUT_DIR / "stihi_ru_sample.txt"
    n = 0
    chars = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in ds:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            body = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            if not body:
                continue
            handle.write(body)
            handle.write("\n")
            n += 1
            chars += len(body)
            if n % 5000 == 0:
                print(f"  stihi poems={n} chars={chars}", flush=True)
            if n >= limit:
                break
    print(f"wrote {out} poems={n} chars={chars}", flush=True)


def main():
    if not RAW_XML.exists() or RAW_XML.stat().st_size < 1_000_000:
        download(DOWNLOAD_URL, RAW_XML)
    else:
        print(f"reuse {RAW_XML} bytes={RAW_XML.stat().st_size}", flush=True)
    n, chars = write_corpus(poems_from_xml(RAW_XML), OUT_TXT)
    print(f"wrote {OUT_TXT} poems={n} chars={chars}", flush=True)
    try_hf_stihi_sample(limit=50000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
