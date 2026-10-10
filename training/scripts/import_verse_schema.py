"""Import RU/EN verse corpora into song-schema JSONL + LM stream txt."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCHEMA_DIR = Path("data/corpus/poetry/schema")
RIFMA = SCHEMA_DIR / "rifma_dataset.json"
ARS = SCHEMA_DIR / "arspoetica.json"
EN_RAW = SCHEMA_DIR / "en" / "rhymedata" / "english_raw"
EN_FORM = SCHEMA_DIR / "en" / "poetry_form" / "data"
OUT_JSONL = SCHEMA_DIR / "verse_schema.jsonl"
OUT_TXT = SCHEMA_DIR / "verse_schema_lm.txt"


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _guess_form(n: int, rhyme: str | None) -> str:
    if n <= 2:
        return "couplet"
    if n <= 4:
        return "quatrain"
    if n <= 8:
        return "couplet"
    return "freestyle"


def _normalize_rhyme(scheme) -> str | None:
    if scheme is None:
        return None
    s = str(scheme).strip()
    if not s or set(s) <= {"-", "*"}:
        return "free"
    # chicago: "a b a b" / "a a *" -> ABAB / AA*
    parts = s.replace("*", "*").split()
    if len(parts) >= 2 and all(len(p) == 1 for p in parts):
        out = []
        for p in parts:
            if p == "*":
                out.append("*")
            else:
                out.append(p.upper())
        return "".join(out)
    return s.upper().replace(" ", "") if " " in s else s


def iter_rifma(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    for i, row in enumerate(rows):
        text = row.get("poem_text") or ""
        body = _lines(text)
        if len(body) < 2:
            continue
        rhyme = _normalize_rhyme(row.get("rhyme_scheme"))
        yield {
            "id": f"rifma:{i}",
            "source": "rifma",
            "license": "MIT",
            "lang": "ru",
            "form": _guess_form(len(body), rhyme),
            "lines": len(body),
            "rhyme": rhyme or "free",
            "meter": None,
            "mood": None,
            "seed": body[0],
            "body": "\n".join(body),
            "accentuation": row.get("accentuation_markup"),
        }


def iter_ars(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    for i, row in enumerate(rows):
        text = row.get("poem_text") or ""
        stanzas = re.split(r"\n\s*\n", text.strip())
        for j, stanza in enumerate(stanzas):
            body = _lines(stanza)
            if len(body) < 2:
                continue
            if len(body) > 12:
                for k in range(0, len(body), 8):
                    chunk = body[k : k + 8]
                    if len(chunk) < 2:
                        continue
                    yield _ars_row(i, j, k, row, chunk)
            else:
                yield _ars_row(i, j, 0, row, body)


def _ars_row(i, j, k, row, body):
    return {
        "id": f"ars:{i}:{j}:{k}",
        "source": "arspoetica",
        "license": "CC-BY-NC-4.0",
        "lang": "ru",
        "form": _guess_form(len(body), None),
        "lines": len(body),
        "rhyme": "free",
        "meter": None,
        "mood": None,
        "seed": body[0],
        "body": "\n".join(body),
        "accentuation": row.get("accentuation"),
        "author": row.get("author"),
        "title": row.get("title"),
    }


def _scheme_to_letters(tok: str) -> str:
    return _normalize_rhyme(tok) or "free"


def iter_chicago_en(raw_dir: Path):
    # Chicago Rhyming Poetry Corpus (public domain). Stanza after RHYME line.
    if not raw_dir.is_dir():
        return
    for path in sorted(raw_dir.glob("*.txt")):
        author = path.stem
        title = None
        rhyme = None
        buf: list[str] = []
        stanza_i = 0

        def flush():
            nonlocal stanza_i, buf, rhyme
            body = [ln for ln in buf if ln.strip()]
            buf = []
            if len(body) < 2:
                rhyme = None
                return
            scheme = _scheme_to_letters(rhyme) if rhyme else "free"
            yield_rec = {
                "id": f"chicago:{path.stem}:{stanza_i}",
                "source": "chicago_rhyme",
                "license": "public-domain",
                "lang": "en",
                "form": _guess_form(len(body), scheme),
                "lines": len(body),
                "rhyme": scheme,
                "meter": None,
                "mood": None,
                "seed": body[0],
                "body": "\n".join(body),
                "author": author,
                "title": title,
            }
            stanza_i += 1
            rhyme = None
            return yield_rec

        text = path.read_text(encoding="utf-8", errors="replace")
        for raw in text.splitlines():
            line = raw.rstrip()
            if line.startswith("AUTHOR "):
                rec = flush()
                if rec:
                    yield rec
                author = line[len("AUTHOR ") :].strip() or author
                continue
            if line.startswith("TITLE "):
                rec = flush()
                if rec:
                    yield rec
                title = line[len("TITLE ") :].strip()
                continue
            if line.startswith("RHYME-POEM"):
                continue
            if line.startswith("RHYME"):
                rec = flush()
                if rec:
                    yield rec
                rhyme = line[len("RHYME") :].strip()
                continue
            if not line.strip():
                continue
            buf.append(line.strip())
        rec = flush()
        if rec:
            yield rec


def iter_poetry_form(data_dir: Path):
    # PoetryMTEB EnglishPoetryFormClassification parquet (form labels, not rhyme letters).
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("skip poetry_form: pyarrow missing", flush=True)
        return
    if not data_dir.is_dir():
        return
    for path in sorted(data_dir.glob("*.parquet")):
        table = pq.read_table(path)
        cols = {name: table.column(name) for name in table.column_names}
        n = table.num_rows
        for i in range(n):
            poem = str(cols["poem"][i].as_py() or "").strip()
            body = _lines(poem)
            if len(body) < 2:
                continue
            label = None
            if "label_name" in cols:
                label = str(cols["label_name"][i].as_py() or "")
            form = label or _guess_form(len(body), None)
            yield {
                "id": f"poetry_form:{path.stem}:{i}",
                "source": "poetry_form_en",
                "license": "CC-BY-SA-4.0",
                "lang": "en",
                "form": form,
                "lines": len(body),
                "rhyme": "free",
                "meter": None,
                "mood": None,
                "seed": body[0],
                "body": "\n".join(body[:16]),
                "author": str(cols["author"][i].as_py()) if "author" in cols else None,
                "title": str(cols["title"][i].as_py()) if "title" in cols else None,
            }


def write_outputs(records, jsonl_path: Path, txt_path: Path):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as jf, txt_path.open(
        "w", encoding="utf-8"
    ) as tf:
        for rec in records:
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tf.write(rec["body"].rstrip() + "\n\n")
    return jsonl_path.stat().st_size, txt_path.stat().st_size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rifma", default=str(RIFMA))
    p.add_argument("--ars", default=str(ARS))
    p.add_argument("--en-raw", default=str(EN_RAW))
    p.add_argument("--en-form", default=str(EN_FORM))
    p.add_argument("--out-jsonl", default=str(OUT_JSONL))
    p.add_argument("--out-txt", default=str(OUT_TXT))
    p.add_argument("--skip-ars-nc", action="store_true")
    p.add_argument("--skip-en", action="store_true")
    args = p.parse_args()

    records = []

    rifma = Path(args.rifma)
    if rifma.is_file():
        chunk = list(iter_rifma(rifma))
        records.extend(chunk)
        print(f"rifma rows={len(chunk)}", flush=True)

    ars = Path(args.ars)
    if not args.skip_ars_nc and ars.is_file():
        chunk = list(iter_ars(ars))
        records.extend(chunk)
        print(f"ars rows={len(chunk)}", flush=True)
    elif args.skip_ars_nc:
        print("skip arspoetica (NC)", flush=True)

    if not args.skip_en:
        en_raw = Path(args.en_raw)
        chunk = list(iter_chicago_en(en_raw))
        records.extend(chunk)
        print(f"chicago_en rows={len(chunk)}", flush=True)
        en_form = Path(args.en_form)
        chunk = list(iter_poetry_form(en_form))
        records.extend(chunk)
        print(f"poetry_form_en rows={len(chunk)}", flush=True)

    jb, tb = write_outputs(records, Path(args.out_jsonl), Path(args.out_txt))
    by_src = {}
    by_lang = {}
    by_rhyme = {}
    for r in records:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
        by_lang[r["lang"]] = by_lang.get(r["lang"], 0) + 1
        by_rhyme[r["rhyme"]] = by_rhyme.get(r["rhyme"], 0) + 1
    print(
        f"wrote {args.out_jsonl} docs={len(records)} bytes={jb} "
        f"txt={args.out_txt} bytes={tb} by_src={by_src} by_lang={by_lang}",
        flush=True,
    )
    top = sorted(by_rhyme.items(), key=lambda x: -x[1])[:12]
    print(f"rhyme_top={top}", flush=True)


if __name__ == "__main__":
    main()
