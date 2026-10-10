"""Build JEF rhyme-classifier dataset: gold (RIFMA/Chicago) + negatives + Ars/Genius."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

RIFMA = Path("data/corpus/poetry/schema/rifma_dataset.json")
ARS = Path("data/corpus/poetry/schema/arspoetica.json")
EN_RAW = Path("data/corpus/poetry/schema/en/rhymedata/english_raw")
GENIUS_DIR = Path("data/corpus")
OUT_DIR = Path("data/rlaif/jef")
NOTES = Path("var/jef_data_notes.md")

SCHEME_BUCKETS = ("ABAB", "ABBA", "AABB", "A-A", "other", "free")
_BRACKET_ONLY = re.compile(r"^\s*[\[\(]([^\]\)]+)[\]\)]\s*$")


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _norm_scheme(raw) -> str:
    if raw is None:
        return "free"
    s = str(raw).strip()
    if not s or set(s) <= {"-", "*"}:
        return "free"
    parts = s.split()
    if len(parts) >= 2 and all(len(p) <= 1 for p in parts):
        letters = "".join(p.upper() if p != "*" else "*" for p in parts)
    else:
        letters = s.upper().replace(" ", "")
    compact = letters.replace("*", "")
    if compact in ("ABAB", "BABA"):
        return "ABAB"
    if compact in ("ABBA", "BAAB"):
        return "ABBA"
    if compact in ("AABB", "BBAA", "AA"):
        return "AABB"
    if compact in ("A-A", "-A-A", "A--A", "A-A-"):
        return "A-A"
    if re.fullmatch(r"A+", compact):
        return "AABB"
    if set(compact) <= {"-"} or not compact:
        return "free"
    return "other"


def _bin_label(scheme: str) -> str:
    return "no_rhyme" if scheme == "free" else "rhyme"


def _row(
    text: str,
    scheme: str | None,
    lang: str,
    source: str,
    *,
    soft: bool = False,
    require_scheme: bool = True,
) -> dict | None:
    body = _lines(text)
    if len(body) < 2:
        return None
    if len(body) > 16:
        body = body[:16]
    rec: dict = {
        "text": "\n".join(body),
        "lang": lang,
        "source": source,
        "soft": soft,
        "lines": len(body),
    }
    if scheme is None and not require_scheme:
        return rec
    scheme_n = _norm_scheme(scheme)
    rec["label_scheme"] = scheme_n
    rec["label_bin"] = _bin_label(scheme_n)
    return rec


def iter_rifma(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        rec = _row(row.get("poem_text") or "", row.get("rhyme_scheme"), "ru", "rifma", soft=False)
        if rec:
            yield rec


def iter_chicago(raw_dir: Path):
    if not raw_dir.is_dir():
        return
    for path in sorted(raw_dir.glob("*.txt")):
        rhyme = None
        buf: list[str] = []

        def flush():
            nonlocal buf, rhyme
            body = list(buf)
            buf = []
            scheme = rhyme
            rhyme = None
            if len(body) < 2:
                return None
            return _row("\n".join(body), scheme, "en", "chicago", soft=False)

        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.rstrip()
            if line.startswith("AUTHOR ") or line.startswith("TITLE "):
                rec = flush()
                if rec:
                    yield rec
                continue
            if line.startswith("RHYME-POEM"):
                continue
            if line.startswith("RHYME"):
                rec = flush()
                if rec:
                    yield rec
                rhyme = line[len("RHYME") :].strip()
                continue
            if line.strip():
                buf.append(line.strip())
        rec = flush()
        if rec:
            yield rec


def iter_arspoetica(path: Path, max_chunks: int | None = None):
    if not path.is_file():
        return
    rows = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for row in rows:
        text = row.get("poem_text") or ""
        stanzas = re.split(r"\n\s*\n", text.strip())
        for stanza in stanzas:
            body = _lines(stanza)
            if len(body) < 2:
                continue
            chunks = [body]
            if len(body) > 12:
                chunks = [body[k : k + 8] for k in range(0, len(body), 8) if len(body[k : k + 8]) >= 2]
            for chunk in chunks:
                rec = _row(
                    "\n".join(chunk),
                    None,
                    "ru",
                    "arspoetica",
                    soft=False,
                    require_scheme=False,
                )
                if not rec:
                    continue
                yield rec
                n += 1
                if max_chunks is not None and n >= max_chunks:
                    return


def _genius_stanza_chunks(text: str) -> list[list[str]]:
    # Genius often inserts a blank line between every bar; split on section
    # tags only, then window long verses into 4–8 line chunks.
    sections: list[list[str]] = []
    buf: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if _BRACKET_ONLY.match(line):
            if len(buf) >= 2:
                sections.append(buf)
            buf = []
            continue
        buf.append(line)
    if len(buf) >= 2:
        sections.append(buf)
    out: list[list[str]] = []
    for body in sections:
        if len(body) <= 16:
            if len(body) >= 2:
                out.append(body)
            continue
        step = 8
        for k in range(0, len(body), step):
            piece = body[k : k + step]
            if len(piece) >= 4:
                out.append(piece)
            elif len(piece) >= 2 and k == 0:
                out.append(piece)
    return out


def iter_genius_slang(corpus_dir: Path, rng: random.Random, max_chunks: int = 4000):
    files = sorted(corpus_dir.glob("genius_*.json"))
    if not files:
        return
    rng.shuffle(files)
    n = 0
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        songs = payload.get("songs") or []
        for song in songs:
            text = song.get("text") or song.get("lyrics") or ""
            for body in _genius_stanza_chunks(text):
                rec = _row(
                    "\n".join(body),
                    None,
                    "ru",
                    "genius_slang",
                    soft=False,
                    require_scheme=False,
                )
                if not rec:
                    continue
                yield rec
                n += 1
                if n >= max_chunks:
                    return


def make_negatives(pos: list[dict], rng: random.Random, per_pos: float = 0.5) -> list[dict]:
    neg = []
    for rec in pos:
        if rec.get("label_scheme") == "free" or rec.get("label_bin") == "no_rhyme":
            continue
        if "label_scheme" not in rec:
            continue
        if rng.random() > per_pos:
            continue
        lines = _lines(rec["text"])
        if len(lines) < 2:
            continue
        shuffled = lines[:]
        rng.shuffle(shuffled)
        if shuffled == lines:
            shuffled = list(reversed(lines))
        neg.append(
            {
                "text": "\n".join(shuffled),
                "label_scheme": "free",
                "label_bin": "no_rhyme",
                "lang": rec["lang"],
                "source": f"{rec['source']}_shuffle",
                "soft": False,
                "lines": len(shuffled),
            }
        )
    return neg


def label_schemes_with_base_jeff(
    rows: list[dict],
    *,
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
    device: str | None = None,
    conf_threshold: float = 0.55,
    max_rows: int | None = None,
    batch_size: int = 4,
) -> list[dict]:
    """Soft-label unlabeled rows via base Jeff/Qwen zeroshot.

    Returns updated copies for rows that get a scheme. Raises on missing deps / load failure
    so callers can skip and keep soft=false honestly.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(f"transformers/torch required for base Jeff labeling: {exc}") from exc

    targets = [r for r in rows if "label_scheme" not in r]
    if max_rows is not None:
        targets = targets[:max_rows]
    if not targets:
        return []

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device is None else None,
        trust_remote_code=True,
    )
    if device is not None:
        model = model.to(device)
    model.eval()

    allowed = ", ".join(SCHEME_BUCKETS)
    labeled: list[dict] = []

    def _parse_scheme(gen: str) -> tuple[str | None, float]:
        m = re.search(
            r"\b(ABAB|ABBA|AABB|A-A|other|free)\b",
            gen,
            flags=re.IGNORECASE,
        )
        if not m:
            return None, 0.0
        return _norm_scheme(m.group(1)), 0.7

    with torch.inference_mode():
        for i in range(0, len(targets), batch_size):
            batch = targets[i : i + batch_size]
            prompts = []
            for rec in batch:
                prompts.append(
                    "Classify the rhyme scheme of this stanza. "
                    f"Reply with exactly one label from [{allowed}].\n\n"
                    f"{rec['text']}\n\nLabel:"
                )
            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            for rec, seq in zip(batch, out):
                gen = tok.decode(seq[inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                scheme, conf = _parse_scheme(gen)
                if scheme is None or conf < conf_threshold:
                    continue
                upd = dict(rec)
                upd["label_scheme"] = scheme
                upd["label_bin"] = _bin_label(scheme)
                upd["soft"] = True
                upd["scheme_conf"] = conf
                labeled.append(upd)
            if (i // batch_size) % 50 == 0:
                print(
                    f"jeff_soft progress={min(i + batch_size, len(targets))}/{len(targets)} "
                    f"labeled={len(labeled)}",
                    flush=True,
                )
    return labeled


def split_rows(rows: list[dict], valid_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n = len(rows)
    n_test = max(1, int(n * test_ratio)) if n >= 10 else max(0, min(1, n // 10))
    n_valid = max(1, int(n * valid_ratio)) if n >= 10 else max(0, min(1, n // 10))
    if n_test + n_valid >= n:
        n_test = max(1, n // 10)
        n_valid = max(1, n // 10)
        if n_test + n_valid >= n:
            n_test = 1 if n > 2 else 0
            n_valid = 1 if n > 1 else 0
    test_i = set(idx[:n_test])
    valid_i = set(idx[n_test : n_test + n_valid])
    train, valid, test = [], [], []
    for i, row in enumerate(rows):
        if i in test_i:
            test.append(row)
        elif i in valid_i:
            valid.append(row)
        else:
            train.append(row)
    return train, valid, test


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_notes(
    path: Path,
    *,
    soft_labeled: int,
    soft_skipped: bool,
    soft_reason: str,
    counts: dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# JEF dataset notes",
        "",
        "## Build",
        "",
        "- Gold binary+scheme: RIFMA + Chicago Rhyming Poetry (`soft=false`).",
        "- Negatives: line-shuffle of gold rhyming stanzas → `label_bin=no_rhyme`, `label_scheme=free`.",
        "- ArsPoetica (CC-BY-NC-4.0, NC ok for this research pipeline): stanza chunks included.",
        "- Genius slang: stanza chunks from `data/corpus/genius_*.json`.",
        "- Splits: train/valid/test, seed=8.",
        "",
        "## Soft scheme labels (base Jeff)",
        "",
    ]
    if soft_skipped:
        lines.extend(
            [
                f"- **Gap:** base Jeff zeroshot labeling was **not** applied ({soft_reason}).",
                "- Ars/Genius rows shipped with `soft=false` and **no** `label_scheme` / `label_bin`.",
                "- Hook ready: `label_schemes_with_base_jeff(...)` in `training/scripts/build_jef_dataset.py`.",
                "- Re-run with `--with-base-jeff` when GPU + transformers model are available.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- Soft-labeled rows from base Jeff: **{soft_labeled}** (`soft=true`).",
                "",
            ]
        )
    lines.extend(
        [
            "## Counts",
            "",
            f"- train={counts.get('train')} valid={counts.get('valid')} test={counts.get('test')}",
            f"- by_source={counts.get('by_source')}",
            f"- bin={counts.get('bin')}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rifma", default=str(RIFMA))
    p.add_argument("--ars", default=str(ARS))
    p.add_argument("--en-raw", default=str(EN_RAW))
    p.add_argument("--genius-dir", default=str(GENIUS_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--neg-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--skip-en", action="store_true")
    p.add_argument("--skip-ars", action="store_true")
    p.add_argument("--skip-genius", action="store_true")
    p.add_argument("--ars-max", type=int, default=8000)
    p.add_argument("--genius-max", type=int, default=4000)
    p.add_argument("--with-base-jeff", action="store_true")
    p.add_argument("--jeff-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--jeff-max", type=int, default=2000)
    p.add_argument("--jeff-conf", type=float, default=0.55)
    args = p.parse_args()

    gold: list[dict] = []
    rifma = Path(args.rifma)
    if rifma.is_file():
        chunk = list(iter_rifma(rifma))
        gold.extend(chunk)
        print(f"rifma={len(chunk)}", flush=True)

    if not args.skip_en:
        chunk = list(iter_chicago(Path(args.en_raw)))
        gold.extend(chunk)
        print(f"chicago={len(chunk)}", flush=True)

    rng = random.Random(args.seed)
    negs = make_negatives(gold, rng, per_pos=args.neg_ratio)
    print(f"negatives={len(negs)}", flush=True)

    unlabeled: list[dict] = []
    if not args.skip_ars:
        chunk = list(iter_arspoetica(Path(args.ars), max_chunks=args.ars_max))
        unlabeled.extend(chunk)
        print(f"arspoetica={len(chunk)}", flush=True)

    if not args.skip_genius:
        chunk = list(iter_genius_slang(Path(args.genius_dir), rng, max_chunks=args.genius_max))
        unlabeled.extend(chunk)
        print(f"genius_slang={len(chunk)}", flush=True)

    soft_labeled = 0
    soft_skipped = True
    soft_reason = "not requested (--with-base-jeff off)"
    if args.with_base_jeff and unlabeled:
        soft_skipped = False
        soft_reason = ""
        try:
            labeled = label_schemes_with_base_jeff(
                unlabeled,
                model_id=args.jeff_model,
                conf_threshold=args.jeff_conf,
                max_rows=args.jeff_max,
            )
            soft_labeled = len(labeled)
            by_text = {r["text"]: r for r in labeled}
            unlabeled = [by_text.get(r["text"], r) for r in unlabeled]
            print(f"soft_jeff_labeled={soft_labeled}", flush=True)
        except Exception as exc:  # noqa: BLE001 — honest gap path
            soft_skipped = True
            soft_reason = f"{type(exc).__name__}: {exc}"
            print(f"soft_jeff_skipped={soft_reason}", flush=True)
    elif args.with_base_jeff:
        soft_reason = "no unlabeled Ars/Genius rows"

    rows = gold + negs + unlabeled
    print(f"total={len(rows)}", flush=True)

    train, valid, test = split_rows(rows, args.valid_ratio, args.test_ratio, args.seed)
    out = Path(args.out_dir)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "valid.jsonl", valid)
    write_jsonl(out / "test.jsonl", test)

    def stats(name, xs):
        c = Counter(r.get("label_scheme", "<none>") for r in xs)
        b = Counter(r.get("label_bin", "<none>") for r in xs)
        s = Counter(r.get("source", "?") for r in xs)
        soft_n = sum(1 for r in xs if r.get("soft"))
        print(
            f"{name} n={len(xs)} soft={soft_n} bin={dict(b)} "
            f"scheme={c.most_common(8)} src={s.most_common(8)}",
            flush=True,
        )
        return b, s

    b_all: Counter = Counter()
    s_all: Counter = Counter()
    for name, xs in (("train", train), ("valid", valid), ("test", test)):
        b, s = stats(name, xs)
        b_all.update(b)
        s_all.update(s)

    _write_notes(
        NOTES,
        soft_labeled=soft_labeled,
        soft_skipped=soft_skipped,
        soft_reason=soft_reason,
        counts={
            "train": len(train),
            "valid": len(valid),
            "test": len(test),
            "by_source": dict(s_all),
            "bin": dict(b_all),
        },
    )
    print(f"wrote {out / 'train.jsonl'} {out / 'valid.jsonl'} {out / 'test.jsonl'}", flush=True)
    print(f"notes {NOTES}", flush=True)


if __name__ == "__main__":
    main()
