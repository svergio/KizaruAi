"""Контрольный набор для гейта JEF (судья рифмы) перед RL.

Набор собирается из русского рэпа корпуса и не пересекается с данными, на которых учился JEF.
Категории:
  положительные   pos_couplet, pos_quatrain — настоящие рифмованные двустишия и четверостишия
  трудные негативы neg_shift (строки одной песни через одну, концы не рифмуются),
                   neg_near_rhyme (последнее слово заменено на «почти рифму»),
                   neg_unrhymed (настоящие подряд идущие строки без рифмы)
  ловушки          trap_repeat_word, trap_yeah, trap_english, trap_tags, trap_fragment, trap_prompt_copy

    python -m training.jef_challenge build --out data/rlaif/jef_challenge/ --seed 1
    python -m training.jef_challenge eval --judge checkpoints/jef_rhyme_lora

Гейт: у положительных доля оценок > 0.8, у негативов и ловушек доля оценок < 0.2
должна быть не меньше --min-frac (по умолчанию 0.8) в каждой категории.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Хелперы пулов живут в training/pools.py; реэкспорт оставлен для старого импорта из jef_challenge.
from training.pools import (  # noqa: F401
    DEFAULT_EXCLUDE_POOL,
    RU_RAP_EXCLUDE_FILE,
    load_ru_rap_exclude,
    load_ru_rap_exclude_pools,
    pool_key,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "dataset" / "lyrics.jsonl"
DEFAULT_JEF_DIR = ROOT / "data" / "rlaif" / "jef"
DEFAULT_OUT = ROOT / "data" / "rlaif" / "jef_challenge"
DEFAULT_POOLS_DIR = ROOT / "training" / "artist_pools"
RAP_POOL_FILES = ("00_existing.txt", "ru_expand.txt")
CHALLENGE_FILE = "challenge.jsonl"
MANIFEST_FILE = "manifest.json"

CATEGORIES = {
    "pos_couplet": ("positive", "rhyme"),
    "pos_quatrain": ("positive", "rhyme"),
    "neg_shift": ("hard_negative", "no_rhyme"),
    "neg_near_rhyme": ("hard_negative", "no_rhyme"),
    "neg_unrhymed": ("hard_negative", "no_rhyme"),
    "trap_repeat_word": ("trap", "no_rhyme"),
    "trap_yeah": ("trap", "no_rhyme"),
    "trap_english": ("trap", "no_rhyme"),
    "trap_tags": ("trap", "no_rhyme"),
    "trap_fragment": ("trap", "no_rhyme"),
    "trap_prompt_copy": ("trap", "no_rhyme"),
}
PER_SONG_CAP = 2
YEAH_FILLERS = ("yeah, yeah", "Yeah", "е-е-е", "эй, эй", "ага, ага", "skrrt, skrrt", "у-у-у", "йе, йе", "oh, yeah")
TAGS = ("[Припев]", "[Куплет 1]", "[Куплет 2]", "[Chorus]", "[Verse 1]", "[Интро]", "[x2]", "[Аутро]",
        "[Бридж]", "[Hook]", "[Припев: Kizaru]")

# ---------------------------------------------------------------- рифма (как в check_dataset)
WORD_RE = re.compile(r"[а-яёa-z0-9]+(?:-[а-яёa-z0-9]+)*")
VOWELS = set("аеёиоуыэюяaeiouy")


def _local_rhyme_zone(word):
    if not word:
        return None
    idx = [i for i, ch in enumerate(word) if ch in VOWELS]
    if not idx:
        return None
    return word[idx[-2] if len(idx) >= 2 else idx[-1]:]


try:  # та же логика, что в training/check_dataset.py
    from training.check_dataset import rhyme_zone
except Exception:  # pragma: no cover - модуль может правиться параллельно
    rhyme_zone = _local_rhyme_zone

_NORM_RE = re.compile(r"[\W_]+", re.U)
CYR_RE = re.compile(r"[а-яё]", re.I)
LAT_RE = re.compile(r"[a-z]", re.I)
UKR_RE = re.compile(r"[іїєґ]", re.I)
ARAB_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]")
BRACKET_RE = re.compile(r"[\[\](){}]")


def norm(text: str) -> str:
    return _NORM_RE.sub("", (text or "").lower())


def text_hash(text: str) -> str:
    try:
        from training.check_dataset import text_hash as _th
        return _th(text)
    except Exception:  # pragma: no cover
        return hashlib.sha1(norm(text).encode("utf-8")).hexdigest()


def line_hash(line: str) -> str:
    return hashlib.sha1(norm(line).encode("utf-8")).hexdigest()


def last_word(line: str):
    words = WORD_RE.findall(line.lower())
    return words[-1] if words else None


def cyr_share(text: str) -> float:
    c, l = len(CYR_RE.findall(text)), len(LAT_RE.findall(text))
    return c / (c + l) if c + l else 0.0


def letter_shares(text: str) -> tuple[float, float]:
    """Доли кириллицы и латиницы среди всех букв, включая арабское письмо."""
    c, l, a = len(CYR_RE.findall(text)), len(LAT_RE.findall(text)), len(ARAB_RE.findall(text))
    n = c + l + a
    return (c / n, l / n) if n else (0.0, 0.0)


def artist_keys(name: str) -> set[str]:
    """Ключи сравнения артиста: 'ЗАМАЙ (ZAMAY)' -> {замайzamay, замай, zamay}."""
    name = name or ""
    keys = {norm(name)}
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", name)
    if m:
        keys |= {norm(m.group(1)), norm(m.group(2))}
    return {k for k in keys if k}


# Совместимость: старый код импортировал DENY_ARTISTS из этого модуля.
DENY_ARTISTS = load_ru_rap_exclude()


def load_rap_pool(pools_dir: Path, files=RAP_POOL_FILES, deny=None) -> set[str]:
    """Ключи рэп-артистов из списков pools_dir минус pools_dir/ru_rap_exclude.txt (или явный deny)."""
    if deny is None:
        deny = load_ru_rap_exclude(Path(pools_dir) / RU_RAP_EXCLUDE_FILE.name)
    keys = set()
    for name in files:
        fp = Path(pools_dir) / name
        if not fp.is_file():
            continue
        for raw in fp.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            for part in re.split(r"-and-", raw, flags=re.I):
                k = pool_key(part)
                if k and k not in deny:
                    keys.add(k)
    return keys


def is_rap_artist(name: str, pool: set[str]) -> bool:
    return bool(artist_keys(name) & pool)


# ---------------------------------------------------------------- исключение данных JEF
def load_exclusion(jef_dir: Path, min_line_chars: int = 10):
    texts, lines = set(), set()
    for split in ("train", "valid", "test"):
        fp = Path(jef_dir) / f"{split}.jsonl"
        if not fp.is_file():
            continue
        for raw in fp.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            t = json.loads(raw).get("text") or ""
            texts.add(text_hash(t))
            for ln in t.split("\n"):
                if len(norm(ln)) >= min_line_chars:
                    lines.add(line_hash(ln))
    return texts, lines


def song_excluded(text: str, excl, min_line_chars: int = 10) -> bool:
    texts, lines = excl
    if text_hash(text) in texts:
        return True
    return any(len(norm(ln)) >= min_line_chars and line_hash(ln) in lines for ln in text.split("\n"))


# ---------------------------------------------------------------- отбор песен
def _rank(seed: int, h: str) -> int:
    return int(hashlib.sha1(f"{seed}:{h}".encode()).hexdigest()[:15], 16)


def iter_corpus(path: Path):
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip():
                yield json.loads(raw)


def select_songs(corpus: Path, pool: set[str], excl, seed: int, max_ru: int, max_en: int):
    """Детерминированная по seed выборка: max_ru русских рэп-песен и max_en англоязычных (для ловушки)."""
    ru_heap, en_heap = [], []
    seen = set()
    stats = Counter()
    for row in iter_corpus(corpus):
        text = row.get("text") or ""
        if len(text.strip()) < 40:
            continue
        h = text_hash(text)
        if h in seen:
            stats["dupe"] += 1
            continue
        seen.add(h)
        share, lat = letter_shares(text)
        if share >= 0.85 and not UKR_RE.search(text):
            if not is_rap_artist(row.get("artist") or "", pool):
                stats["ru_not_rap"] += 1
                continue
            heap, cap = ru_heap, max_ru * 2
        elif lat >= 0.95:
            heap, cap = en_heap, max_en * 2
        else:
            stats["mixed"] += 1
            continue
        item = (-_rank(seed, h), h, row)
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    def finish(heap, cap, tag):
        out = []
        for _, h, row in sorted(heap, key=lambda x: (-x[0], x[1])):
            if song_excluded(row.get("text") or "", excl):
                stats[f"{tag}_excluded_jef"] += 1
                continue
            out.append((h, row))
            if len(out) >= cap:
                break
        return out

    ru = finish(ru_heap, max_ru, "ru")
    en = finish(en_heap, max_en, "en")
    stats["ru_songs"], stats["en_songs"] = len(ru), len(en)
    return ru, en, stats


# ---------------------------------------------------------------- строки и кандидаты
def ok_ru_line(line: str) -> bool:
    if BRACKET_RE.search(line) or UKR_RE.search(line) or ARAB_RE.search(line):
        return False
    words = line.split()
    if not (2 <= len(words) <= 12 and 8 <= len(line) <= 90):
        return False
    if cyr_share(line) < 0.9:
        return False
    w = last_word(line)
    return bool(w and len(w) >= 3 and rhyme_zone(w))


def ok_en_line(line: str) -> bool:
    words = line.split()
    return (3 <= len(words) <= 12 and 12 <= len(line) <= 90 and cyr_share(line) == 0.0
            and len(LAT_RE.findall(line)) >= 8 and not BRACKET_RE.search(line)
            and not ARAB_RE.search(line))


def song_lines(text: str):
    out = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        w = last_word(ln)
        out.append({"text": ln, "ok": ok_ru_line(ln), "word": w, "zone": rhyme_zone(w) if w else None})
    return out


def rhymes(a: dict, b: dict) -> bool:
    return bool(a["zone"]) and a["zone"] == b["zone"] and a["word"] != b["word"]


def clearly_unrhymed(items) -> bool:
    zones = [x["zone"] for x in items]
    tails = [x["word"][-2:] for x in items]
    return all(zones) and len(set(zones)) == len(zones) and len(set(tails)) == len(tails)


def replace_last_word(line: str, new: str) -> str:
    matches = list(WORD_RE.finditer(line.lower()))
    if not matches:
        return line
    m = matches[-1]
    return line[:m.start()] + new + line[m.end():]


def build_vocab(songs) -> dict:
    """Концовки строк корпуса: хвост из 2 и 1 буквы -> слова."""
    vocab = defaultdict(set)
    for _, lines in songs:
        for x in lines:
            if x["ok"]:
                w = x["word"]
                vocab["2:" + w[-2:]].add(w)
                vocab["1:" + w[-1:]].add(w)
    return {k: sorted(v) for k, v in vocab.items()}


def near_rhyme_word(word: str, avoid_zones: set, vocab: dict, rng: random.Random):
    """Слово с тем же хвостом (2 или 1 буква), но другой зоной рифмы и другой ударной гласной зоны.

    Другая первая гласная зоны отсекает ассонансы вроде «говорим / одиноким», которые в рэпе считаются рифмой.
    """
    avoid_vowels = {z[0] for z in avoid_zones if z}
    for key in ("2:" + word[-2:], "1:" + word[-1:]):
        cands = [w for w in vocab.get(key, ()) if w != word and len(w) >= 3
                 and rhyme_zone(w) not in avoid_zones and (rhyme_zone(w) or "?")[0] not in avoid_vowels]
        if cands:
            return rng.choice(cands)
    return None


def candidates_for_song(h, row, lines, vocab, rng):
    """Кандидаты по категориям из одной русской песни (не больше PER_SONG_CAP на категорию)."""
    out = defaultdict(list)
    n = len(lines)
    ok = [x["ok"] for x in lines]

    def add(cat, texts, extra=None):
        if len(out[cat]) < PER_SONG_CAP:
            rec = {"category": cat, "text": "\n".join(texts), "song_hash": h, "song_hashes": [h],
                   "source_id": row.get("id"), "artist": row.get("artist")}
            if extra:
                rec.update(extra)
            out[cat].append(rec)

    order = list(range(n))
    rng.shuffle(order)
    for i in order:
        if i + 1 < n and ok[i] and ok[i + 1]:
            a, b = lines[i], lines[i + 1]
            if rhymes(a, b):
                add("pos_couplet", [a["text"], b["text"]], {"scheme": "AA"})
                new = near_rhyme_word(b["word"], {a["zone"], b["zone"]}, vocab, rng)
                if new:
                    add("neg_near_rhyme", [a["text"], replace_last_word(b["text"], new)],
                        {"replaced": [b["word"], new]})
            elif clearly_unrhymed([a, b]):
                add("trap_prompt_copy", [a["text"], b["text"], a["text"], b["text"]])
                f = rng.choice(YEAH_FILLERS)
                pattern = rng.choice(("a_f_b_f", "f_f_a_f", "a_b_f_f"))
                lines_y = {"a_f_b_f": [a["text"], f, b["text"], f], "f_f_a_f": [f, f, a["text"], f],
                           "a_b_f_f": [a["text"], b["text"], f, f]}[pattern]
                add("trap_yeah", lines_y)
                t1, t2 = rng.sample(TAGS, 2)
                tag_variant = rng.choice(("tag_line", "tag_suffix"))
                if tag_variant == "tag_line":
                    add("trap_tags", [t1, a["text"], t2, b["text"]])
                else:
                    add("trap_tags", [f"{a['text']} {t1}", f"{b['text']} {t1}"])
        if i + 3 < n and all(ok[i:i + 4]):
            q = lines[i:i + 4]
            scheme = None
            if rhymes(q[0], q[1]) and rhymes(q[2], q[3]) and q[0]["zone"] != q[2]["zone"]:
                scheme = "AABB"
            elif rhymes(q[0], q[2]) and rhymes(q[1], q[3]) and q[0]["zone"] != q[1]["zone"]:
                scheme = "ABAB"
            elif rhymes(q[0], q[3]) and rhymes(q[1], q[2]) and q[0]["zone"] != q[1]["zone"]:
                scheme = "ABBA"
            if scheme:
                add("pos_quatrain", [x["text"] for x in q], {"scheme": scheme})
            elif clearly_unrhymed(q):
                add("neg_unrhymed", [x["text"] for x in q])
                w = q[0]["word"]
                add("trap_repeat_word", [q[0]["text"]] + [replace_last_word(x["text"], w) for x in q[1:]],
                    {"repeated": w})
                frag = [" ".join(x["text"].split()[:rng.choice((1, 2))]) for x in q[:rng.choice((1, 2, 3))]]
                add("trap_fragment", frag)
    # строки одной песни через одну и больше, чтобы концы не рифмовались
    good = [i for i in range(n) if ok[i]]
    for _ in range(6):
        if len(good) < 7:
            break
        idx = sorted(rng.sample(good, 4))
        if any(b - a < 2 for a, b in zip(idx, idx[1:])):
            continue
        items = [lines[j] for j in idx]
        if clearly_unrhymed(items):
            add("neg_shift", [x["text"] for x in items], {"line_idx": idx})
    return out


def english_traps(en_songs, rng, n):
    pool = []
    for h, row in en_songs:
        lines = [ln.strip() for ln in (row.get("text") or "").split("\n") if ok_en_line(ln.strip())]
        rng.shuffle(lines)
        pool.extend((h, row, ln) for ln in lines[:3])
    out = []
    if len(pool) < 4:
        return out
    for _ in range(n * 3):
        if len(out) >= n:
            break
        pick = rng.sample(pool, 4)
        if len({p[0] for p in pick}) < 4:
            continue
        hs = [p[0] for p in pick]
        out.append({"category": "trap_english", "text": "\n".join(p[2] for p in pick), "song_hash": hs[0],
                    "song_hashes": hs, "source_id": pick[0][1].get("id"), "artist": pick[0][1].get("artist")})
    return out


def build(corpus: Path, out_dir: Path, seed: int, per_category: int = 400, jef_dir: Path = DEFAULT_JEF_DIR,
          pools_dir: Path = DEFAULT_POOLS_DIR, max_songs: int = 6000, max_en_songs: int = 1500) -> dict:
    rng = random.Random(seed)
    pool = load_rap_pool(pools_dir)
    if not pool:
        raise SystemExit(f"нет рэп-пулов артистов в {pools_dir}")
    excl = load_exclusion(jef_dir)
    ru, en, stats = select_songs(corpus, pool, excl, seed, max_songs, max_en_songs)
    parsed = [((h, row), song_lines(row.get("text") or "")) for h, row in ru]
    vocab = build_vocab([(None, lines) for _, lines in parsed])
    cands = defaultdict(list)
    for (h, row), lines in parsed:
        for cat, recs in candidates_for_song(h, row, lines, vocab, rng).items():
            cands[cat].extend(recs)
    cands["trap_english"] = english_traps(en, rng, per_category)

    records, counts, available = [], {}, {}
    seen_text = set()
    for cat in CATEGORIES:
        group, label = CATEGORIES[cat]
        uniq = []
        for r in cands.get(cat, []):
            k = text_hash(r["text"])
            if k not in seen_text:
                seen_text.add(k)
                uniq.append(r)
        available[cat] = len(uniq)
        rng.shuffle(uniq)
        chosen = uniq[:per_category]
        for i, r in enumerate(chosen):
            r.update({"id": f"{cat}-{i:04d}", "group": group, "label_bin": label,
                      "lang": "en" if cat == "trap_english" else "ru"})
            records.append(r)
        counts[cat] = len(chosen)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ("id", "category", "group", "label_bin", "text", "lang", "song_hash", "song_hashes",
            "source_id", "artist")
    with open(out_dir / CHALLENGE_FILE, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            row = {k: r.get(k) for k in keys}
            row.update({k: v for k, v in r.items() if k not in row})
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"seed": seed, "per_category": per_category, "counts": counts, "available": available,
                "total": len(records), "songs": dict(stats), "corpus": str(corpus), "jef_dir": str(jef_dir),
                "rap_pool_size": len(pool),
                "artists_top": Counter(r["artist"] for r in records if r["lang"] == "ru").most_common(30),
                "short": [c for c, n in counts.items() if n < per_category]}
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------- оценка
def load_challenge(path: Path) -> list[dict]:
    path = Path(path)
    if path.is_dir():
        path = path / CHALLENGE_FILE
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def evaluate(rows, score_fn, *, hi: float = 0.8, lo: float = 0.2, min_frac: float = 0.8, limit: int = 0) -> dict:
    by_cat = defaultdict(list)
    for r in rows:
        if not limit or len(by_cat[r["category"]]) < limit:
            by_cat[r["category"]].append(r)
    report = {"hi": hi, "lo": lo, "min_frac": min_frac, "categories": {}}
    ok_all = True
    for cat in CATEGORIES:
        items = by_cat.get(cat, [])
        group = CATEGORIES[cat][0]
        if not items:
            report["categories"][cat] = {"group": group, "n": 0, "mean": None, "frac_ok": 0.0, "pass": False}
            ok_all = False
            continue
        scores = [float(s) for s in score_fn([r["text"] for r in items])]
        if group == "positive":
            good = sum(s > hi for s in scores)
        else:
            good = sum(s < lo for s in scores)
        frac = good / len(scores)
        passed = frac >= min_frac
        ok_all &= passed
        report["categories"][cat] = {"group": group, "n": len(scores), "mean": sum(scores) / len(scores),
                                     "frac_ok": frac, "pass": passed}
    report["gate"] = "PASS" if ok_all else "FAIL"
    return report


def render_report(rep: dict) -> str:
    out = [f"{'категория':<18} {'группа':<14} {'n':>5} {'среднее':>8} {'доля ок':>8}  гейт"]
    for cat, r in rep["categories"].items():
        mean = "-" if r["mean"] is None else f"{r['mean']:.3f}"
        cond = f">{rep['hi']}" if r["group"] == "positive" else f"<{rep['lo']}"
        out.append(f"{cat:<18} {r['group']:<14} {r['n']:>5} {mean:>8} {r['frac_ok']:>8.2%}  "
                   f"{'PASS' if r['pass'] else 'FAIL'} ({cond})")
    out.append(f"ИТОГ: {rep['gate']} (порог доли {rep['min_frac']:.0%})")
    return "\n".join(out)


def load_judge(checkpoint: Path, *, model_id: str, device: str = "cuda", max_len: int = 256,
               batch_size: int = 8, gpu_max_mem: str = "6GiB", cpu_max_mem: str = "48GiB",
               offload: str = "var/jef_offload"):
    """Судья из чекпойнта JEF, тем же кодом загрузки, что train_jef / train_rl_deepseek."""
    import torch

    from training.train_jef import BIN_LABELS, build_model

    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    model, tok = build_model(model_id, use_cuda, lora_r=16, lora_alpha=32, device=dev,
                             adapter_dir=Path(checkpoint), gpu_max_mem=gpu_max_mem if use_cuda else "0GiB",
                             cpu_max_mem=cpu_max_mem, offload_folder=offload, for_train=False)
    model.eval()
    rhyme_idx = BIN_LABELS.index("rhyme")

    @torch.inference_mode()
    def score(texts):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok([t or " " for t in texts[i:i + batch_size]], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            first = next(model.parameters())
            logits = model(input_ids=enc["input_ids"].to(first.device),
                           attention_mask=enc["attention_mask"].to(first.device), use_scheme=False).logits
            out.extend(torch.softmax(logits.float(), dim=-1)[:, rhyme_idx].tolist())
        return out

    return score


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.jef_challenge", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="собрать контрольный набор")
    b.add_argument("--out", type=Path, default=DEFAULT_OUT)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="lyrics.jsonl (поля text, artist, id)")
    b.add_argument("--jef-dir", type=Path, default=DEFAULT_JEF_DIR, help="данные JEF для исключения")
    b.add_argument("--pools-dir", type=Path, default=DEFAULT_POOLS_DIR, help="списки рэп-артистов")
    b.add_argument("--per-category", type=int, default=400)
    b.add_argument("--max-songs", type=int, default=6000)
    b.add_argument("--max-en-songs", type=int, default=1500)
    e = sub.add_parser("eval", help="прогнать судью по набору и проверить гейт")
    e.add_argument("--judge", type=Path, required=True, help="папка адаптера JEF")
    e.add_argument("--data", type=Path, default=DEFAULT_OUT, help="папка или challenge.jsonl")
    e.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    e.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--max-len", type=int, default=256)
    e.add_argument("--limit", type=int, default=0, help="не больше N примеров на категорию (0 = все)")
    e.add_argument("--min-frac", type=float, default=0.8)
    e.add_argument("--hi", type=float, default=0.8)
    e.add_argument("--lo", type=float, default=0.2)
    e.add_argument("--report", type=Path, help="куда записать json-отчёт")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if a.cmd == "build":
        m = build(a.corpus, a.out, a.seed, a.per_category, a.jef_dir, a.pools_dir, a.max_songs, a.max_en_songs)
        print(json.dumps({k: m[k] for k in ("counts", "available", "total", "songs", "short")},
                         ensure_ascii=False, indent=2))
        return 0
    rows = load_challenge(a.data)
    score = load_judge(a.judge, model_id=a.model, device=a.device, max_len=a.max_len, batch_size=a.batch_size)
    rep = evaluate(rows, score, hi=a.hi, lo=a.lo, min_frac=a.min_frac, limit=a.limit)
    rep["judge"] = str(a.judge)
    print(render_report(rep))
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if rep["gate"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
