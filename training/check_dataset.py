"""Проверка датасета: шум Genius, оставшиеся теги [..], рифма, дубли кэша.

Ничего не меняет в данных. Отчёт печатается в stdout, с --out пишется в markdown
(с --dry-run файл не пишется). Собрано из разовых скриптов var/tmp
(clean_dryrun, audit_corpus_noise*, corpus_study*, rhyme_clean).

    python -m training.check_dataset all
    python -m training.check_dataset noise --sample 25 --seed 1
    python -m training.check_dataset dupes --corpus-dir data/corpus
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = ROOT / "data" / "corpus"
DEFAULT_LYRICS = DEFAULT_CORPUS_DIR / "kizaru_lyrics.txt"

WORD_RE = re.compile(r"[а-яёa-z0-9]+(?:-[а-яёa-z0-9]+)*")
VOWELS = set("аеёиоуыэюяaeiouy")
JSON_HINT = ("apipath", "relationshiptype", "currentusermetadata", "lyricsstate", "annotatable")
BRACKET_RE = re.compile(r"\[([^\]]{1,80})\]")
LONG_LINE = 300

NOISE_PATTERNS = {
    "section_tag": re.compile(
        r"\[(?:Verse|Chorus|Bridge|Intro|Outro|Hook|Pre-Chorus|Refrain|Part\s*\d|"
        r"Куплет|Припев|Интро|Аутро|Бридж|Рефрен|Часть|Интерлюд|Переход|Текст песни)",
        re.I,
    ),
    "credits": re.compile(r"[\[(](?:Produced|Written|Mixed|Mastered|Recorded|Prod\.)\s", re.I),
    "contributors": re.compile(r"\d+\s+Contributors", re.I),
    "genius_ui": re.compile(r"Read\s+More|You might also like|(?:^|\s)Embed(?:$|\s)", re.I),
    "json_leak": re.compile(r'"\w+"\s*:\s*[\[{]|"view_\w+"|add_pinned_annotation'),
    "any_bracket": BRACKET_RE,
}


def norm_slug(slug: str) -> str:
    """Ключ артиста для поиска дублей. Кириллица не схлопывается в пустую строку."""
    from training.pools import pool_key

    return pool_key(slug) or slug.lower()


def song_texts(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    songs = data if isinstance(data, list) else data.get("songs") or []
    return [s.get("text") or s.get("lyrics") or "" for s in songs if isinstance(s, dict)]


def _prepare(text: str) -> str:
    from core.corpus.lyrics import prepare_corpus_text

    return prepare_corpus_text(text, genius=True)


def check_noise(corpus_dir: Path, sample: int, seed: int, prepare=None) -> dict:
    """Сколько песен содержат шум до и после штатной чистки."""
    prepare = prepare or _prepare
    files = sorted(corpus_dir.glob("genius_*.json"))
    rnd = random.Random(seed)
    picked = rnd.sample(files, min(sample, len(files))) if sample else files
    raw, after = Counter(), Counter()
    songs = dropped = 0
    for fp in picked:
        for body in song_texts(fp):
            if not body.strip():
                continue
            songs += 1
            clean = prepare(body)
            if not clean.strip():
                dropped += 1
            for key, pat in NOISE_PATTERNS.items():
                raw[key] += bool(pat.search(body))
                after[key] += bool(pat.search(clean))
    return {"files": len(picked), "songs": songs, "dropped": dropped,
            "raw": dict(raw), "after": dict(after)}


def _lines(text: str, drop_junk: bool) -> list[str]:
    out = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if drop_junk and (len(ln) > LONG_LINE or any(k in ln.lower() for k in JSON_HINT)):
            continue
        out.append(ln)
    return out


def check_brackets(text: str, top: int = 20) -> dict:
    """Оставшиеся [теги] и мусор из JSON в готовом корпусе."""
    lines = _lines(text, drop_junk=False)
    tags = Counter(m.strip().lower() for ln in lines for m in BRACKET_RE.findall(ln))
    json_lines = sum(1 for ln in lines if any(k in ln.lower() for k in JSON_HINT))
    long_lines = sum(1 for ln in lines if len(ln) > LONG_LINE)
    return {"lines": len(lines), "json_lines": json_lines, "long_lines": long_lines,
            "bracket_total": sum(tags.values()), "top_tags": tags.most_common(top)}


def rhyme_zone(word: str | None) -> str | None:
    """Суффикс от предпоследней гласной: грубая зона рифмы без ударений."""
    if not word:
        return None
    idx = [i for i, ch in enumerate(word) if ch in VOWELS]
    if not idx:
        return None
    return word[idx[-2] if len(idx) >= 2 else idx[-1]:]


def _last_word(line: str) -> str | None:
    words = WORD_RE.findall(line.lower())
    return words[-1] if words else None


def _zone_rate(seq: list[str], offset: int) -> float:
    pairs = hits = 0
    for a, b in zip(seq, seq[offset:]):
        if a == b:
            continue
        za = rhyme_zone(a)
        if not za:
            continue
        pairs += 1
        hits += za == rhyme_zone(b)
    return 100.0 * hits / pairs if pairs else 0.0


def check_rhyme(text: str, seed: int = 0, offsets=(1, 2, 3, 4)) -> dict:
    """Насколько соседние строки рифмуются по сравнению со случайными парами."""
    seq = [w for w in (_last_word(ln) for ln in _lines(text, drop_junk=True)) if w]
    shuffled = seq[:]
    random.Random(seed).shuffle(shuffled)
    base = _zone_rate(shuffled, 1)
    window = total = 0
    for i in range(len(seq) - 3):
        za = rhyme_zone(seq[i])
        if not za:
            continue
        total += 1
        window += any(rhyme_zone(seq[i + k]) == za and seq[i + k] != seq[i] for k in (1, 2, 3))
    return {"words": len(seq), "baseline": base,
            "by_offset": {o: _zone_rate(seq, o) for o in offsets},
            "rhyme_in_next3": 100.0 * window / total if total else 0.0}


def check_dupes(corpus_dir: Path) -> dict:
    """Пары genius_X-.json / genius_X.json и прочие слаги с одной нормой. Ничего не удаляет."""
    groups: dict[str, list[Path]] = {}
    for fp in sorted(corpus_dir.glob("genius_*.json")):
        groups.setdefault(norm_slug(fp.stem[len("genius_"):]), []).append(fp)
    dupes = []
    for key, fam in groups.items():
        if len(fam) < 2:
            continue
        info = [(fp.name, len([t for t in song_texts(fp) if t.strip()]), fp.stat().st_size) for fp in fam]
        keep = max(info, key=lambda x: (x[1], x[2], not x[0].endswith("-.json")))[0]
        dupes.append({"norm": key, "files": info, "keep": keep})
    dash_only = [f[0].name for f in groups.values() if len(f) == 1 and f[0].stem.endswith("-")]
    return {"groups": len(groups), "dupes": dupes, "dash_only": dash_only}


_NORM_RE = re.compile(r"[^\w]+", re.U)
CYR_RE = re.compile(r"[а-яё]", re.I)
LAT_RE = re.compile(r"[a-z]", re.I)
ARAB_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]")


def text_hash(text: str) -> str:
    """Хеш текста песни без регистра, пунктуации и пробелов."""
    import hashlib

    return hashlib.sha1(_NORM_RE.sub("", text.lower()).encode("utf-8")).hexdigest()


def check_textdupes(corpus_dir: Path, top: int = 20) -> dict:
    """Одинаковые тексты песен в разных файлах кэша, независимо от имени."""
    seen: dict[str, list[str]] = {}
    songs = 0
    for fp in sorted(corpus_dir.glob("genius_*.json")):
        for body in song_texts(fp):
            if len(body.strip()) < 40:
                continue
            songs += 1
            seen.setdefault(text_hash(body), []).append(fp.name)
    groups = {h: f for h, f in seen.items() if len(f) > 1}
    extra = sum(len(f) - 1 for f in groups.values())
    pairs = Counter()
    for files in groups.values():
        names = sorted(set(files))
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairs[(a, b)] += 1
    return {"songs": songs, "unique": len(seen), "extra_copies": extra,
            "dupe_share": 100.0 * extra / songs if songs else 0.0, "top_pairs": pairs.most_common(top)}


def line_script(line: str) -> str | None:
    """Письмо строки: cyr, lat, arab или None, если букв нет."""
    c, l, a = len(CYR_RE.findall(line)), len(LAT_RE.findall(line)), len(ARAB_RE.findall(line))
    if not (c or l or a):
        return None
    return max((c, "cyr"), (l, "lat"), (a, "arab"))[1]


def check_script(corpus_dir: Path, en_threshold: float = 70.0) -> dict:
    """Доли кириллицы, латиницы и арабского письма по артистам."""
    total = Counter()
    per_artist = []
    for fp in sorted(corpus_dir.glob("genius_*.json")):
        cnt = Counter()
        for body in song_texts(fp):
            for ln in body.splitlines():
                s = line_script(ln)
                if s:
                    cnt[s] += 1
        n = sum(cnt.values())
        if not n:
            continue
        total.update(cnt)
        per_artist.append((fp.name, n, {k: 100.0 * cnt[k] / n for k in ("cyr", "lat", "arab")}))
    n = sum(total.values()) or 1
    return {"lines": sum(total.values()),
            "share": {k: 100.0 * total[k] / n for k in ("cyr", "lat", "arab")},
            "arab_artists": [a for a, _, s in per_artist if s["arab"] > 10],
            "en_artists": [a for a, _, s in per_artist if s["lat"] > en_threshold],
            "per_artist": per_artist}


def render(results: dict) -> str:
    out = ["# Проверка датасета", ""]
    if "noise" in results:
        r = results["noise"]
        out += ["## Шум Genius до и после чистки", "",
                f"Файлов: {r['files']}, песен: {r['songs']}, выброшено чисткой целиком: {r['dropped']}.", "",
                "| Шум | До | После |", "|---|---|---|"]
        out += [f"| {k} | {r['raw'].get(k, 0)} | {r['after'].get(k, 0)} |" for k in NOISE_PATTERNS]
        out.append("")
    if "brackets" in results:
        r = results["brackets"]
        out += ["## Теги и мусор в корпусе", "",
                f"Строк: {r['lines']}, с JSON: {r['json_lines']}, длиннее {LONG_LINE}: {r['long_lines']}, "
                f"тегов [..]: {r['bracket_total']}.", ""]
        out += [f"- `[{t}]`: {c}" for t, c in r["top_tags"]]
        out.append("")
    if "rhyme" in results:
        r = results["rhyme"]
        base = r["baseline"] or 1e-9
        out += ["## Рифма", "", f"Слов в конце строк: {r['words']}, случайные пары: {r['baseline']:.2f}%.", "",
                "| Сдвиг | Совпадение зоны рифмы | Выше случайности |", "|---|---|---|"]
        out += [f"| +{o} | {v:.1f}% | x{v / base:.1f} |" for o, v in r["by_offset"].items()]
        out += ["", f"Рифма в следующих 3 строках: {r['rhyme_in_next3']:.1f}%.", ""]
    if "dupes" in results:
        r = results["dupes"]
        out += ["## Дубли кэша Genius", "",
                f"Групп: {r['groups']}, с дублями: {len(r['dupes'])}, только с дефисом: {len(r['dash_only'])}.", ""]
        for d in r["dupes"]:
            files = ", ".join(f"{n} ({s} песен, {b} Б)" for n, s, b in d["files"])
            out.append(f"- {d['norm']}: {files}; оставить {d['keep']}")
        out.append("")
    if "textdupes" in results:
        r = results["textdupes"]
        out += ["## Одинаковые тексты", "",
                f"Песен: {r['songs']}, уникальных: {r['unique']}, лишних копий: {r['extra_copies']} "
                f"({r['dupe_share']:.1f}%).", ""]
        out += [f"- {a} + {b}: {c}" for (a, b), c in r["top_pairs"]]
        out.append("")
    if "script" in results:
        r = results["script"]
        s = r["share"]
        out += ["## Письмо", "", f"Строк: {r['lines']}, кириллица {s['cyr']:.1f}%, латиница {s['lat']:.1f}%, "
                f"арабское {s['arab']:.1f}%.", "",
                f"Артистов с арабским письмом больше 10%: {len(r['arab_artists'])}: " + ", ".join(r["arab_artists"]), "",
                f"Артистов с латиницей больше порога: {len(r['en_artists'])}.", ""]
    return "\n".join(out)


CHECKS = ("noise", "brackets", "rhyme", "dupes", "textdupes", "script")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.check_dataset", description=__doc__.splitlines()[0])
    ap.add_argument("checks", nargs="+", choices=CHECKS + ("all",), help="какие проверки запустить")
    ap.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR, help="папка с genius_*.json")
    ap.add_argument("--lyrics", type=Path, default=DEFAULT_LYRICS, help="готовый корпус .txt для brackets/rhyme")
    ap.add_argument("--sample", type=int, default=25, help="сколько файлов артистов брать для noise (0 = все)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--en-threshold", type=float, default=70.0, help="порог латиницы, %%, для script")
    ap.add_argument("--out", type=Path, help="записать отчёт в markdown")
    ap.add_argument("--dry-run", action="store_true", help="ничего не писать на диск, только stdout")
    a = ap.parse_args(argv)
    todo = CHECKS if "all" in a.checks else tuple(dict.fromkeys(a.checks))
    results = {}
    text = None
    for name in todo:
        if name in ("brackets", "rhyme"):
            if not a.lyrics.is_file():
                print(f"нет файла корпуса: {a.lyrics}", file=sys.stderr)
                return 2
            text = text if text is not None else a.lyrics.read_text(encoding="utf-8")
        if name == "noise":
            results[name] = check_noise(a.corpus_dir, a.sample, a.seed)
        elif name == "brackets":
            results[name] = check_brackets(text)
        elif name == "rhyme":
            results[name] = check_rhyme(text, a.seed)
        elif name == "dupes":
            results[name] = check_dupes(a.corpus_dir)
        elif name == "textdupes":
            results[name] = check_textdupes(a.corpus_dir)
        elif name == "script":
            results[name] = check_script(a.corpus_dir, a.en_threshold)
    report = render(results)
    sys.stdout.reconfigure(encoding="utf-8")
    print(report)
    if a.out and not a.dry_run:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
