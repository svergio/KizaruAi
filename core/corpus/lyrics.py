import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from core.corpus.clean import NewlineMode, normalize_song_body, strip_corpus_noise

SEQ_LEN = 256
BATCH_SIZE = 16

DEFAULT_CORPUS = Path("data/corpus/kizaru_lyrics.txt")
FALLBACK_MESSAGES = Path("data/corpus/messages_fallback.csv")

_KUPLET = "\u043a\u0443\u043f\u043b\u0435\u0442"

_GENIUS_SECTION_HINT = re.compile(
    r"(?i)(?:"
    r"intro|outro|bridge|chorus|verse|hook|pre-?chorus|post-?chorus|instrumental|"
    r"refrain|interlude|skit|transition|"
    + _KUPLET
    + r"|припев|бридж|интро|аутро|текст|перевод|translation|"
    r"рефрен|хук|интерлюд|переход|постприпев|скит|реклама|часть|"
    r"part\s*\d+|verse\s*\d+|chorus\s*\d+|"
    + _KUPLET
    + r"\s*\d+|припев\s*\d+|"
    + _KUPLET
    + r"\s*\d+\s*:|verse\s*\d+\s*:|"
    r"текст\s+песни|song\s+text|"
    r"(?:produced|written|mixed|mastered|recorded)\s+by|"
    r"(?:слова|words|lyrics)\b"
    r")"
)

_GENIUS_NOISE_LINE = re.compile(
    r"(?i)^\s*(?:"
    r"\d+\s+contributors?(?:\s*and\s+translations?|\s*\/\s*)?(?:\s*\w+)*\s*$|"
    r"(?:\d+\s+)?contributors?(?:translations?|english|russian)\s*$|"
    r"you might also like\s*$|"
    r"read\s+more\s*$|"
    r"embed\s*$|"
    r"translations?\s*$|"
    r"english\s*$|"
    r"russian\s*$|"
    r"\d+\s+embed\s*$"
    r")"
)

_BRACKET_ONLY = re.compile(r"^\s*[\[\(]([^\]\)]+)[\]\)]\s*$")
_LEADING_TAG = re.compile(r"^\s*(?:\[[^\]]+\]|\([^\)]+\))\s*")


def _section_label_core(label):
    core = label.strip()
    if ":" in core:
        core = core.split(":", 1)[0].strip()
    return core


def _is_genius_section_label(label):
    text = _section_label_core(label)
    if not text:
        return True
    if text in ("?", "..", "..."):
        return True
    if re.match(r"(?i)^\d+$", text):
        return True
    if _GENIUS_SECTION_HINT.search(text):
        return True
    if re.match(
        rf"(?i)^(?:{_KUPLET}|припев|verse|chorus|bridge|intro|outro|рефрен|хук|"
        r"интерлюд|переход|постприпев|часть|part|hook|refrain|interlude)\b",
        text,
    ):
        return True
    if re.match(r"(?i)^(?:часть|part)\s*[\dIVXLC]+$", text):
        return True
    if re.match(r"(?i)^реклама\s*\d*$", text):
        return True
    if len(text) <= 48 and re.search(
        rf"(?i)(?:{_KUPLET}|припев|verse|chorus|bridge|intro|outro|текст|рефрен|"
        r"хук|интерлюд|переход|часть)",
        text,
    ):
        return True
    if re.match(
        r"(?i)^(?:acoustic version|live version|remix|bonus track|deluxe edition)$",
        text,
    ):
        return True
    return False


def strip_genius_annotations(text):
    kept = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if _GENIUS_NOISE_LINE.match(line):
            continue
        low = line.lower()
        if "you might also like" in low or low in ("embed", "read more"):
            continue
        bracket = _BRACKET_ONLY.match(line)
        if bracket and _is_genius_section_label(bracket.group(1)):
            continue
        while True:
            tag = _LEADING_TAG.match(line)
            if not tag:
                break
            inner = tag.group(0).strip()
            if inner.startswith("["):
                label = inner[1 : inner.index("]")]
            else:
                label = inner[1 : inner.index(")")]
            if _is_genius_section_label(label):
                line = line[tag.end() :].strip()
                continue
            break
        if not line:
            continue
        if _GENIUS_NOISE_LINE.match(line):
            continue
        kept.append(line)
    out = "\n".join(kept).strip()
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out


def _normalize_lyrics_chars(text):
    for junk in ("\u2005", "\u200a", "\u205f", "\xa0", "\u200b"):
        text = text.replace(junk, " " if junk != "\u200b" else "")
    text = text.replace("’", "'").replace("`", "'").replace("…", "...")
    text = text.replace("“", '"').replace("”", '"').replace("„", '"').replace("«", '"').replace("»", '"')
    lines = [" ".join(raw.split()) for raw in text.splitlines() if raw.strip()]
    cleaned = "\n".join(lines).strip()
    counts = Counter(cleaned)
    rare = {
        char for char, count in counts.items()
        if count < 4 and not char.isalnum() and char not in " \n.,!?'-"
    }
    if rare:
        cleaned = "".join(" " if char in rare else char for char in cleaned)
        # Схлопывать пробелы только внутри строки: \n — единственный маркер конца строки куплета
        cleaned = "\n".join(
            " ".join(line.split())
            for line in cleaned.split("\n")
            if line.strip()
        )
    return cleaned.strip()


def clean_lyrics(text, newline_mode=NewlineMode.LM_STREAM):
    cleaned = _normalize_lyrics_chars(text)
    if not cleaned:
        return "\n"
    body = normalize_song_body(cleaned, mode=newline_mode)
    if not body:
        return "\n"
    return body.strip() + "\n"


def prepare_corpus_text(text, genius=False, newline_mode=NewlineMode.LM_STREAM):
    if genius:
        text = strip_genius_annotations(text)
    text = strip_corpus_noise(text)
    cleaned = _normalize_lyrics_chars(text)
    if not cleaned:
        return ""
    body = normalize_song_body(cleaned, mode=newline_mode)
    return (body.strip() + "\n") if body else ""


def load_corpus(path=None):
    if path is None:
        if DEFAULT_CORPUS.exists():
            path = DEFAULT_CORPUS
        elif FALLBACK_MESSAGES.exists():
            path = FALLBACK_MESSAGES
        else:
            raise FileNotFoundError("No corpus found. Run: python prepare_data.py")
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return _load_messages_csv(path)
    parts = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = _normalize_lyrics_chars(raw)
        if line:
            parts.append(line)
    if not parts:
        raise ValueError(f"Corpus is empty: {path}")
    return "\n".join(parts) + "\n"


def _load_messages_csv(path):
    parts = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = (row.get("text") or "").strip()
            if text:
                parts.append(text)
    if not parts:
        raise ValueError(f"No texts in {path}")
    return "\n\n".join(parts)


def text_to_seq(text_sample):
    char_counts = Counter(text_sample)
    ordered = sorted(char_counts.items(), key=lambda item: item[1], reverse=True)
    chars = [char for char, _ in ordered]
    char_to_idx = {char: index for index, char in enumerate(chars)}
    idx_to_char = {index: char for char, index in char_to_idx.items()}
    sequence = np.array([char_to_idx[char] for char in text_sample], dtype=np.int64)
    return sequence, char_to_idx, idx_to_char


def get_batch(sequence, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
    if len(sequence) <= seq_len:
        raise ValueError(f"Corpus is shorter than SEQ_LEN={seq_len}: {len(sequence)} chars")
    trains = []
    targets = []
    max_start = len(sequence) - seq_len
    for _ in range(batch_size):
        batch_start = int(np.random.randint(0, max_start))
        chunk = sequence[batch_start:batch_start + seq_len]
        trains.append(torch.LongTensor(chunk[:-1]).view(-1, 1))
        targets.append(torch.LongTensor(chunk[1:]).view(-1, 1))
    return torch.stack(trains, dim=0), torch.stack(targets, dim=0)
