import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch

SEQ_LEN = 256
BATCH_SIZE = 16

DEFAULT_CORPUS = Path("data/kizaru_lyrics.txt")
LEGACY_MESSAGES = Path("legacy/messages.csv")


def clean_lyrics(text):
    for junk in ("\u2005", "\u200a", "\u205f", "\xa0", "\u200b"):
        text = text.replace(junk, " " if junk != "\u200b" else "")
    text = text.replace("’", "'").replace("`", "'").replace("…", "...")
    text = text.replace("“", '"').replace("”", '"').replace("„", '"').replace("«", '"').replace("»", '"')
    lines = []
    blank = False
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line:
            if lines and not blank:
                lines.append("")
            blank = True
            continue
        blank = False
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    counts = Counter(cleaned)
    rare = {
        char for char, count in counts.items()
        if count < 4 and not char.isalnum() and char not in " \n.,!?'-"
    }
    if rare:
        cleaned = "".join(" " if char in rare else char for char in cleaned)
        cleaned = "\n".join(" ".join(line.split()) for line in cleaned.splitlines())
    return cleaned.strip() + "\n"


def load_corpus(path=None):
    if path is None:
        if DEFAULT_CORPUS.exists():
            path = DEFAULT_CORPUS
        elif LEGACY_MESSAGES.exists():
            path = LEGACY_MESSAGES
        else:
            raise FileNotFoundError("No corpus found. Run: python prepare_data.py")
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return _load_messages_csv(path)
    text = clean_lyrics(path.read_text(encoding="utf-8"))
    if not text.strip():
        raise ValueError(f"Corpus is empty: {path}")
    return text


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
