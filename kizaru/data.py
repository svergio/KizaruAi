import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch

SEQ_LEN = 256
BATCH_SIZE = 16

DEFAULT_CORPUS = Path("data/kizaru_lyrics.txt")
LEGACY_MESSAGES = Path("legacy/messages.csv")


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
    text = path.read_text(encoding="utf-8")
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
