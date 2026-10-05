import re
from collections import Counter

import numpy as np

TOKEN_UNK = "<unk>"
TOKEN_NL = "<nl>"
SPECIAL = (TOKEN_UNK, TOKEN_NL)

# слово / число / одиночная пунктуация; перевод строки — отдельный токен
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

WORD_SEQ_LEN = 64
WORD_MIN_FREQ = 3


def tokenize(text: str) -> list[str]:
    tokens = []
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    for index, line in enumerate(normalized.split("\n")):
        if index:
            tokens.append(TOKEN_NL)
        for match in _TOKEN_RE.finditer(line):
            tokens.append(match.group(0).lower())
    return tokens


def detokenize(tokens) -> str:
    parts = []
    for token in tokens:
        if token == TOKEN_NL:
            parts.append("\n")
            continue
        if not parts or parts[-1].endswith("\n"):
            parts.append(token)
            continue
        if len(token) == 1 and not token.isalnum() and token not in {"ё", "Ё"}:
            parts.append(token)
        else:
            parts.append(" ")
            parts.append(token)
    return "".join(parts)


def build_word_vocab(text: str, min_freq: int = WORD_MIN_FREQ, whitelist=None):
    counts = Counter(tokenize(text))
    allow = {str(word).lower() for word in (whitelist or []) if str(word).strip()}
    words = [TOKEN_UNK, TOKEN_NL]
    seen = set(SPECIAL)
    for word, freq in counts.most_common():
        if word in SPECIAL:
            continue
        if freq < min_freq and word not in allow:
            continue
        words.append(word)
        seen.add(word)
    for word in sorted(allow):
        if word in seen or word in SPECIAL:
            continue
        words.append(word)
        seen.add(word)
    token_to_idx = {token: index for index, token in enumerate(words)}
    idx_to_token = {index: token for token, index in token_to_idx.items()}
    return token_to_idx, idx_to_token


def text_to_word_seq(
    text: str, token_to_idx=None, min_freq: int = WORD_MIN_FREQ, whitelist=None
):
    if token_to_idx is None:
        token_to_idx, idx_to_token = build_word_vocab(
            text, min_freq=min_freq, whitelist=whitelist
        )
    else:
        idx_to_token = {index: token for token, index in token_to_idx.items()}
    unk = token_to_idx[TOKEN_UNK]
    ids = [token_to_idx.get(token, unk) for token in tokenize(text)]
    if not ids:
        raise ValueError("word sequence is empty")
    sequence = np.array(ids, dtype=np.int64)
    return sequence, token_to_idx, idx_to_token


def encode_words(text: str, token_to_idx: dict) -> list[int]:
    unk = token_to_idx[TOKEN_UNK]
    ids = [token_to_idx.get(token, unk) for token in tokenize(text)]
    if ids:
        return ids
    if TOKEN_NL in token_to_idx:
        return [token_to_idx[TOKEN_NL]]
    return [unk]
