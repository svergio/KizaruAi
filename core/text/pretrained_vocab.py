import json
from pathlib import Path

import torch

from core.text.tokenize import SPECIAL, TOKEN_NL, TOKEN_UNK, tokenize

DEFAULT_NAVEC_DIR = Path("data/embeddings/navec_hudlit")
DEFAULT_TOKENIZER = DEFAULT_NAVEC_DIR / "tokenizer.json"
DEFAULT_WEIGHTS = DEFAULT_NAVEC_DIR / "model.safetensors"
NAVEC_DIM = 300


def load_wordlevel_vocab(tokenizer_path=None) -> dict[str, int]:
    path = Path(tokenizer_path or DEFAULT_TOKENIZER)
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model") or {}
    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"no WordLevel vocab in {path}")
    return {str(token): int(index) for token, index in vocab.items()}


def load_whitelist_file(path: str | Path) -> list[str]:
    words = []
    seen = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if not word or word.startswith("#") or word in SPECIAL or word in seen:
            continue
        seen.add(word)
        words.append(word)
    return words


def seed_tokens(seed_texts) -> list[str]:
    seen = set()
    out = []
    for text in seed_texts:
        for token in tokenize(str(text)):
            if token in SPECIAL or token in seen:
                continue
            seen.add(token)
            out.append(token)
    return out


def filter_known(words, pretrained_vocab: dict[str, int]) -> list[str]:
    return [word for word in words if word in pretrained_vocab]


def merge_whitelist(token_to_idx: dict, whitelist) -> tuple[dict, dict]:
    merged = dict(token_to_idx)
    for word in whitelist:
        word = str(word).lower()
        if not word or word in SPECIAL or word in merged:
            continue
        merged[word] = len(merged)
    idx_to_token = {index: token for token, index in merged.items()}
    return merged, idx_to_token


def build_seed_whitelist(
    seed_texts,
    tokenizer_path=None,
    *,
    require_pretrained: bool = True,
) -> tuple[list[str], list[str]]:
    candidates = seed_tokens(seed_texts)
    if not require_pretrained:
        return candidates, []
    pretrained = load_wordlevel_vocab(tokenizer_path)
    known = filter_known(candidates, pretrained)
    missing = [word for word in candidates if word not in pretrained]
    return known, missing


def load_navec_matrix(weights_path=None) -> torch.Tensor:
    path = Path(weights_path or DEFAULT_WEIGHTS)
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required to load Navec weights; pip install safetensors"
        ) from exc
    tensors = load_file(str(path))
    if "embedding.weight" not in tensors:
        raise KeyError(f"embedding.weight missing in {path}")
    weight = tensors["embedding.weight"]
    if weight.ndim != 2:
        raise ValueError(f"expected 2D embedding matrix, got {tuple(weight.shape)}")
    return weight


def init_embedding_from_navec(
    embedding: torch.nn.Embedding,
    token_to_idx: dict,
    *,
    tokenizer_path=None,
    weights_path=None,
    pretrained_vocab: dict[str, int] | None = None,
) -> dict:
    if pretrained_vocab is None:
        pretrained_vocab = load_wordlevel_vocab(tokenizer_path)
    matrix = load_navec_matrix(weights_path)
    source_dim = matrix.shape[1]
    target_dim = embedding.embedding_dim
    filled = 0
    skipped = 0
    with torch.no_grad():
        for token, index in token_to_idx.items():
            if token in SPECIAL:
                continue
            source_index = pretrained_vocab.get(token)
            if source_index is None:
                skipped += 1
                continue
            if not (0 <= source_index < matrix.shape[0]):
                skipped += 1
                continue
            vector = matrix[source_index]
            if target_dim == source_dim:
                embedding.weight[index].copy_(vector)
            elif target_dim < source_dim:
                embedding.weight[index].copy_(vector[:target_dim])
            else:
                embedding.weight[index][:source_dim].copy_(vector)
            filled += 1
    return {
        "filled": filled,
        "skipped": skipped,
        "source_dim": source_dim,
        "target_dim": target_dim,
        "vocab": len(token_to_idx),
    }
