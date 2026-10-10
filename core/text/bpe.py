"""BPE tokenizer (HuggingFace tokenizers) for train_lm level=bpe."""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer


def tokenizer_path_for_ckpt(ckpt: str | Path) -> Path:
    path = Path(ckpt)
    return path.with_name(f"{path.stem}.tokenizer.json")


def train_bpe(
    texts: list[str] | str,
    vocab_size: int,
    out_path: str | Path,
    *,
    min_frequency: int = 2,
) -> Tokenizer:
    if isinstance(texts, str):
        iterator = (line for line in texts.splitlines() if line.strip())
    else:
        iterator = (t for t in texts if t and str(t).strip())
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.post_processor = ByteLevelProcessor(trim_offsets=False)
    trainer = BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        show_progress=False,
    )
    tok.train_from_iterator(iterator, trainer=trainer)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


def load_bpe(path: str | Path) -> Tokenizer:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"BPE tokenizer not found: {path}")
    return Tokenizer.from_file(str(path))


def resolve_tokenizer_file(ckpt_or_tok: str | Path) -> Path:
    path = Path(ckpt_or_tok)
    if path.suffix == ".json" and path.is_file():
        return path
    companion = tokenizer_path_for_ckpt(path)
    if companion.is_file():
        return companion
    raise FileNotFoundError(
        f"BPE tokenizer missing next to checkpoint: expected {companion}"
    )


def encode_ids(tok: Tokenizer, text: str) -> list[int]:
    return list(tok.encode(str(text)).ids)


def decode_ids(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=False)


def vocab_maps(tok: Tokenizer) -> tuple[dict[str, int], dict[int, str]]:
    vocab = tok.get_vocab()
    token_to_idx = {str(k): int(v) for k, v in vocab.items()}
    idx_to_token = {int(v): str(k) for k, v in vocab.items()}
    return token_to_idx, idx_to_token


def text_to_bpe_seq(text: str, tok: Tokenizer):
    import numpy as np

    ids = encode_ids(tok, text)
    if len(ids) < 2:
        raise ValueError("BPE sequence too short")
    token_to_idx, idx_to_token = vocab_maps(tok)
    return np.asarray(ids, dtype=np.int64), token_to_idx, idx_to_token
