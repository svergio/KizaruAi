import argparse
import json
from pathlib import Path

from core.text.pretrained_vocab import (
    DEFAULT_TOKENIZER,
    DEFAULT_WEIGHTS,
    build_seed_whitelist,
    filter_known,
    load_whitelist_file,
    load_wordlevel_vocab,
    seed_tokens,
)
from core.text.tokenize import TOKEN_UNK, encode_words, tokenize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Navec whitelist for word-LSTM vocab / embedding init"
    )
    parser.add_argument(
        "--navec-tokenizer",
        default=str(DEFAULT_TOKENIZER),
        help="HF WordLevel tokenizer.json from BorisTM navec hudlit",
    )
    parser.add_argument(
        "--seeds",
        default="experiments/shared/seeds.json",
        help="JSON list of seed strings (Discord-style prompts)",
    )
    parser.add_argument(
        "--extra-words",
        default=None,
        help="Optional txt: one word per line to force into whitelist",
    )
    parser.add_argument(
        "--out",
        default="data/embeddings/navec_hudlit/seed_whitelist.txt",
        help="Write known-in-Navec whitelist here",
    )
    parser.add_argument(
        "--report-missing",
        default="data/embeddings/navec_hudlit/seed_missing.txt",
        help="Words from seeds not present in Navec",
    )
    parser.add_argument(
        "--check-ckpt",
        default=None,
        help="Optional checkpoints/*.pt: report which seed tokens map to <unk>",
    )
    return parser.parse_args()


def _load_seeds(path: str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("seeds JSON must be a list of strings")
    return [str(item) for item in payload]


def _write_lines(path: str, lines) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _ckpt_unk_report(ckpt_path: str, seed_texts) -> None:
    import torch

    payload = torch.load(ckpt_path, map_location="cpu")
    token_to_idx = payload.get("char_to_idx") or payload.get("vocab")
    if not isinstance(token_to_idx, dict):
        raise ValueError(f"no vocab in {ckpt_path}")
    unk = token_to_idx.get(TOKEN_UNK)
    print(f"ckpt={ckpt_path} vocab={len(token_to_idx)} unk_id={unk}")
    for text in seed_texts:
        tokens = tokenize(text)
        ids = encode_words(text, token_to_idx)
        oov = [tok for tok, idx in zip(tokens, ids) if idx == unk]
        print(f"  seed={text!r} oov={oov or '-'}")


def main():
    args = parse_args()
    seeds = _load_seeds(args.seeds)
    known, missing = build_seed_whitelist(seeds, args.navec_tokenizer)
    if args.extra_words:
        extra = load_whitelist_file(args.extra_words)
        pretrained = load_wordlevel_vocab(args.navec_tokenizer)
        known_extra = filter_known(extra, pretrained)
        missing_extra = [word for word in extra if word not in pretrained]
        known = list(dict.fromkeys([*known, *known_extra]))
        missing = list(dict.fromkeys([*missing, *missing_extra]))

    _write_lines(args.out, known)
    _write_lines(args.report_missing, missing)
    print(
        f"navec={args.navec_tokenizer} seeds={len(seeds)} "
        f"seed_tokens={len(seed_tokens(seeds))} "
        f"whitelist={len(known)} missing={len(missing)}"
    )
    print(f"wrote {args.out}")
    print(f"wrote {args.report_missing}")
    if missing:
        print("missing:", ", ".join(missing))
    weights = Path(DEFAULT_WEIGHTS)
    print(f"weights={'ok' if weights.is_file() else 'missing'} path={weights}")
    if args.check_ckpt:
        _ckpt_unk_report(args.check_ckpt, seeds)


if __name__ == "__main__":
    main()
