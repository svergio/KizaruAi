import argparse
import json
import re
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.text.checkpoint import (
    ARCH_DEEPSEEK,
    ARCH_GPT,
    ARCH_RNN,
    ARCH_TRANSFORMER,
    get_device,
    load_model,
)
from core.text.sample import evaluate, evaluate_lm, evaluate_lm_words, evaluate_words
from core.text.tokenize import TOKEN_UNK, encode_words

CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
LETTER_RE = re.compile(r"[A-Za-zа-яА-ЯёЁ]")
BOT_MAX_LINES = 6
BOT_CHAR_LEN = 360
BOT_WORD_LEN = 72


def _is_word(meta, vocab):
    level = meta.get("level")
    if level in ("word", "char"):
        return level == "word"
    return TOKEN_UNK in vocab


def _bot_temp(arch, word):
    if word:
        return 0.8
    if arch == ARCH_RNN:
        return 0.4
    return 0.8


def _encode_char(text, char_to_idx):
    return [char_to_idx[ch] for ch in text if ch in char_to_idx]


def _encode(text, vocab, word):
    if word:
        return encode_words(text, vocab)
    return _encode_char(text, vocab)


def _rnn_ce(model, ids, device, chunk=512):
    hidden = model.init_hidden(1, device=device)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    ntok = 0
    seq = torch.tensor(ids, dtype=torch.long, device=device)
    with torch.no_grad():
        for start in range(0, len(ids) - 1, chunk):
            end = min(start + chunk, len(ids) - 1)
            inp = seq[start:end].view(-1, 1, 1)
            tgt = seq[start + 1 : end + 1]
            logits, hidden = model(inp, hidden)
            hidden = tuple(part.detach() for part in hidden)
            total += float(criterion(logits.squeeze(1), tgt).item())
            ntok += int(tgt.numel())
    return total / max(ntok, 1), ntok


def _lm_ce(model, ids, device):
    block = int(model.block_size)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    ntok = 0
    seq = torch.tensor(ids, dtype=torch.long, device=device)
    with torch.no_grad():
        start = 0
        while start < len(ids) - 1:
            end = min(start + block, len(ids) - 1)
            inp = seq[start:end].view(1, -1)
            tgt = seq[start + 1 : end + 1]
            logits = model(inp)[0]
            steps = min(logits.size(0), tgt.size(0))
            total += float(criterion(logits[:steps], tgt[:steps]).item())
            ntok += int(steps)
            start += block
    return total / max(ntok, 1), ntok


def _repeat_ratio(text):
    lines = [ln.strip().lower() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return 0.0
    dup = 0
    seen = set()
    for line in lines:
        if line in seen:
            dup += 1
        seen.add(line)
    return dup / len(lines)


def _cyrillic_ratio(text):
    letters = LETTER_RE.findall(text)
    if not letters:
        return 0.0
    cyr = sum(1 for ch in letters if CYRILLIC_RE.match(ch))
    return cyr / len(letters)


def _unk_ratio(text, word):
    if not word or not text:
        return 0.0
    return text.count(TOKEN_UNK) / max(len(text.split()), 1)


def _generate(model, vocab, idx_map, arch, word, seed, temp, device):
    kwargs = dict(
        start_text=seed,
        temp=temp,
        device=device,
        max_lines=BOT_MAX_LINES,
    )
    if word:
        kwargs["prediction_len"] = BOT_WORD_LEN
        if arch == ARCH_RNN:
            return evaluate_words(model, vocab, idx_map, **kwargs)
        return evaluate_lm_words(model, vocab, idx_map, **kwargs)
    kwargs["prediction_len"] = BOT_CHAR_LEN
    if arch == ARCH_RNN:
        return evaluate(model, vocab, idx_map, **kwargs)
    return evaluate_lm(model, vocab, idx_map, **kwargs)


def bench(ckpt, test_path, seeds_path, out_path, device=None):
    if device is None:
        device = get_device()
    model, vocab, idx_map, meta = load_model(ckpt, device=device)
    arch = meta.get("arch") or ARCH_RNN
    word = _is_word(meta, vocab)
    temp = _bot_temp(arch, word)
    test_text = Path(test_path).read_text(encoding="utf-8")
    ids = _encode(test_text, vocab, word)
    if len(ids) < 2:
        raise ValueError("test sequence too short after vocab filter")
    if arch in {ARCH_TRANSFORMER, ARCH_GPT, ARCH_DEEPSEEK}:
        ce, ntok = _lm_ce(model, ids, device)
    else:
        ce, ntok = _rnn_ce(model, ids, device)

    seeds = json.loads(Path(seeds_path).read_text(encoding="utf-8"))
    samples = []
    repeats = []
    cyr = []
    unks = []
    for seed in seeds:
        text = _generate(model, vocab, idx_map, arch, word, seed, temp, device)
        samples.append({"seed": seed, "text": text})
        repeats.append(_repeat_ratio(text))
        cyr.append(_cyrillic_ratio(text))
        unks.append(_unk_ratio(text, word))

    score = {
        "ckpt": str(Path(ckpt).resolve()),
        "arch": arch,
        "level": "word" if word else "char",
        "temp": temp,
        "max_lines": BOT_MAX_LINES,
        "ce": ce,
        "n_tokens": ntok,
        "repeat_ratio": sum(repeats) / max(len(repeats), 1),
        "cyrillic_ratio": sum(cyr) / max(len(cyr), 1),
        "unk_ratio": sum(unks) / max(len(unks), 1),
        "samples": samples,
        "per_seed": [
            {
                "seed": item["seed"],
                "repeat_ratio": repeats[i],
                "cyrillic_ratio": cyr[i],
                "unk_ratio": unks[i],
            }
            for i, item in enumerate(samples)
        ],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return score


def parse_args():
    parser = argparse.ArgumentParser(description="Test CE + fixed-seed generation for a .pt")
    parser.add_argument("ckpt")
    parser.add_argument("--test", default=str(ROOT / "experiments" / "shared" / "test.txt"))
    parser.add_argument("--seeds", default=str(ROOT / "experiments" / "shared" / "seeds.json"))
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.ckpt)
    out = Path(args.out) if args.out else ckpt.with_suffix(".score.json")
    score = bench(ckpt, args.test, args.seeds, out)
    print(
        f"wrote {out} arch={score['arch']} level={score['level']} "
        f"ce={score['ce']:.6f} n_tokens={score['n_tokens']} "
        f"repeat={score['repeat_ratio']:.3f} cyr={score['cyrillic_ratio']:.3f} "
        f"unk={score['unk_ratio']:.3f}"
    )


if __name__ == "__main__":
    main()
