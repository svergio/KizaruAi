import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from core.text.checkpoint import (
    ARCH_BERT_GEN,
    ARCH_DEEPSEEK,
    ARCH_GPT,
    ARCH_TRANSFORMER,
    get_device,
    save_model,
)
from core.corpus.lyrics import BATCH_SIZE, get_batch, load_corpus, text_to_seq
from core.text.bert_gen import BertMLM, apply_mlm_mask, generate_mlm
from core.text.deepseek import MiniDeepSeek
from core.text.lm import MiniGPT, SimpleTransformer
from core.text.pretrained_vocab import (
    DEFAULT_TOKENIZER,
    DEFAULT_WEIGHTS,
    NAVEC_DIM,
    init_embedding_from_navec,
    load_whitelist_file,
)
from core.text.sample import evaluate_lm, evaluate_lm_words
from core.text.tokenize import WORD_MIN_FREQ, text_to_word_seq

ARCHES = {
    ARCH_TRANSFORMER: SimpleTransformer,
    ARCH_GPT: MiniGPT,
    ARCH_DEEPSEEK: MiniDeepSeek,
    ARCH_BERT_GEN: BertMLM,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train transformer, mini-GPT, MiniDeepSeek, or BERT-gen MLM"
    )
    parser.add_argument("--arch", choices=sorted(ARCHES), required=True)
    parser.add_argument("--level", choices=("char", "word"), default="char")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=None,
        help="tok emb dim (default=d_model); use 300 with --init-navec",
    )
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--kv-lora-rank",
        type=int,
        default=None,
        help="MLA KV latent rank (deepseek only; default d_model//4)",
    )
    parser.add_argument(
        "--q-lora-rank",
        type=int,
        default=None,
        help="MLA Q latent rank (deepseek only; default d_model//4)",
    )
    parser.add_argument("--min-freq", type=int, default=WORD_MIN_FREQ)
    parser.add_argument(
        "--whitelist",
        default=None,
        help="txt word list (e.g. data/embeddings/navec_hudlit/seed_whitelist.txt)",
    )
    parser.add_argument(
        "--init-navec",
        action="store_true",
        help="init word tok embedding rows from local Navec hudlit safetensors",
    )
    parser.add_argument("--navec-tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--navec-weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--gen-len", type=int, default=None)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--start", default="Я")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    level = args.level
    out = args.out or (
        f"checkpoints/{args.arch}_word.pt" if level == "word" else f"checkpoints/{args.arch}.pt"
    )
    block_size = args.block_size or (64 if level == "word" else 256)
    gen_len = args.gen_len or (48 if level == "word" else 200)
    embedding_size = args.embedding_size
    if embedding_size is None and args.init_navec:
        embedding_size = NAVEC_DIM
    if embedding_size is None:
        embedding_size = args.d_model

    if args.init_navec and level != "word":
        raise SystemExit("--init-navec requires --level word (char models have no word rows)")
    if args.whitelist and level != "word":
        raise SystemExit("--whitelist requires --level word")

    device = get_device()
    text_sample = load_corpus(args.corpus)
    whitelist = None
    if level == "word" and args.whitelist:
        whitelist = load_whitelist_file(args.whitelist)
    if level == "word":
        sequence, token_to_idx, idx_to_token = text_to_word_seq(
            text_sample, min_freq=args.min_freq, whitelist=whitelist
        )
    else:
        sequence, token_to_idx, idx_to_token = text_to_seq(text_sample)

    model_kwargs = dict(
        vocab_size=len(idx_to_token),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        block_size=block_size,
        embedding_size=embedding_size,
    )
    if args.arch == ARCH_DEEPSEEK:
        model_kwargs["kv_lora_rank"] = args.kv_lora_rank
        model_kwargs["q_lora_rank"] = args.q_lora_rank
    if args.arch == ARCH_BERT_GEN:
        model_kwargs["vocab_size"] = len(idx_to_token) + 1
    model = ARCHES[args.arch](**model_kwargs).to(device)
    if level == "word" and args.init_navec:
        if not Path(args.navec_weights).is_file():
            raise FileNotFoundError(args.navec_weights)
        if embedding_size != NAVEC_DIM:
            print(
                f"navec_note: embedding_size={embedding_size} != navec={NAVEC_DIM}; "
                f"init will truncate/pad into model.tok"
            )
        if embedding_size != args.d_model:
            print(
                f"navec_note: tok emb {embedding_size}d -> Linear -> d_model={args.d_model}"
            )
        stats = init_embedding_from_navec(
            model.tok,
            token_to_idx,
            tokenizer_path=args.navec_tokenizer,
            weights_path=args.navec_weights,
        )
        print(
            f"navec_init filled={stats['filled']} skipped={stats['skipped']} "
            f"dim={stats['source_dim']}->{stats['target_dim']}"
        )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"arch={args.arch} level={level} device={device} "
        f"text_chars={len(text_sample)} tokens={len(sequence)} vocab={len(token_to_idx)} "
        f"whitelist={0 if whitelist is None else len(whitelist)} "
        f"d_model={args.d_model} emb={embedding_size} params={n_params}"
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    seq_len = block_size + 1

    loss_avg = []
    best_loss = float("inf")
    for step in range(args.steps):
        model.train()
        if args.arch == ARCH_BERT_GEN:
            train, _target = get_batch(
                sequence, seq_len=block_size, batch_size=args.batch_size
            )
            tokens = train.squeeze(-1).to(device)[:, :block_size]
            inputs, labels = apply_mlm_mask(tokens, model.mask_id, model.vocab_size)
            logits = model(inputs)
        else:
            train, target = get_batch(sequence, seq_len=seq_len, batch_size=args.batch_size)
            tokens = train.squeeze(-1).to(device)
            labels = target.squeeze(-1).to(device)
            logits = model(tokens)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_avg.append(loss.item())
        if len(loss_avg) >= args.eval_every:
            mean_loss = float(np.mean(loss_avg))
            print(f"step={step + 1} loss={mean_loss:.6f}")
            if args.arch == ARCH_BERT_GEN:
                sample = generate_mlm(
                    model,
                    token_to_idx,
                    idx_to_token,
                    start_text=args.start,
                    seq_len=min(128, block_size),
                    n_iters=16,
                    temp=args.temp,
                    device=device,
                    max_lines=6,
                    level=level,
                )
            elif level == "word":
                sample = evaluate_lm_words(
                    model,
                    token_to_idx,
                    idx_to_token,
                    start_text=args.start,
                    prediction_len=gen_len,
                    temp=args.temp,
                    device=device,
                    max_lines=6,
                )
            else:
                sample = evaluate_lm(
                    model,
                    token_to_idx,
                    idx_to_token,
                    start_text=args.start,
                    prediction_len=gen_len,
                    temp=args.temp,
                    device=device,
                    max_lines=6,
                )
            print(sample)
            print("-" * 40)
            loss_avg = []
            if mean_loss < best_loss:
                best_loss = mean_loss
                save_model(
                    out,
                    model,
                    token_to_idx,
                    idx_to_token,
                    epoch=step + 1,
                    mean_loss=mean_loss,
                    level=level,
                )
                print(f"saved {out}")

    save_model(
        out,
        model,
        token_to_idx,
        idx_to_token,
        epoch=args.steps,
        mean_loss=best_loss,
        level=level,
    )
    print(f"done, model={out} best_loss={best_loss:.6f}")


if __name__ == "__main__":
    main()
