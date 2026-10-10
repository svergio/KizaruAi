import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from core.text.checkpoint import get_device, load_model, save_model
from core.corpus.lyrics import BATCH_SIZE, SEQ_LEN, get_batch, load_corpus, text_to_seq
from core.text.pretrained_vocab import (
    DEFAULT_TOKENIZER,
    DEFAULT_WEIGHTS,
    init_embedding_from_navec,
    load_whitelist_file,
)
from core.text.rnn import TextRNN
from core.text.sample import evaluate, evaluate_words
from core.text.tokenize import WORD_MIN_FREQ, WORD_SEQ_LEN, text_to_word_seq
from training import mlflow_track


def parse_args():
    parser = argparse.ArgumentParser(description="Train TextRNN and export a .pt file")
    parser.add_argument("--level", choices=("char", "word"), default="char")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=12000)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--bptt", type=int, default=None)
    parser.add_argument("--min-freq", type=int, default=WORD_MIN_FREQ)
    parser.add_argument(
        "--whitelist",
        default=None,
        help="txt word list (e.g. data/embeddings/navec_hudlit/seed_whitelist.txt)",
    )
    parser.add_argument(
        "--init-navec",
        action="store_true",
        help="init word embedding rows from local Navec hudlit safetensors",
    )
    parser.add_argument("--navec-tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--navec-weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument(
        "--resume",
        default=None,
        help="load .pt weights+vocab and continue from --start-epoch or ckpt epoch",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="epoch index to continue from when --resume is set (default: ckpt meta epoch)",
    )
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--gen-len", type=int, default=None)
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--start", default="Я")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_bptt(model, criterion, optimizer, train, target, bptt):
    time_steps = train.shape[0]
    batch_size = train.shape[1]
    device = train.device
    hidden = model.init_hidden(batch_size, device=device)
    total_loss = 0.0
    chunks = 0
    for start in range(0, time_steps, bptt):
        end = min(start + bptt, time_steps)
        chunk_x = train[start:end]
        chunk_y = target[start:end]
        if isinstance(hidden, tuple):
            hidden = tuple(part.detach() for part in hidden)
        else:
            hidden = hidden.detach()
        output, hidden = model(chunk_x, hidden)
        loss = criterion(output.permute(1, 2, 0), chunk_y.squeeze(-1).permute(1, 0))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
        chunks += 1
    return total_loss / max(chunks, 1)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    level = args.level
    out = args.out or (
        "checkpoints/kizaru_word.pt" if level == "word" else "checkpoints/kizaru.pt"
    )
    seq_len = args.seq_len or (WORD_SEQ_LEN if level == "word" else SEQ_LEN)
    bptt = args.bptt or (32 if level == "word" else 64)
    gen_len = args.gen_len or (48 if level == "word" else 400)
    temp = args.temp if args.temp is not None else (0.8 if level == "word" else 0.3)

    device = get_device()
    text_sample = load_corpus(args.corpus)
    whitelist = None
    if level == "word" and args.whitelist:
        whitelist = load_whitelist_file(args.whitelist)

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file() or resume_path.stat().st_size < 1024:
            raise FileNotFoundError(f"resume checkpoint missing/empty: {resume_path}")
        model, token_to_idx, idx_to_token, meta = load_model(str(resume_path), device=device)
        ckpt_level = meta.get("level") or level
        if ckpt_level != level:
            raise ValueError(f"resume level={ckpt_level!r} != --level {level!r}")
        if level == "word":
            sequence, token_to_idx, idx_to_token = text_to_word_seq(
                text_sample, token_to_idx=token_to_idx
            )
        else:
            sequence, _, _ = text_to_seq(text_sample)
            # keep checkpoint vocab for char resume
        start_epoch = (
            args.start_epoch
            if args.start_epoch is not None
            else int(meta.get("epoch") or 0)
        )
        start_epoch = max(0, min(start_epoch, args.epochs))
        if meta.get("mean_loss") is not None:
            best_loss = float(meta["mean_loss"])
        print(
            f"resume={resume_path} start_epoch={start_epoch}/{args.epochs} "
            f"best_loss={best_loss:.6f} vocab={len(token_to_idx)}"
        )
    else:
        if level == "word":
            sequence, token_to_idx, idx_to_token = text_to_word_seq(
                text_sample, min_freq=args.min_freq, whitelist=whitelist
            )
        else:
            sequence, token_to_idx, idx_to_token = text_to_seq(text_sample)

        model = TextRNN(
            input_size=len(idx_to_token),
            hidden_size=args.hidden_size,
            embedding_size=args.embedding_size,
            n_layers=args.n_layers,
        ).to(device)
        if level == "word" and args.init_navec:
            if not Path(args.navec_weights).is_file():
                raise FileNotFoundError(args.navec_weights)
            stats = init_embedding_from_navec(
                model.encoder,
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
        f"level={level} device={device} text_chars={len(text_sample)} "
        f"tokens={len(sequence)} vocab={len(token_to_idx)} "
        f"whitelist={0 if whitelist is None else len(whitelist)} "
        f"lstm={args.embedding_size}/{args.hidden_size}x{args.n_layers} "
        f"seq={seq_len} bptt={bptt} params={n_params}"
    )
    run_id = mlflow_track.start_run(
        "kizaru-legacy",
        run_name=f"textrnn_{level}_{Path(out).stem}",
        params={
            "level": level,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "embedding_size": args.embedding_size,
            "n_layers": args.n_layers,
            "lr": args.lr,
            "start_epoch": start_epoch,
            "out": out,
            "n_params": n_params,
        },
        tags={"project": "kizaruai", "entrypoint": "train"},
    )
    if run_id:
        print(f"mlflow_run={run_id}", flush=True)

    if start_epoch >= args.epochs:
        save_model(
            out,
            model,
            token_to_idx,
            idx_to_token,
            epoch=args.epochs,
            mean_loss=best_loss if best_loss < float("inf") else None,
            level=level,
        )
        print(f"done (already at epochs), model={out} best_loss={best_loss:.6f}")
        mlflow_track.end_run()
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    loss_avg = []
    bptt = max(1, int(bptt))
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train, target = get_batch(sequence, seq_len=seq_len, batch_size=args.batch_size)
        train = train.permute(1, 0, 2).to(device)
        target = target.permute(1, 0, 2).to(device)
        mean_chunk_loss = train_bptt(model, criterion, optimizer, train, target, bptt)

        loss_avg.append(mean_chunk_loss)
        if len(loss_avg) >= args.eval_every:
            mean_loss = float(np.mean(loss_avg))
            print(f"Loss: {mean_loss:.6f}")
            mlflow_track.log_metrics({"loss": mean_loss}, step=epoch + 1)
            scheduler.step(mean_loss)
            loss_avg = []
            if level == "word":
                sample = evaluate_words(
                    model,
                    token_to_idx,
                    idx_to_token,
                    start_text=args.start,
                    prediction_len=gen_len,
                    temp=temp,
                    device=device,
                    max_lines=6,
                )
            else:
                sample = evaluate(
                    model,
                    token_to_idx,
                    idx_to_token,
                    start_text=args.start,
                    prediction_len=gen_len,
                    temp=temp,
                    device=device,
                    max_lines=6,
                )
            print(sample)
            print("-" * 40)
            if mean_loss < best_loss:
                best_loss = mean_loss
                save_model(
                    out,
                    model,
                    token_to_idx,
                    idx_to_token,
                    epoch=epoch + 1,
                    mean_loss=mean_loss,
                    level=level,
                )
                print(f"saved {out}")

    save_model(
        out,
        model,
        token_to_idx,
        idx_to_token,
        epoch=args.epochs,
        mean_loss=best_loss if best_loss < float("inf") else None,
        level=level,
    )
    print(f"done, model={out} best_loss={best_loss:.6f}")
    mlflow_track.log_metrics({"best_loss": best_loss if best_loss < float("inf") else 0.0})
    mlflow_track.end_run()


if __name__ == "__main__":
    main()
