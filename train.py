import argparse

import numpy as np
import torch
from torch import nn

from kizaru.checkpoint import get_device, save_model
from kizaru.data import BATCH_SIZE, SEQ_LEN, get_batch, load_corpus, text_to_seq
from kizaru.model import TextRNN
from kizaru.sample import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Train TextRNN and export a .pt file")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--out", default="checkpoints/kizaru.pt")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--gen-len", type=int, default=400)
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--start", default="Я")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    text_sample = load_corpus(args.corpus)
    sequence, char_to_idx, idx_to_char = text_to_seq(text_sample)
    print(f"device={device} chars={len(text_sample)} vocab={len(char_to_idx)}")

    model = TextRNN(
        input_size=len(idx_to_char),
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        n_layers=args.n_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    loss_avg = []
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train, target = get_batch(sequence, seq_len=args.seq_len, batch_size=args.batch_size)
        train = train.permute(1, 0, 2).to(device)
        target = target.permute(1, 0, 2).to(device)
        hidden = model.init_hidden(args.batch_size, device=device)

        output, hidden = model(train, hidden)
        loss = criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_avg.append(loss.item())
        if len(loss_avg) >= args.eval_every:
            mean_loss = float(np.mean(loss_avg))
            print(f"Loss: {mean_loss:.6f}")
            scheduler.step(mean_loss)
            loss_avg = []
            print(evaluate(
                model,
                char_to_idx,
                idx_to_char,
                start_text=args.start,
                prediction_len=args.gen_len,
                temp=args.temp,
                device=device,
                max_lines=6,
            ))
            print("-" * 40)
            if mean_loss < best_loss:
                best_loss = mean_loss
                save_model(
                    args.out,
                    model,
                    char_to_idx,
                    idx_to_char,
                    epoch=epoch + 1,
                    mean_loss=mean_loss,
                )
                print(f"saved {args.out}")

    save_model(
        args.out,
        model,
        char_to_idx,
        idx_to_char,
        epoch=args.epochs,
        mean_loss=best_loss,
    )
    print(f"done, model={args.out} best_loss={best_loss:.6f}")


if __name__ == "__main__":
    main()
