import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from kizaru_rnn.data import BATCH_SIZE, SEQ_LEN, get_batch, load_corpus, text_to_seq
from kizaru_rnn.generate import evaluate
from kizaru_rnn.model import TextRNN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(path, model, char_to_idx, idx_to_char, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "char_to_idx": char_to_idx,
        "idx_to_char": {int(k): v for k, v in idx_to_char.items()},
        "hidden_size": model.hidden_size,
        "embedding_size": model.embedding_size,
        "n_layers": model.n_layers,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device=None):
    if device is None:
        device = get_device()
    payload = torch.load(path, map_location=device, weights_only=False)
    idx_to_char = {int(k): v for k, v in payload["idx_to_char"].items()}
    char_to_idx = payload["char_to_idx"]
    model = TextRNN(
        input_size=len(idx_to_char),
        hidden_size=payload["hidden_size"],
        embedding_size=payload["embedding_size"],
        n_layers=payload["n_layers"],
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return model, char_to_idx, idx_to_char, payload


def parse_args():
    parser = argparse.ArgumentParser(description="Train char-level LSTM on Kizaru lyrics")
    parser.add_argument("--corpus", default=None, help="Path to lyrics .txt or messages .csv")
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
    parser.add_argument("--checkpoint", default="checkpoints/kizaru.pt")
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.5,
    )

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
            predicted_text = evaluate(
                model,
                char_to_idx,
                idx_to_char,
                start_text=args.start,
                prediction_len=args.gen_len,
                temp=args.temp,
                device=device,
            )
            print(predicted_text)
            print("-" * 40)
            extra = {"epoch": epoch + 1, "mean_loss": mean_loss}
            if mean_loss < best_loss:
                best_loss = mean_loss
                save_checkpoint(args.checkpoint, model, char_to_idx, idx_to_char, extra)
                print(f"saved {args.checkpoint}")

    save_checkpoint(
        args.checkpoint,
        model,
        char_to_idx,
        idx_to_char,
        {"epoch": args.epochs, "mean_loss": best_loss},
    )
    print(f"done, checkpoint={args.checkpoint} best_loss={best_loss:.6f}")


if __name__ == "__main__":
    main()
