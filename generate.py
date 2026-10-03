import argparse

from train import get_device, load_checkpoint
from kizaru_rnn.generate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Generate next lines from a trained TextRNN")
    parser.add_argument("--checkpoint", default="checkpoints/kizaru.pt")
    parser.add_argument("--start", default="Я")
    parser.add_argument("--length", type=int, default=400)
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--samples", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    model, char_to_idx, idx_to_char, payload = load_checkpoint(args.checkpoint, device=device)
    print(
        f"loaded {args.checkpoint} vocab={len(char_to_idx)} "
        f"epoch={payload.get('epoch')} loss={payload.get('mean_loss')}"
    )
    for i in range(args.samples):
        text = evaluate(
            model,
            char_to_idx,
            idx_to_char,
            start_text=args.start,
            prediction_len=args.length,
            temp=args.temp,
            device=device,
        )
        print(text)
        if i + 1 < args.samples:
            print("-" * 40)


if __name__ == "__main__":
    main()
