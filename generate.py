import argparse

from kizaru.checkpoint import get_device, load_model
from kizaru.sample import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Continue a line with an exported TextRNN")
    parser.add_argument("model", help="Path to the .pt file from train.py or the notebook")
    parser.add_argument("--start", default="Я")
    parser.add_argument("--length", type=int, default=400)
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--lines", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    model, char_to_idx, idx_to_char, meta = load_model(args.model, device=device)
    print(
        f"loaded {args.model} vocab={len(char_to_idx)} "
        f"epoch={meta.get('epoch')} loss={meta.get('mean_loss')}"
    )
    for index in range(args.samples):
        print(evaluate(
            model,
            char_to_idx,
            idx_to_char,
            start_text=args.start,
            prediction_len=args.length,
            temp=args.temp,
            device=device,
            max_lines=args.lines or None,
        ))
        if index + 1 < args.samples:
            print("-" * 40)


if __name__ == "__main__":
    main()
