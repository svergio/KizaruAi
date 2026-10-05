import argparse

from core.mood.model import (
    VECTOR_DEMO,
    accuracy,
    axis_accuracy,
    fit_mood,
    predict_mood,
    save_mood,
    train_phrases,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the 2-axis mood RNN and export a .pt file")
    parser.add_argument("--out", default="checkpoints/mood.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    model, vocab = fit_mood(epochs=args.epochs, seed=args.seed)
    train_acc = accuracy(model, vocab, train_phrases())
    train_axis_acc = axis_accuracy(model, vocab, train_phrases())
    print(
        f"train_phrase_acc={train_acc:.3f} train_axis_acc={train_axis_acc:.3f} "
        f"vocab={len(vocab)}"
    )
    print()
    for text in VECTOR_DEMO:
        kind_name, kind, fun_name, fun = predict_mood(model, vocab, text)
        print(
            f"{text:24} {kind_name} {kind:.2f} | {fun_name} {fun:.2f}   "
            f"vector=({kind:.2f}, {fun:.2f})"
        )
    save_mood(
        args.out,
        model,
        vocab,
        train_phrase_acc=train_acc,
        train_axis_acc=train_axis_acc,
    )
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
