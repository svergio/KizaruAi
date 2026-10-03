import argparse

from kizaru.mood import NOTEBOOK_TEST, accuracy, fit_mood, predict_mood, save_mood, train_phrases


def parse_args():
    parser = argparse.ArgumentParser(description="Train the reply mood RNN and export a .pt file")
    parser.add_argument("--out", default="checkpoints/mood.pt")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    model, vocab = fit_mood(epochs=args.epochs, seed=args.seed)
    train_acc = accuracy(model, vocab, train_phrases())
    test_acc = accuracy(model, vocab, NOTEBOOK_TEST)
    print(f"train_acc={train_acc:.3f} notebook_test_acc={test_acc:.3f} vocab={len(vocab)}")
    for text, label in NOTEBOOK_TEST.items():
        pred, score = predict_mood(model, vocab, text)
        print(f"{int(label)} {pred:10} {score:.2f}  {text}")
    save_mood(args.out, model, vocab, train_acc=train_acc, notebook_test_acc=test_acc)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
