import torch

from core.bus.handlers import job_text_lstm_word
from core.text.checkpoint import load_model
from core.bus.cli import base_parser
from core.bus.worker_loop import run_worker_loop

KIND = "text_lstm_word"


def main():
    parser = base_parser("Word LSTM text worker")
    parser.add_argument("model", help="Path to word-level LSTM .pt")
    args = parser.parse_args()
    model, token_to_idx, idx_to_token, meta = load_model(
        args.model, device=torch.device("cpu")
    )
    if meta.get("level") not in (None, "word"):
        print(f"warn: checkpoint level={meta.get('level')!r}, expected word")
    print(
        f"loaded lstm-word {args.model} vocab={len(token_to_idx)} "
        f"level={meta.get('level')} loss={meta.get('mean_loss')} "
        f"weights=cpu infer=lazy"
    )
    runtime = {
        "lstm_model": model,
        "token_to_idx": token_to_idx,
        "idx_to_token": idx_to_token,
        "device": None,
    }
    run_worker_loop(
        KIND, lambda payload: job_text_lstm_word(payload, runtime), args.db, args.poll
    )


if __name__ == "__main__":
    main()
