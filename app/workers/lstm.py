import torch

from core.bus.handlers import job_text_lstm
from core.text.checkpoint import load_model
from core.bus.cli import base_parser
from core.bus.worker_loop import run_worker_loop

KIND = "text_lstm"


def main():
    parser = base_parser("LSTM text worker")
    parser.add_argument("model", help="Path to kizaru LSTM .pt")
    args = parser.parse_args()
    model, char_to_idx, idx_to_char, meta = load_model(
        args.model, device=torch.device("cpu")
    )
    print(
        f"loaded lstm {args.model} vocab={len(char_to_idx)} "
        f"loss={meta.get('mean_loss')} weights=cpu infer=lazy"
    )
    runtime = {
        "lstm_model": model,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "device": None,
    }
    run_worker_loop(KIND, lambda payload: job_text_lstm(payload, runtime), args.db, args.poll)


if __name__ == "__main__":
    main()
