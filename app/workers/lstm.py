from core.bus.handlers import job_text_lstm
from core.text.checkpoint import get_device, load_model
from core.bus.cli import base_parser
from core.bus.worker_loop import run_worker_loop

KIND = "text_lstm"


def main():
    parser = base_parser("LSTM text worker")
    parser.add_argument("model", help="Path to kizaru LSTM .pt")
    args = parser.parse_args()
    device = get_device()
    model, char_to_idx, idx_to_char, meta = load_model(args.model, device=device)
    print(f"loaded lstm {args.model} vocab={len(char_to_idx)} loss={meta.get('mean_loss')}")
    runtime = {
        "lstm_model": model,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "device": device,
    }
    run_worker_loop(KIND, lambda payload: job_text_lstm(payload, runtime), args.db, args.poll)


if __name__ == "__main__":
    main()
