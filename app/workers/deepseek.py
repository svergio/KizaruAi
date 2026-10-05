from core.bus.handlers import job_text_deepseek
from core.text.checkpoint import get_device, load_model
from core.bus.cli import base_parser
from core.bus.worker_loop import run_worker_loop

KIND = "text_deepseek"


def main():
    parser = base_parser("DeepSeek text worker")
    parser.add_argument("model", help="Path to deepseek .pt")
    args = parser.parse_args()
    device = get_device()
    model, chars, idx, meta = load_model(args.model, device=device)
    print(f"loaded deepseek {args.model} arch={meta.get('arch')} vocab={len(chars)}")
    runtime = {"deepseek": (model, chars, idx), "device": device}
    run_worker_loop(
        KIND, lambda payload: job_text_deepseek(payload, runtime), args.db, args.poll
    )


if __name__ == "__main__":
    main()
