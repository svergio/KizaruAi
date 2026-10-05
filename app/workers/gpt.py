from core.bus.handlers import job_text_gpt
from core.text.checkpoint import get_device, load_model
from core.bus.cli import base_parser
from core.bus.worker_loop import run_worker_loop

KIND = "text_gpt"


def main():
    parser = base_parser("GPT text worker")
    parser.add_argument("model", help="Path to gpt .pt")
    args = parser.parse_args()
    device = get_device()
    model, chars, idx, meta = load_model(args.model, device=device)
    print(f"loaded gpt {args.model} arch={meta.get('arch')} vocab={len(chars)}")
    runtime = {"gpt": (model, chars, idx), "device": device}
    run_worker_loop(KIND, lambda payload: job_text_gpt(payload, runtime), args.db, args.poll)


if __name__ == "__main__":
    main()
