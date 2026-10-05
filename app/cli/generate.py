import argparse
from pathlib import Path

from core.bus.broker import DEFAULT_DB, emit_job_result, enqueue_and_wait
from core.bus.queue import JobStore
from core.bus.routing import build_text_payload
from core.text.params import describe, resolve_text_params


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enqueue text generation (same queue as Discord /ai text)"
    )
    parser.add_argument(
        "--arch",
        choices=("lstm", "lstm-word", "transformer", "gpt", "deepseek", "bert-gen"),
        default="lstm",
        help="Model worker (default: lstm)",
    )
    parser.add_argument("--start", default="Я", help="Seed line")
    parser.add_argument(
        "--temp",
        type=float,
        default=None,
        help="Sampling temperature; lower repeats lines, higher breaks words",
    )
    parser.add_argument(
        "--lines", type=int, default=None, help="How many non-empty lines to emit"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="Budget in chars (tokens for lstm-word, window for bert-gen)",
    )
    parser.add_argument(
        "--iters", type=int, default=None, help="bert-gen only: Gibbs passes"
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help="SQLite job broker (default: data/bot.db)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    store = JobStore(Path(args.db), recover_running=False)
    payload = build_text_payload(
        args.arch,
        args.start,
        temp=args.temp,
        length=args.length,
        lines=args.lines,
        iters=args.iters,
    )
    print(describe(args.arch, resolve_text_params(args.arch, payload)))
    result = enqueue_and_wait(store, "text", payload)
    emit_job_result(result)


if __name__ == "__main__":
    main()
