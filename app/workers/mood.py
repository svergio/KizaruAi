from pathlib import Path

import torch

from core.bus.cli import base_parser
from core.bus.handlers import job_mood
from core.bus.worker_loop import run_worker_loop
from core.logging_setup import configure_logging
from core.mood.model import MOOD_CHECKPOINT_KIND, load_mood

KIND = "mood"


def main():
    log = configure_logging("worker", worker_kind=KIND)
    parser = base_parser("Mood worker")
    parser.add_argument("model", help="Path to mood .pt")
    args = parser.parse_args()
    mood_model, vocab, meta = load_mood(args.model, device=torch.device("cpu"))
    log.info(
        "loaded mood %s kind=%s vocab=%s weights=cpu infer=lazy",
        args.model,
        meta.get("kind", MOOD_CHECKPOINT_KIND),
        len(vocab),
    )
    runtime = {
        "mood": (mood_model, vocab),
        "device": None,
        "checkpoint": Path(args.model).name,
        "count_tokens": lambda text: max(1, len(str(text).split())),
    }
    run_worker_loop(
        KIND,
        lambda payload: job_mood(payload, runtime),
        args.db,
        args.poll,
        runtime=runtime,
    )


if __name__ == "__main__":
    main()
