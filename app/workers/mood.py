import torch

from core.bus.handlers import job_mood
from core.mood.model import MOOD_CHECKPOINT_KIND, load_mood
from core.bus.cli import base_parser
from core.bus.worker_loop import run_worker_loop

KIND = "mood"


def main():
    parser = base_parser("Mood worker")
    parser.add_argument("model", help="Path to mood .pt")
    args = parser.parse_args()
    mood_model, vocab, meta = load_mood(args.model, device=torch.device("cpu"))
    print(
        f"loaded mood {args.model} kind={meta.get('kind', MOOD_CHECKPOINT_KIND)} "
        f"vocab={len(vocab)} weights=cpu infer=lazy"
    )
    runtime = {"mood": (mood_model, vocab), "device": None}
    run_worker_loop(KIND, lambda payload: job_mood(payload, runtime), args.db, args.poll)


if __name__ == "__main__":
    main()
