from pathlib import Path

import torch

from core.bus.cli import base_parser
from core.bus.handlers import job_text_lstm_word
from core.bus.worker_loop import run_worker_loop
from core.logging_setup import configure_logging
from core.text.checkpoint import load_model
from core.text.model_tokenizer import ModelVocabTokenizer

KIND = "text_lstm_word"


def main():
    log = configure_logging("worker", worker_kind=KIND)
    parser = base_parser("Word LSTM text worker")
    parser.add_argument("model", help="Path to word-level LSTM .pt")
    args = parser.parse_args()
    model, token_to_idx, idx_to_token, meta = load_model(
        args.model, device=torch.device("cpu")
    )
    if meta.get("level") not in (None, "word"):
        log.warning("checkpoint level=%r, expected word", meta.get("level"))
    log.info(
        "loaded lstm-word %s vocab=%s level=%s loss=%s weights=cpu infer=lazy",
        args.model,
        len(token_to_idx),
        meta.get("level"),
        meta.get("mean_loss"),
    )
    runtime = {
        "lstm_model": model,
        "token_to_idx": token_to_idx,
        "idx_to_token": idx_to_token,
        "device": None,
        "checkpoint": Path(args.model).name,
        "hf_tokenizer": ModelVocabTokenizer(
            token_to_idx, level="word", idx_to_token=idx_to_token
        ),
    }
    run_worker_loop(
        KIND,
        lambda payload: job_text_lstm_word(payload, runtime),
        args.db,
        args.poll,
        runtime=runtime,
    )


if __name__ == "__main__":
    main()
