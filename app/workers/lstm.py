from pathlib import Path

import torch

from core.bus.cli import base_parser
from core.bus.handlers import job_text_lstm
from core.bus.worker_loop import run_worker_loop
from core.logging_setup import configure_logging
from core.text.checkpoint import load_model
from core.text.model_tokenizer import ModelVocabTokenizer

KIND = "text_lstm"


def main():
    log = configure_logging("worker", worker_kind=KIND)
    parser = base_parser("LSTM text worker")
    parser.add_argument("model", help="Path to kizaru LSTM .pt")
    args = parser.parse_args()
    model, char_to_idx, idx_to_char, meta = load_model(
        args.model, device=torch.device("cpu")
    )
    log.info(
        "loaded lstm %s vocab=%s loss=%s weights=cpu infer=lazy",
        args.model,
        len(char_to_idx),
        meta.get("mean_loss"),
    )
    runtime = {
        "lstm_model": model,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "device": None,
        "checkpoint": Path(args.model).name,
        "hf_tokenizer": ModelVocabTokenizer(
            char_to_idx, level="char", idx_to_token=idx_to_char
        ),
    }
    run_worker_loop(
        KIND,
        lambda payload: job_text_lstm(payload, runtime),
        args.db,
        args.poll,
        runtime=runtime,
    )


if __name__ == "__main__":
    main()
