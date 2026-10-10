from pathlib import Path

import torch

from core.bus.cli import base_parser
from core.bus.handlers import job_text_bert_gen
from core.bus.worker_loop import run_worker_loop
from core.logging_setup import configure_logging
from core.text.checkpoint import load_model
from core.text.model_tokenizer import ModelVocabTokenizer

KIND = "text_bert_gen"


def main():
    log = configure_logging("worker", worker_kind=KIND)
    parser = base_parser("BERT-gen MLM text worker")
    parser.add_argument("model", help="Path to bert_gen .pt")
    args = parser.parse_args()
    model, chars, idx, meta = load_model(args.model, device=torch.device("cpu"))
    level = meta.get("level") or "char"
    log.info(
        "loaded bert_gen %s arch=%s level=%s vocab=%s weights=cpu infer=lazy",
        args.model,
        meta.get("arch"),
        level,
        len(chars),
    )
    runtime = {
        "bert_gen": (model, chars, idx),
        "device": None,
        "level": level,
        "checkpoint": Path(args.model).name,
        "hf_tokenizer": ModelVocabTokenizer(chars, level=level, idx_to_token=idx),
    }
    run_worker_loop(
        KIND,
        lambda payload: job_text_bert_gen(payload, runtime),
        args.db,
        args.poll,
        runtime=runtime,
    )


if __name__ == "__main__":
    main()
