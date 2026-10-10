from pathlib import Path

import torch

from core.bus.cli import base_parser
from core.bus.handlers import job_text_gpt
from core.bus.worker_loop import run_worker_loop
from core.logging_setup import configure_logging
from core.text.checkpoint import load_model
from core.text.model_tokenizer import ModelVocabTokenizer

KIND = "text_gpt"


def main():
    log = configure_logging("worker", worker_kind=KIND)
    parser = base_parser("GPT text worker")
    parser.add_argument("model", help="Path to gpt .pt")
    args = parser.parse_args()
    model, chars, idx, meta = load_model(args.model, device=torch.device("cpu"))
    level = meta.get("level") or "char"
    bpe_tok = None
    if level == "bpe":
        from core.text.bpe import load_bpe, resolve_tokenizer_file

        tok_path = resolve_tokenizer_file(meta.get("tokenizer_path") or args.model)
        bpe_tok = load_bpe(tok_path)
        log.info("loaded bpe tokenizer %s vocab=%s", tok_path, bpe_tok.get_vocab_size())
    log.info(
        "loaded gpt %s arch=%s level=%s vocab=%s weights=cpu infer=lazy",
        args.model,
        meta.get("arch"),
        level,
        len(chars),
    )
    runtime = {
        "gpt": (model, chars, idx),
        "device": None,
        "level": level,
        "checkpoint": Path(args.model).name,
        "bpe_tokenizer": bpe_tok,
        "hf_tokenizer": ModelVocabTokenizer(
            chars, level=level, idx_to_token=idx, bpe_tokenizer=bpe_tok
        ),
    }
    run_worker_loop(
        KIND,
        lambda payload: job_text_gpt(payload, runtime),
        args.db,
        args.poll,
        runtime=runtime,
    )


if __name__ == "__main__":
    main()
