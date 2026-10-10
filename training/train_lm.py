import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from core.text.checkpoint import (
    ARCH_BERT_GEN,
    ARCH_DEEPSEEK,
    ARCH_GPT,
    ARCH_TRANSFORMER,
    get_device,
    load_model,
    save_model,
)
from core.corpus.lyrics import BATCH_SIZE, get_batch, load_corpus, text_to_seq
from core.text.bert_gen import BertMLM, apply_mlm_mask, generate_mlm
from core.text.bpe import (
    decode_ids,
    load_bpe,
    resolve_tokenizer_file,
    text_to_bpe_seq,
    tokenizer_path_for_ckpt,
    train_bpe,
    vocab_maps,
)
from core.text.deepseek import MiniDeepSeek
from core.text.lm import MiniGPT, SimpleTransformer
from core.text.pretrained_vocab import (
    DEFAULT_TOKENIZER,
    DEFAULT_WEIGHTS,
    NAVEC_DIM,
    init_embedding_from_navec,
    load_whitelist_file,
)
from core.text.sample import evaluate_lm, evaluate_lm_bpe, evaluate_lm_words
from core.text.tokenize import TOKEN_UNK, WORD_MIN_FREQ, detokenize, text_to_word_seq, tokenize
from training import mlflow_track

ARCHES = {
    ARCH_TRANSFORMER: SimpleTransformer,
    ARCH_GPT: MiniGPT,
    ARCH_DEEPSEEK: MiniDeepSeek,
    ARCH_BERT_GEN: BertMLM,
}

DEFAULT_VAL_BATCHES = 32
LEVELS = ("char", "word", "bpe")
BEST_METRICS = ("val_focus", "val_all")


def checkpoint_paths(out: str | Path) -> tuple[Path, Path]:
    path = Path(out)
    return path.with_name(f"{path.stem}_best{path.suffix}"), path.with_name(
        f"{path.stem}_last{path.suffix}"
    )


def default_val_paths(corpus: str | None) -> tuple[str | None, str | None]:
    if not corpus:
        return None, None
    base = Path(corpus)
    return str(base.with_name("val.txt")), str(base.with_name("val_ru_rap.txt"))


def encode_with_vocab(text: str, token_to_idx: dict, level: str, bpe_tok=None):
    if level == "bpe":
        if bpe_tok is None:
            raise ValueError("bpe tokenizer required")
        sequence, _, _ = text_to_bpe_seq(text, bpe_tok)
        return sequence
    if level == "word":
        sequence, _, _ = text_to_word_seq(text, token_to_idx=token_to_idx)
        return sequence
    ids = [token_to_idx[ch] for ch in text if ch in token_to_idx]
    if len(ids) < 2:
        raise ValueError("char sequence too short after vocab filter")
    return np.asarray(ids, dtype=np.int64)


def unk_rate(text: str, token_to_idx: dict, level: str, bpe_tok=None) -> float:
    if level == "bpe":
        if bpe_tok is None or not text:
            return 0.0
        ids = bpe_tok.encode(text).ids
        if not ids:
            return 0.0
        unk_id = bpe_tok.token_to_id("<unk>")
        if unk_id is None:
            return 0.0
        return float(sum(1 for i in ids if i == unk_id)) / float(len(ids))
    if level == "word":
        tokens = tokenize(text)
        if not tokens:
            return 0.0
        unk = token_to_idx[TOKEN_UNK]
        n_unk = sum(1 for tok in tokens if token_to_idx.get(tok, unk) == unk)
        return float(n_unk) / float(len(tokens))
    if not text:
        return 0.0
    missing = sum(1 for ch in text if ch not in token_to_idx)
    return float(missing) / float(len(text))


def ids_to_char_count(ids, level: str, idx_to_token: dict, bpe_tok=None) -> int:
    flat = [int(x) for x in np.asarray(ids).reshape(-1).tolist()]
    if level == "char":
        return len(flat)
    if level == "bpe":
        if bpe_tok is None:
            return max(1, len(flat))
        return max(1, len(decode_ids(bpe_tok, flat)))
    tokens = [idx_to_token.get(i, "") for i in flat]
    return max(1, len(detokenize([t for t in tokens if t])))


def make_fixed_batches(sequence, seq_len, batch_size, n_batches, seed, level, idx_to_token, bpe_tok=None):
    if len(sequence) <= seq_len:
        raise ValueError(f"Corpus is shorter than seq_len={seq_len}: {len(sequence)}")
    rng = np.random.RandomState(seed)
    max_start = len(sequence) - seq_len
    batches = []
    for _ in range(n_batches):
        trains = []
        targets = []
        n_chars = 0
        for _ in range(batch_size):
            start = int(rng.randint(0, max_start))
            chunk = sequence[start : start + seq_len]
            trains.append(torch.LongTensor(chunk[:-1]).view(-1, 1))
            tgt = chunk[1:]
            targets.append(torch.LongTensor(tgt).view(-1, 1))
            n_chars += ids_to_char_count(tgt, level, idx_to_token, bpe_tok=bpe_tok)
        batches.append(
            (torch.stack(trains, dim=0), torch.stack(targets, dim=0), int(n_chars))
        )
    return batches


def _batch_loss(model, train, target, criterion, device, arch, reduction="mean"):
    crit = criterion if reduction == "mean" else nn.CrossEntropyLoss(reduction=reduction)
    if arch == ARCH_BERT_GEN:
        tokens = train.squeeze(-1).to(device)
        block = min(tokens.size(1), model.block_size)
        tokens = tokens[:, :block]
        inputs, labels = apply_mlm_mask(tokens, model.mask_id, model.vocab_size)
        logits = model(inputs)
        return crit(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    tokens = train.squeeze(-1).to(device)
    labels = target.squeeze(-1).to(device)
    logits = model(tokens)
    return crit(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def mean_ce_on_batches(model, batches, criterion, device, arch) -> float:
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for train, target, _n_chars in batches:
            loss = _batch_loss(model, train, target, criterion, device, arch)
            losses.append(float(loss.item()))
    model.train(was_training)
    return float(np.mean(losses)) if losses else float("nan")


def bpc_on_batches(model, batches, device, arch) -> float:
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_chars = 0
    with torch.no_grad():
        for train, target, n_chars in batches:
            loss = _batch_loss(
                model, train, target, nn.CrossEntropyLoss(), device, arch, reduction="sum"
            )
            total_nll += float(loss.item())
            total_chars += int(n_chars)
    model.train(was_training)
    if total_chars <= 0:
        return float("nan")
    return float(total_nll / total_chars)


def bits_per_char(total_nll: float, n_chars: int) -> float:
    if n_chars <= 0:
        return float("nan")
    return float(total_nll) / float(n_chars)


def eval_val_metrics(
    model,
    focus_batches,
    all_batches,
    criterion,
    device,
    arch,
) -> tuple[float, float, float]:
    val_focus = mean_ce_on_batches(model, focus_batches, criterion, device, arch)
    val_all = mean_ce_on_batches(model, all_batches, criterion, device, arch)
    bpc = bpc_on_batches(model, all_batches, device, arch)
    return val_focus, val_all, bpc


def _copy_tokenizer_beside(src_tok: Path, ckpt_path: Path) -> Path:
    dest = tokenizer_path_for_ckpt(ckpt_path)
    if src_tok.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_tok, dest)
    return dest


def _assert_shape_match(model, state_dict: dict, src: Path):
    own = model.state_dict()
    for key, tensor in state_dict.items():
        if key not in own:
            continue
        if tuple(own[key].shape) != tuple(tensor.shape):
            raise ValueError(
                f"--init-from {src}: shape mismatch for {key}: "
                f"ckpt {tuple(tensor.shape)} != model {tuple(own[key].shape)}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train transformer, mini-GPT, MiniDeepSeek, or BERT-gen MLM"
    )
    parser.add_argument("--arch", choices=sorted(ARCHES), required=True)
    parser.add_argument("--level", choices=LEVELS, default="char")
    parser.add_argument("--corpus", default=None)
    parser.add_argument(
        "--val-corpus",
        default=None,
        help="full val corpus (default: val.txt next to --corpus)",
    )
    parser.add_argument(
        "--val-focus",
        default=None,
        help="focus val corpus (default: val_ru_rap.txt next to --corpus)",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=DEFAULT_VAL_BATCHES,
        help="fixed number of val batches per eval",
    )
    parser.add_argument(
        "--best-metric",
        choices=BEST_METRICS,
        default="val_focus",
        help="metric for _best.pt and --patience (default: val_focus)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="stop after N evals without best-metric improvement (0=off)",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=None,
        help="tok emb dim (default=d_model); use 300 with --init-navec",
    )
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--kv-lora-rank",
        type=int,
        default=None,
        help="MLA KV latent rank (deepseek only; default d_model//4)",
    )
    parser.add_argument(
        "--q-lora-rank",
        type=int,
        default=None,
        help="MLA Q latent rank (deepseek only; default d_model//4)",
    )
    parser.add_argument("--min-freq", type=int, default=WORD_MIN_FREQ)
    parser.add_argument(
        "--whitelist",
        default=None,
        help="txt word list (e.g. data/embeddings/navec_hudlit/seed_whitelist.txt)",
    )
    parser.add_argument(
        "--init-navec",
        action="store_true",
        help="init word tok embedding rows from local Navec hudlit safetensors",
    )
    parser.add_argument("--navec-tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--navec-weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="BPE vocab size when --level bpe (train on --corpus)",
    )
    parser.add_argument(
        "--vocab-from",
        default=None,
        help="load BPE tokenizer from checkpoint (or .tokenizer.json)",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="load weights (+tokenizer if bpe) from checkpoint before training",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="load .pt weights+vocab and continue from --start-step or ckpt epoch",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=None,
        help="step index to continue from when --resume is set (default: ckpt epoch)",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="log rolling/batch loss to MLflow every N steps (no sample gen)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=500,
        help="val metrics + sample + checkpoint-on-best every N steps",
    )
    parser.add_argument("--gen-len", type=int, default=None)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--start", default="Я")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment", default="kizaru-lm")
    parser.add_argument("--mlflow-run-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_mlflow:
        import os

        os.environ["KIZARU_MLFLOW"] = "0"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    level = args.level
    out = args.out or (
        f"checkpoints/{args.arch}_{level}.pt"
        if level != "char"
        else f"checkpoints/{args.arch}.pt"
    )
    out_best, out_last = checkpoint_paths(out)
    block_size = args.block_size or (64 if level in ("word", "bpe") else 256)
    gen_len = args.gen_len or (48 if level in ("word", "bpe") else 200)
    embedding_size = args.embedding_size
    if embedding_size is None and args.init_navec:
        embedding_size = NAVEC_DIM
    if embedding_size is None:
        embedding_size = args.d_model

    if args.init_navec and level == "bpe":
        raise SystemExit("--init-navec is not available for --level bpe (word-only Navec init)")
    if args.init_navec and level != "word":
        raise SystemExit("--init-navec requires --level word (char models have no word rows)")
    if args.whitelist and level != "word":
        raise SystemExit("--whitelist requires --level word")
    if level == "bpe" and args.arch == ARCH_BERT_GEN:
        raise SystemExit("--level bpe is not supported for bert_gen")

    default_val, default_focus = default_val_paths(args.corpus)
    val_corpus = args.val_corpus or default_val
    val_focus = args.val_focus or default_focus
    if not val_corpus or not Path(val_corpus).is_file():
        raise FileNotFoundError(
            f"val corpus missing: {val_corpus!r} (pass --val-corpus or place val.txt next to train)"
        )
    if not val_focus or not Path(val_focus).is_file():
        raise FileNotFoundError(
            f"val focus missing: {val_focus!r} "
            f"(pass --val-focus or place val_ru_rap.txt next to train)"
        )

    device = get_device()
    text_sample = load_corpus(args.corpus)
    text_val_all = load_corpus(val_corpus)
    text_val_focus = load_corpus(val_focus)
    whitelist = None
    if level == "word" and args.whitelist:
        whitelist = load_whitelist_file(args.whitelist)

    bpe_tok = None
    tok_file = None
    start_step = 0
    best_metric_val = float("inf")
    best_step = 0
    best_focus_at_best = None
    tie_weights = level == "bpe"
    init_from = args.init_from

    if args.resume and init_from:
        raise SystemExit("use either --resume or --init-from, not both")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file() or resume_path.stat().st_size < 1024:
            raise FileNotFoundError(f"resume checkpoint missing/empty: {resume_path}")
        model, token_to_idx, idx_to_token, meta = load_model(str(resume_path), device=device)
        ckpt_level = meta.get("level") or level
        if ckpt_level != level:
            raise ValueError(f"resume level={ckpt_level!r} != --level {level!r}")
        ckpt_arch = meta.get("arch") or args.arch
        if ckpt_arch != args.arch:
            raise ValueError(f"resume arch={ckpt_arch!r} != --arch {args.arch!r}")
        if level == "bpe":
            tok_file = resolve_tokenizer_file(meta.get("tokenizer_path") or resume_path)
            bpe_tok = load_bpe(tok_file)
            sequence, token_to_idx, idx_to_token = text_to_bpe_seq(text_sample, bpe_tok)
        elif level == "word":
            sequence, token_to_idx, idx_to_token = text_to_word_seq(
                text_sample, token_to_idx=token_to_idx
            )
        else:
            sequence, _, _ = text_to_seq(text_sample)
        start_step = (
            args.start_step
            if args.start_step is not None
            else int(meta.get("epoch") or meta.get("step") or 0)
        )
        start_step = max(0, min(start_step, args.steps))
        for key in (args.best_metric, "val_focus", "val_ru_rap", "mean_loss"):
            if meta.get(key) is not None:
                best_metric_val = float(meta[key])
                break
        best_step = int(meta.get("step") or meta.get("epoch") or 0)
        best_focus_at_best = meta.get("val_focus", meta.get("val_ru_rap"))
        embedding_size = int(meta.get("embedding_size") or embedding_size)
        block_size = int(meta.get("block_size") or block_size)
        tie_weights = bool(meta.get("tie_weights", tie_weights))
        print(
            f"resume={resume_path} start_step={start_step}/{args.steps} "
            f"best_{args.best_metric}={best_metric_val:.6f} vocab={len(token_to_idx)}"
        )
    else:
        if level == "bpe":
            if args.vocab_from:
                tok_file = resolve_tokenizer_file(args.vocab_from)
                bpe_tok = load_bpe(tok_file)
                print(f"vocab_from={tok_file} vocab={bpe_tok.get_vocab_size()}")
            elif init_from:
                tok_file = resolve_tokenizer_file(init_from)
                bpe_tok = load_bpe(tok_file)
                print(f"bpe_from_init={tok_file} vocab={bpe_tok.get_vocab_size()}")
            else:
                tok_file = tokenizer_path_for_ckpt(out)
                bpe_tok = train_bpe(text_sample, args.vocab_size, tok_file)
                print(f"bpe_trained {tok_file} vocab={bpe_tok.get_vocab_size()}")
            sequence, token_to_idx, idx_to_token = text_to_bpe_seq(text_sample, bpe_tok)
            embedding_size = args.d_model
            tie_weights = True
        elif level == "word":
            sequence, token_to_idx, idx_to_token = text_to_word_seq(
                text_sample, min_freq=args.min_freq, whitelist=whitelist
            )
        else:
            sequence, token_to_idx, idx_to_token = text_to_seq(text_sample)

        model_kwargs = dict(
            vocab_size=len(idx_to_token),
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            block_size=block_size,
            embedding_size=embedding_size,
        )
        if args.arch in (ARCH_GPT, ARCH_TRANSFORMER):
            model_kwargs["tie_weights"] = tie_weights
        if args.arch == ARCH_DEEPSEEK:
            model_kwargs["kv_lora_rank"] = args.kv_lora_rank
            model_kwargs["q_lora_rank"] = args.q_lora_rank
        if args.arch == ARCH_BERT_GEN:
            model_kwargs["vocab_size"] = len(idx_to_token) + 1
        model = ARCHES[args.arch](**model_kwargs).to(device)

        if init_from:
            src = Path(init_from)
            if not src.is_file():
                raise FileNotFoundError(f"--init-from missing: {src}")
            payload = torch.load(src, map_location="cpu", weights_only=False)
            state = payload.get("state_dict") or payload.get("model_state")
            if not isinstance(state, dict):
                raise ValueError(f"--init-from {src}: no state_dict")
            _assert_shape_match(model, state, src)
            model.load_state_dict(state)
            print(f"init_from={src} loaded weights lr={args.lr}")

        if level == "word" and args.init_navec:
            if not Path(args.navec_weights).is_file():
                raise FileNotFoundError(args.navec_weights)
            if embedding_size != NAVEC_DIM:
                print(
                    f"navec_note: embedding_size={embedding_size} != navec={NAVEC_DIM}; "
                    f"init will truncate/pad into model.tok"
                )
            if embedding_size != args.d_model:
                print(
                    f"navec_note: tok emb {embedding_size}d -> Linear -> d_model={args.d_model}"
                )
            stats = init_embedding_from_navec(
                model.tok,
                token_to_idx,
                tokenizer_path=args.navec_tokenizer,
                weights_path=args.navec_weights,
            )
            print(
                f"navec_init filled={stats['filled']} skipped={stats['skipped']} "
                f"dim={stats['source_dim']}->{stats['target_dim']}"
            )

    if level == "bpe" and tok_file is not None:
        _copy_tokenizer_beside(Path(tok_file), Path(out))
        tok_file = tokenizer_path_for_ckpt(out)

    seq_val_all = encode_with_vocab(text_val_all, token_to_idx, level, bpe_tok=bpe_tok)
    seq_val_focus = encode_with_vocab(text_val_focus, token_to_idx, level, bpe_tok=bpe_tok)
    val_unk = unk_rate(text_val_all, token_to_idx, level, bpe_tok=bpe_tok)
    seq_len = block_size + 1
    n_val_batches = max(1, int(args.val_batches))
    focus_batches = make_fixed_batches(
        seq_val_focus,
        seq_len,
        args.batch_size,
        n_val_batches,
        args.seed + 1,
        level,
        idx_to_token,
        bpe_tok=bpe_tok,
    )
    all_batches = make_fixed_batches(
        seq_val_all,
        seq_len,
        args.batch_size,
        n_val_batches,
        args.seed + 2,
        level,
        idx_to_token,
        bpe_tok=bpe_tok,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"arch={args.arch} level={level} device={device} "
        f"text_chars={len(text_sample)} tokens={len(sequence)} vocab={len(token_to_idx)} "
        f"whitelist={0 if whitelist is None else len(whitelist)} "
        f"d_model={args.d_model} emb={embedding_size} tie={tie_weights} params={n_params} "
        f"val_unk_rate={val_unk:.6f} val_batches={n_val_batches} "
        f"patience={args.patience} best_metric={args.best_metric}"
    )

    run_name = args.mlflow_run_name or f"{args.arch}_{level}_{Path(out).stem}"
    run_id = mlflow_track.start_run(
        args.mlflow_experiment,
        run_name=run_name,
        params={
            "arch": args.arch,
            "level": level,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "d_model": args.d_model,
            "embedding_size": embedding_size,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "block_size": block_size,
            "lr": args.lr,
            "log_every": args.log_every,
            "eval_every": args.eval_every,
            "val_batches": n_val_batches,
            "patience": args.patience,
            "best_metric": args.best_metric,
            "start_step": start_step,
            "resume": args.resume or "",
            "init_from": init_from or "",
            "vocab_from": args.vocab_from or "",
            "out": out,
            "out_best": str(out_best),
            "out_last": str(out_last),
            "val_corpus": val_corpus,
            "val_focus": val_focus,
            "n_params": n_params,
            "vocab": len(token_to_idx),
            "val_unk_rate": val_unk,
            "tie_weights": tie_weights,
        },
        tags={"project": "kizaruai", "entrypoint": "train_lm"},
    )
    if run_id:
        print(f"mlflow_run={run_id}", flush=True)
    mlflow_track.log_metrics({"val_unk_rate": val_unk}, step=start_step)

    def _save_ckpt(path, step, val_focus_v, val_all_v, bpc_v):
        from training.ckpt_meta import write_sidecar_from_train

        meta = dict(
            epoch=step,
            step=step,
            mean_loss=val_focus_v,
            val_focus=val_focus_v,
            val_all=val_all_v,
            val_ru_rap=val_focus_v,
            bpc=bpc_v,
            level=level,
            tie_weights=tie_weights,
        )
        tok_label = level
        if tok_file is not None:
            tok_copied = str(_copy_tokenizer_beside(Path(tok_file), Path(path)))
            meta["tokenizer_path"] = tok_copied
            tok_label = tok_copied
        save_model(path, model, token_to_idx, idx_to_token, **meta)
        write_sidecar_from_train(
            Path(path),
            tokenizer=tok_label,
            vocab_size=len(token_to_idx),
            step=int(step) if step is not None else None,
            val_ru_rap=float(val_focus_v) if val_focus_v is not None else None,
            bpc=float(bpc_v) if bpc_v is not None else None,
            arch=args.arch,
            level=level,
        )

    if start_step >= args.steps:
        _save_ckpt(
            out_last,
            args.steps,
            best_metric_val if best_metric_val < float("inf") else None,
            None,
            None,
        )
        print(
            f"done (already at steps), best={out_best} last={out_last} "
            f"best_{args.best_metric}={best_metric_val:.6f}"
        )
        mlflow_track.end_run()
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log_every = max(1, int(args.log_every))
    eval_every = max(1, int(args.eval_every))
    patience = max(0, int(args.patience))
    loss_log = []
    loss_eval = []
    bad_evals = 0
    status = "FINISHED"
    stopped_early = False
    step_1 = start_step
    try:
        for step in range(start_step, args.steps):
            model.train()
            if args.arch == ARCH_BERT_GEN:
                train, _target = get_batch(
                    sequence, seq_len=block_size, batch_size=args.batch_size
                )
                tokens = train.squeeze(-1).to(device)[:, :block_size]
                inputs, labels = apply_mlm_mask(tokens, model.mask_id, model.vocab_size)
                logits = model(inputs)
            else:
                train, target = get_batch(sequence, seq_len=seq_len, batch_size=args.batch_size)
                tokens = train.squeeze(-1).to(device)
                labels = target.squeeze(-1).to(device)
                logits = model(tokens)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_item = loss.item()
            loss_log.append(loss_item)
            loss_eval.append(loss_item)
            step_1 = step + 1

            if step_1 % log_every == 0:
                roll_loss = float(np.mean(loss_log))
                mlflow_track.log_metrics({"loss": roll_loss}, step=step_1)
                loss_log = []
                if step_1 % eval_every != 0:
                    if log_every >= 50 or step_1 % max(eval_every, 50) == 0:
                        print(f"step={step_1} loss={roll_loss:.6f}", flush=True)

            if step_1 % eval_every == 0:
                mean_loss = float(np.mean(loss_eval))
                val_focus_v, val_all_v, bpc_v = eval_val_metrics(
                    model, focus_batches, all_batches, criterion, device, args.arch
                )
                print(
                    f"step={step_1} loss={mean_loss:.6f} "
                    f"val_focus={val_focus_v:.6f} val_all={val_all_v:.6f} "
                    f"bpc={bpc_v:.6f} val_unk_rate={val_unk:.6f}",
                    flush=True,
                )
                mlflow_track.log_metrics(
                    {
                        "val_focus": val_focus_v,
                        "val_all": val_all_v,
                        "bpc": bpc_v,
                        "val_unk_rate": val_unk,
                    },
                    step=step_1,
                )
                if args.arch == ARCH_BERT_GEN:
                    sample = generate_mlm(
                        model,
                        token_to_idx,
                        idx_to_token,
                        start_text=args.start,
                        seq_len=min(128, block_size),
                        n_iters=16,
                        temp=args.temp,
                        device=device,
                        max_lines=6,
                        level=level if level != "bpe" else "char",
                    )
                elif level == "bpe":
                    sample = evaluate_lm_bpe(
                        model,
                        bpe_tok,
                        start_text=args.start,
                        prediction_len=gen_len,
                        temp=args.temp,
                        device=device,
                        max_lines=6,
                    )
                elif level == "word":
                    sample = evaluate_lm_words(
                        model,
                        token_to_idx,
                        idx_to_token,
                        start_text=args.start,
                        prediction_len=gen_len,
                        temp=args.temp,
                        device=device,
                        max_lines=6,
                    )
                else:
                    sample = evaluate_lm(
                        model,
                        token_to_idx,
                        idx_to_token,
                        start_text=args.start,
                        prediction_len=gen_len,
                        temp=args.temp,
                        device=device,
                        max_lines=6,
                    )
                print(sample)
                print("-" * 40)
                loss_eval = []
                scored = val_focus_v if args.best_metric == "val_focus" else val_all_v
                if scored < best_metric_val:
                    best_metric_val = scored
                    best_step = step_1
                    best_focus_at_best = val_focus_v
                    bad_evals = 0
                    _save_ckpt(out_best, step_1, val_focus_v, val_all_v, bpc_v)
                    print(
                        f"saved best {out_best} step={step_1} "
                        f"{args.best_metric}={scored:.6f} val_focus={val_focus_v:.6f}"
                    )
                    mlflow_track.log_metrics(
                        {f"best_{args.best_metric}": best_metric_val}, step=step_1
                    )
                else:
                    bad_evals += 1
                    if patience and bad_evals >= patience:
                        print(
                            f"early_stop patience={patience} bad_evals={bad_evals} "
                            f"best_{args.best_metric}={best_metric_val:.6f} "
                            f"best_step={best_step}",
                            flush=True,
                        )
                        stopped_early = True
                        break

        final_step = step_1 if stopped_early else args.steps
        _save_ckpt(
            out_last,
            final_step,
            best_focus_at_best if best_focus_at_best is not None else best_metric_val,
            None,
            None,
        )
        print(
            f"done, best={out_best} last={out_last} "
            f"best_{args.best_metric}={best_metric_val:.6f} best_step={best_step}"
            + (" early_stop" if stopped_early else "")
        )
        mlflow_track.log_metrics(
            {f"best_{args.best_metric}": best_metric_val},
            step=final_step,
        )
    except KeyboardInterrupt:
        status = "KILLED"
        raise
    except Exception:
        status = "FAILED"
        raise
    finally:
        mlflow_track.end_run(status=status)


if __name__ == "__main__":
    main()
