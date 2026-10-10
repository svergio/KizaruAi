"""LoRA binary rhyme classifier (JEF) on Qwen3 Instruct / Qwen3.5."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

BIN_LABELS = ("no_rhyme", "rhyme")
SCHEME_LABELS = ("ABAB", "ABBA", "AABB", "A-A", "other", "free")
BIN2ID = {k: i for i, k in enumerate(BIN_LABELS)}
SCHEME2ID = {k: i for i, k in enumerate(SCHEME_LABELS)}

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUT = "checkpoints/jef_rhyme_lora"
GATE_F1_MIN = 0.80
GATE_ECE_MAX = 0.10


def parse_args():
    p = argparse.ArgumentParser(description="Train JEF LoRA binary rhyme classifier")
    p.add_argument("--data-dir", default="data/rlaif/jef")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--scheme-aux-weight", type=float, default=0.15)
    p.add_argument("--soft-weight", type=float, default=0.5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-train", type=int, default=0, help="0 = all labeled rows")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--gate-f1", type=float, default=GATE_F1_MIN)
    p.add_argument("--gate-ece", type=float, default=GATE_ECE_MAX)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Load adapter from --out and evaluate test only",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Load existing LoRA adapter from --out before training",
    )
    p.add_argument(
        "--gpu-max-mem",
        default="6GiB",
        help="Per-GPU max_memory for device_map (rest offloads to CPU RAM)",
    )
    p.add_argument(
        "--cpu-max-mem",
        default="48GiB",
        help="CPU RAM budget for accelerate/device_map offload",
    )
    p.add_argument(
        "--offload-folder",
        default="var/jef_offload",
        help="Disk spill folder for accelerate offload",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def labeled_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("label_bin") in BIN2ID:
            out.append(r)
    return out


class JefDataset(Dataset):
    soft_weight = 0.5

    def __init__(self, rows: list[dict], tokenizer, max_len: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        enc = self.tok(
            r["text"],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(BIN2ID[r["label_bin"]], dtype=torch.long),
            "weight": torch.tensor(
                float(self.soft_weight if r.get("soft") else 1.0), dtype=torch.float32
            ),
            "has_scheme": torch.tensor(
                1 if r.get("label_scheme") in SCHEME2ID else 0, dtype=torch.long
            ),
            "scheme": torch.tensor(
                SCHEME2ID.get(r.get("label_scheme"), 0), dtype=torch.long
            ),
        }


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


@torch.inference_mode()
def evaluate(model, loader, device):
    from sklearn.metrics import f1_score

    model.eval()
    all_prob = []
    all_y = []
    total_loss = 0.0
    n = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        w = batch["weight"].to(device)
        out = model(input_ids=ids, attention_mask=mask, use_scheme=False)
        logits = out.logits.float()
        per = nn.functional.cross_entropy(logits, y, reduction="none")
        loss = (per * w).mean()
        total_loss += float(loss.item()) * len(y)
        n += len(y)
        all_prob.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_y.append(y.cpu().numpy())
    probs = np.concatenate(all_prob, axis=0)
    labels = np.concatenate(all_y, axis=0)
    preds = probs.argmax(axis=1)
    f1 = float(f1_score(labels, preds, average="binary", pos_label=1, zero_division=0))
    ece = expected_calibration_error(probs, labels)
    acc = float((preds == labels).mean()) if len(labels) else float("nan")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "loss": total_loss / max(n, 1),
        "f1": f1,
        "ece": ece,
        "acc": acc,
        "n": int(n),
        "pred_counts": dict(Counter(int(x) for x in preds.tolist())),
        "label_counts": dict(Counter(int(x) for x in labels.tolist())),
    }


class SchemeAuxModel(nn.Module):
    """PEFT seq-clf + soft scheme head (aux only; binary KPI is logits)."""

    def __init__(self, base, scheme_n: int):
        super().__init__()
        self.base = base
        hidden = base.config.hidden_size
        self.scheme_head = nn.Linear(hidden, scheme_n)

    @property
    def config(self):
        return self.base.config

    def forward(self, input_ids, attention_mask, use_scheme: bool = True, **kwargs):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=use_scheme,
            **kwargs,
        )
        if use_scheme and out.hidden_states is not None:
            hs = out.hidden_states[-1]
            m = attention_mask.unsqueeze(-1).to(hs.dtype)
            pooled = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
            out.scheme_logits = self.scheme_head(pooled.to(self.scheme_head.weight.dtype))
        else:
            out.scheme_logits = None
        return out

    def print_trainable_parameters(self):
        self.base.print_trainable_parameters()

    def save_pretrained(self, path, **kwargs):
        self.base.save_pretrained(path, **kwargs)
        torch.save(self.scheme_head.state_dict(), Path(path) / "scheme_head.pt")


def _bnb_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def _max_memory(gpu_max: str, cpu_max: str) -> dict:
    mem = {"cpu": cpu_max}
    if torch.cuda.is_available():
        mem[0] = gpu_max
    return mem


def _load_base_clf(
    model_id: str,
    use_4bit: bool,
    *,
    gpu_max_mem: str = "6GiB",
    cpu_max_mem: str = "48GiB",
    offload_folder: str | None = None,
    for_train: bool = True,
):
    from transformers import AutoModelForSequenceClassification

    Path(offload_folder or "var/jef_offload").mkdir(parents=True, exist_ok=True)
    max_memory = _max_memory(gpu_max_mem, cpu_max_mem)
    kwargs = {
        "num_labels": 2,
        "trust_remote_code": True,
        "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto",
        "max_memory": max_memory,
        "offload_folder": offload_folder or "var/jef_offload",
        "low_cpu_mem_usage": True,
    }
    if use_4bit:
        kwargs["quantization_config"] = _bnb_config()
    print(
        f"load_base 4bit={use_4bit} max_memory={max_memory} offload={kwargs['offload_folder']} "
        f"for_train={for_train}",
        flush=True,
    )
    return AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)


def build_model(
    model_id: str,
    use_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    device,
    *,
    adapter_dir: Path | None = None,
    gpu_max_mem: str = "6GiB",
    cpu_max_mem: str = "48GiB",
    offload_folder: str = "var/jef_offload",
    for_train: bool = True,
):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(adapter_dir) if adapter_dir and (adapter_dir / "tokenizer_config.json").is_file() else model_id,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = _load_base_clf(
        model_id,
        use_4bit,
        gpu_max_mem=gpu_max_mem,
        cpu_max_mem=cpu_max_mem,
        offload_folder=offload_folder,
        for_train=for_train,
    )
    base.config.pad_token_id = tok.pad_token_id
    if use_4bit and for_train:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    if adapter_dir is not None and (adapter_dir / "adapter_config.json").is_file():
        base = PeftModel.from_pretrained(
            base,
            str(adapter_dir),
            is_trainable=for_train,
            max_memory=_max_memory(gpu_max_mem, cpu_max_mem),
            offload_folder=offload_folder,
        )
        print(f"loaded_adapter={adapter_dir} trainable={for_train}", flush=True)
    else:
        lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        base = get_peft_model(base, lora)

    if for_train:
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        if hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    model = SchemeAuxModel(base, len(SCHEME_LABELS))
    head_device = device
    try:
        for p in base.parameters():
            if p.device.type == "cuda":
                head_device = p.device
                break
    except StopIteration:
        pass
    scheme_path = adapter_dir / "scheme_head.pt" if adapter_dir else None
    if scheme_path is not None and scheme_path.is_file():
        state = torch.load(scheme_path, map_location="cpu", weights_only=True)
        model.scheme_head.load_state_dict(state)
        print(f"loaded_scheme_head={scheme_path}", flush=True)
    model.scheme_head = model.scheme_head.to(head_device)
    return model, tok


def write_results(out_dir: Path, result: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"RESULTS test_f1={result['test_f1']:.4f} test_ece={result['test_ece']:.4f} "
        f"test_acc={result['test_acc']:.4f} gate={result['gate']} out={out_dir}",
        flush=True,
    )
    try:
        from training import mlflow_track

        mlflow_track.log_metrics(
            {
                "test_f1": float(result["test_f1"]),
                "test_ece": float(result["test_ece"]),
                "test_acc": float(result["test_acc"]),
            }
        )
        mlflow_track.set_tag("gate", str(result.get("gate", "")))
    except Exception as exc:
        print(f"mlflow_warn write_results: {exc}", flush=True)


def run_eval_only(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    if not (out_dir / "adapter_config.json").is_file():
        raise SystemExit(f"missing adapter at {out_dir}")
    data_dir = Path(args.data_dir)
    test_rows = labeled_rows(load_jsonl(data_dir / "test.jsonl"))
    train_rows = labeled_rows(load_jsonl(data_dir / "train.jsonl"))
    soft_n = sum(1 for r in train_rows if r.get("soft"))
    print(f"eval_only test={len(test_rows)} adapter={out_dir} device={device}", flush=True)

    use_4bit = (not args.no_4bit) and torch.cuda.is_available()
    model, tok = build_model(
        args.model,
        use_4bit,
        args.lora_r,
        args.lora_alpha,
        device,
        adapter_dir=out_dir,
        gpu_max_mem=args.gpu_max_mem,
        cpu_max_mem=args.cpu_max_mem,
        offload_folder=args.offload_folder,
        for_train=False,
    )
    model.eval()
    test_loader = DataLoader(
        JefDataset(test_rows, tok, args.max_len),
        batch_size=max(1, args.batch_size),
    )
    test_m = evaluate(model, test_loader, device)
    gate_pass = (test_m["f1"] >= args.gate_f1) and (test_m["ece"] <= args.gate_ece)
    result = {
        "model": args.model,
        "out": str(out_dir),
        "mode": "eval_only",
        "test_f1": test_m["f1"],
        "test_ece": test_m["ece"],
        "test_acc": test_m["acc"],
        "test_n": test_m["n"],
        "gate_f1_min": args.gate_f1,
        "gate_ece_max": args.gate_ece,
        "gate": "PASS" if gate_pass else "FAIL",
        "soft_train": soft_n,
        "train_n": len(train_rows),
        "pred_counts": test_m["pred_counts"],
        "label_counts": test_m["label_counts"],
    }
    write_results(out_dir, result)
    return result


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    train_rows = labeled_rows(load_jsonl(data_dir / "train.jsonl"))
    valid_rows = labeled_rows(load_jsonl(data_dir / "valid.jsonl"))
    test_rows = labeled_rows(load_jsonl(data_dir / "test.jsonl"))
    if args.max_train > 0:
        train_rows = train_rows[: args.max_train]

    soft_n = sum(1 for r in train_rows if r.get("soft"))
    print(
        f"labeled train={len(train_rows)} soft={soft_n} "
        f"valid={len(valid_rows)} test={len(test_rows)} device={device}",
        flush=True,
    )
    if not train_rows or not test_rows:
        raise SystemExit("no labeled rows for train/test — run build_jef_dataset first")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_4bit = (not args.no_4bit) and torch.cuda.is_available()
    adapter = out_dir if args.resume and (out_dir / "adapter_config.json").is_file() else None
    model, tok = build_model(
        args.model,
        use_4bit,
        args.lora_r,
        args.lora_alpha,
        device,
        adapter_dir=adapter,
        gpu_max_mem=args.gpu_max_mem,
        cpu_max_mem=args.cpu_max_mem,
        offload_folder=args.offload_folder,
        for_train=True,
    )
    model.print_trainable_parameters()

    JefDataset.soft_weight = args.soft_weight
    train_loader = DataLoader(
        JefDataset(train_rows, tok, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        JefDataset(valid_rows, tok, args.max_len),
        batch_size=max(1, args.batch_size),
    )
    test_loader = DataLoader(
        JefDataset(test_rows, tok, args.max_len),
        batch_size=max(1, args.batch_size),
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    best_f1 = -1.0
    step = 0
    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            w = batch["weight"].to(device)
            use_scheme = args.scheme_aux_weight > 0
            out = model(input_ids=ids, attention_mask=mask, use_scheme=use_scheme)
            per = nn.functional.cross_entropy(out.logits.float(), y, reduction="none")
            loss = (per * w).mean()
            if use_scheme and out.scheme_logits is not None:
                has = batch["has_scheme"].to(device).bool()
                if has.any():
                    sch_y = batch["scheme"].to(device)[has]
                    sch_logits = out.scheme_logits[has].float()
                    aux = nn.functional.cross_entropy(sch_logits, sch_y)
                    loss = loss + args.scheme_aux_weight * aux
            (loss / args.grad_accum).backward()
            running += float(loss.item())
            if (bi + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.eval_every == 0:
                    metrics = evaluate(model, valid_loader, device)
                    train_loss = running / args.eval_every
                    print(
                        f"epoch={epoch} step={step} train_loss={train_loss:.4f} "
                        f"valid_f1={metrics['f1']:.4f} valid_ece={metrics['ece']:.4f} "
                        f"valid_acc={metrics['acc']:.4f}",
                        flush=True,
                    )
                    try:
                        from training import mlflow_track

                        mlflow_track.log_metrics(
                            {
                                "train_loss": float(train_loss),
                                "valid_f1": float(metrics["f1"]),
                                "valid_ece": float(metrics["ece"]),
                                "valid_acc": float(metrics["acc"]),
                            },
                            step=step,
                        )
                    except Exception as exc:
                        print(f"mlflow_warn train step: {exc}", flush=True)
                    running = 0.0
                    if metrics["f1"] > best_f1:
                        best_f1 = metrics["f1"]
                        model.save_pretrained(out_dir)
                        tok.save_pretrained(out_dir)
                        (out_dir / "bin_labels.json").write_text(
                            json.dumps(list(BIN_LABELS), ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    model.train()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        if (bi + 1) % args.grad_accum != 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

    if best_f1 < 0:
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        (out_dir / "bin_labels.json").write_text(
            json.dumps(list(BIN_LABELS), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # reload best adapter for test gate
    if (out_dir / "adapter_config.json").is_file():
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, tok = build_model(
            args.model,
            use_4bit,
            args.lora_r,
            args.lora_alpha,
            device,
            adapter_dir=out_dir,
            gpu_max_mem=args.gpu_max_mem,
            cpu_max_mem=args.cpu_max_mem,
            offload_folder=args.offload_folder,
            for_train=False,
        )
        test_loader = DataLoader(
            JefDataset(test_rows, tok, args.max_len),
            batch_size=max(1, args.batch_size),
        )

    test_m = evaluate(model, test_loader, device)
    gate_pass = (test_m["f1"] >= args.gate_f1) and (test_m["ece"] <= args.gate_ece)
    result = {
        "model": args.model,
        "out": str(out_dir),
        "mode": "train",
        "test_f1": test_m["f1"],
        "test_ece": test_m["ece"],
        "test_acc": test_m["acc"],
        "test_n": test_m["n"],
        "gate_f1_min": args.gate_f1,
        "gate_ece_max": args.gate_ece,
        "gate": "PASS" if gate_pass else "FAIL",
        "soft_train": soft_n,
        "train_n": len(train_rows),
        "best_valid_f1": best_f1,
        "pred_counts": test_m["pred_counts"],
        "label_counts": test_m["label_counts"],
    }
    write_results(out_dir, result)
    return result


def main():
    args = parse_args()
    from training import mlflow_track

    run_id = mlflow_track.start_run(
        "kizaru-jef",
        run_name=f"jef_{Path(args.out).name}",
        params={
            "model": getattr(args, "model", ""),
            "out": args.out,
            "eval_only": bool(args.eval_only),
            "data_dir": getattr(args, "data_dir", ""),
        },
        tags={"project": "kizaruai", "entrypoint": "train_jef"},
    )
    if run_id:
        print(f"mlflow_run={run_id}", flush=True)
    status = "FINISHED"
    try:
        if args.eval_only:
            run_eval_only(args)
        else:
            train(args)
    except Exception:
        status = "FAILED"
        raise
    finally:
        mlflow_track.end_run(status=status)


if __name__ == "__main__":
    main()
