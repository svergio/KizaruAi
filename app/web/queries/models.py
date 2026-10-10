"""Checkpoint metadata (sidecar JSON only) and models API."""
import json
from pathlib import Path

from app.web.queries.common import ROOT

PROD_CHECKPOINT_NAMES = frozenset(
    {
        "kizaru.pt",
        "kizaru_word.pt",
        "transformer.pt",
        "gpt.pt",
        "gpt_ru_rap_v1-prod.pt",
        "deepseek.pt",
        "bert_gen.pt",
        "mood.pt",
    }
)

CHECKPOINTS = (
    ("LSTM", ROOT / "checkpoints" / "kizaru.pt"),
    ("Transformer", ROOT / "checkpoints" / "transformer.pt"),
    ("GPT", ROOT / "checkpoints" / "gpt_ru_rap_v1-prod.pt"),
    ("DeepSeek", ROOT / "checkpoints" / "deepseek.pt"),
    ("Mood", ROOT / "checkpoints" / "mood.pt"),
)


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _read_sidecar_meta(path: Path) -> dict | None:
    side = _sidecar_path(path)
    if not side.is_file():
        return None
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _collect_model_scores():
    rows = []
    for label, path in CHECKPOINTS:
        meta = _read_sidecar_meta(path) if path.is_file() else None
        if meta is None and not path.is_file():
            rows.append(
                {
                    "label": label,
                    "path": str(path.relative_to(ROOT)),
                    "present": False,
                    "mean_loss": None,
                    "train_axis_acc": None,
                    "extra": "",
                }
            )
            continue
        mean_loss = None
        if meta:
            mean_loss = meta.get("val_ru_rap")
            if mean_loss is None:
                mean_loss = meta.get("mean_loss") or meta.get("best_loss")
        extra_bits = []
        if meta:
            if meta.get("step") is not None:
                extra_bits.append(f"step={meta['step']}")
            if meta.get("arch"):
                extra_bits.append(f"arch={meta['arch']}")
            if meta.get("bpc") is not None:
                extra_bits.append(f"bpc={meta['bpc']}")
        elif path.is_file():
            extra_bits.append("нет метаданных")
        rows.append(
            {
                "label": label,
                "path": str(path.relative_to(ROOT)),
                "present": path.is_file(),
                "mean_loss": mean_loss,
                "train_axis_acc": None,
                "extra": ", ".join(extra_bits),
            }
        )
    return rows


def api_models(db_path: Path | None = None) -> dict:
    ckpt_dir = ROOT / "checkpoints"
    models = []
    if not ckpt_dir.is_dir():
        return {"models": models}
    for path in sorted(ckpt_dir.glob("*.pt")):
        sidecar = _read_sidecar_meta(path)
        missing_metadata = path.is_file() and sidecar is None
        meta = sidecar or {}
        models.append(
            {
                "name": path.name,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "present": path.is_file(),
                "prod": path.name in PROD_CHECKPOINT_NAMES,
                "missing_metadata": missing_metadata,
                "tokenizer": meta.get("tokenizer"),
                "vocab_size": meta.get("vocab_size"),
                "step": meta.get("step"),
                "val_ru_rap": meta.get("val_ru_rap"),
                "bpc": meta.get("bpc"),
                "date": meta.get("date"),
                "arch": meta.get("arch"),
                "level": meta.get("level"),
                "note": "нет метаданных" if missing_metadata else None,
            }
        )
    return {"models": models}
