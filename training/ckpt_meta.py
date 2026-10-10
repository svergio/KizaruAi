"""Sidecar metadata for checkpoints: <name>.pt.meta.json (no weights in admin).

CLI: python -m training.ckpt_meta --all checkpoints/
     python -m training.ckpt_meta path/to/model.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Classes that may appear in older local checkpoints under checkpoints/.
# weights_only=False is allowed only in this module for those files (see load_checkpoint_dict).
_SAFE_GLOBAL_REPORT: list[str] = []


def sidecar_path(ckpt: Path) -> Path:
    return ckpt.with_name(ckpt.name + ".meta.json")


def _iso_from_mtime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return datetime.now(timezone.utc).isoformat()


def meta_from_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    char_to_idx = payload.get("char_to_idx") or {}
    vocab_size = payload.get("vocab_size")
    if vocab_size is None and isinstance(char_to_idx, dict):
        vocab_size = len(char_to_idx)
    tokenizer = payload.get("tokenizer_path") or payload.get("tokenizer") or payload.get("level")
    if tokenizer is not None:
        tokenizer = str(tokenizer)
    step = payload.get("step")
    if step is None:
        step = payload.get("epoch")
    val_ru_rap = payload.get("val_ru_rap")
    if val_ru_rap is None:
        val_ru_rap = payload.get("val_focus")
    if val_ru_rap is None:
        val_ru_rap = payload.get("mean_loss")
    bpc = payload.get("bpc")
    date = payload.get("date") or _iso_from_mtime(path)
    return {
        "name": path.name,
        "tokenizer": tokenizer,
        "vocab_size": int(vocab_size) if vocab_size is not None else None,
        "step": int(step) if step is not None else None,
        "val_ru_rap": float(val_ru_rap) if val_ru_rap is not None else None,
        "bpc": float(bpc) if bpc is not None else None,
        "date": str(date),
        "arch": payload.get("arch"),
        "level": payload.get("level"),
    }


def write_sidecar(path: Path, meta: dict[str, Any]) -> Path:
    path = Path(path)
    side = sidecar_path(path)
    side.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return side


def write_sidecar_from_train(
    path: Path,
    *,
    tokenizer: str | None,
    vocab_size: int | None,
    step: int | None,
    val_ru_rap: float | None,
    bpc: float | None,
    arch: str | None = None,
    level: str | None = None,
) -> Path:
    meta = {
        "name": Path(path).name,
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "step": step,
        "val_ru_rap": val_ru_rap,
        "bpc": bpc,
        "date": datetime.now(timezone.utc).isoformat(),
        "arch": arch,
        "level": level,
    }
    return write_sidecar(path, meta)


def _try_add_safe_globals(exc: BaseException) -> list[str]:
    """Register classes named in a weights_only / UnpicklingError message."""
    import torch

    added: list[str] = []
    text = str(exc)
    # Typical: "Unsupported global: GLOBAL pathlib.Path was not an allowed global"
    for part in text.replace("`", " ").replace("'", " ").split():
        if part.count(".") != 1 and "pathlib" not in part and "torch." not in part:
            continue
        if not any(ch.isalpha() for ch in part):
            continue
        try:
            mod_name, cls_name = part.rsplit(".", 1)
            if not cls_name[0:1].isupper() and cls_name not in ("Path", "PosixPath", "WindowsPath"):
                continue
            import importlib

            mod = importlib.import_module(mod_name)
            obj = getattr(mod, cls_name, None)
            if obj is None:
                continue
            torch.serialization.add_safe_globals([obj])
            name = f"{mod_name}.{cls_name}"
            if name not in _SAFE_GLOBAL_REPORT:
                _SAFE_GLOBAL_REPORT.append(name)
            added.append(name)
        except Exception:
            continue
    # Always allow pathlib.Path for local ckpts.
    try:
        from pathlib import Path as PathCls

        torch.serialization.add_safe_globals([PathCls])
        if "pathlib.Path" not in _SAFE_GLOBAL_REPORT:
            _SAFE_GLOBAL_REPORT.append("pathlib.Path")
            added.append("pathlib.Path")
    except Exception:
        pass
    return added


def load_checkpoint_dict(path: Path) -> dict[str, Any]:
    """Load checkpoint metadata payload from a local file under checkpoints/ (or CLI path)."""
    import torch

    path = Path(path)
    try:
        data = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    except Exception as exc:
        _try_add_safe_globals(exc)
        try:
            data = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        except Exception as exc2:
            _try_add_safe_globals(exc2)
            # Last resort for pre-weights_only local checkpoints only.
            # weights_only=False is intentional here: files are operator-local under
            # checkpoints/, not loaded by the admin web process.
            data = torch.load(
                path, map_location="cpu", weights_only=False, mmap=True
            )
    if not isinstance(data, dict):
        raise TypeError(f"checkpoint is not a dict: {path}")
    return data


def extract_and_write(path: Path) -> dict[str, Any]:
    path = Path(path)
    payload = load_checkpoint_dict(path)
    meta = meta_from_payload(path, payload)
    write_sidecar(path, meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write .pt.meta.json sidecars for checkpoints")
    parser.add_argument("paths", nargs="*", type=Path, help="checkpoint .pt files")
    parser.add_argument(
        "--all",
        type=Path,
        metavar="DIR",
        help="process all *.pt in DIR (non-recursive)",
    )
    args = parser.parse_args(argv)
    targets: list[Path] = []
    if args.all:
        d = args.all
        if not d.is_dir():
            print(f"not a directory: {d}", file=sys.stderr)
            return 2
        targets.extend(sorted(d.glob("*.pt")))
    targets.extend(args.paths)
    if not targets:
        parser.error("pass .pt paths or --all DIR")
    for path in targets:
        if not path.is_file():
            print(f"skip missing: {path}", file=sys.stderr)
            continue
        try:
            meta = extract_and_write(path)
            print(f"ok {path} -> {sidecar_path(path)} step={meta.get('step')}")
        except Exception as exc:
            print(f"fail {path}: {exc}", file=sys.stderr)
    if _SAFE_GLOBAL_REPORT:
        print("add_safe_globals: " + ", ".join(_SAFE_GLOBAL_REPORT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
