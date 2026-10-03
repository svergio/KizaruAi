from pathlib import Path

import torch
from torch import nn

from kizaru.model import TextRNN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _vocab_on_model(model, char_to_idx, idx_to_char):
    model.char_to_idx = dict(char_to_idx)
    model.idx_to_char = {int(key): value for key, value in idx_to_char.items()}


def save_model(path, model, char_to_idx, idx_to_char, **meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    was_training = model.training
    _vocab_on_model(model, char_to_idx, idx_to_char)
    model.to("cpu")
    model.eval()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "char_to_idx": model.char_to_idx,
            "idx_to_char": model.idx_to_char,
            "hidden_size": model.hidden_size,
            "embedding_size": model.embedding_size,
            "n_layers": model.n_layers,
            **meta,
        },
        path,
    )
    model.to(device)
    model.train(was_training)
    return path


def load_model(path, device=None):
    if device is None:
        device = get_device()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, nn.Module):
        model = payload
        char_to_idx = dict(model.char_to_idx)
        idx_to_char = {int(key): value for key, value in model.idx_to_char.items()}
        meta = {}
    elif isinstance(payload, dict) and ("state_dict" in payload or "model_state" in payload):
        idx_to_char = {int(key): value for key, value in payload["idx_to_char"].items()}
        char_to_idx = dict(payload["char_to_idx"])
        model = TextRNN(
            input_size=len(idx_to_char),
            hidden_size=payload["hidden_size"],
            embedding_size=payload["embedding_size"],
            n_layers=payload["n_layers"],
        )
        model.load_state_dict(payload.get("state_dict", payload.get("model_state")))
        _vocab_on_model(model, char_to_idx, idx_to_char)
        meta = {
            key: value
            for key, value in payload.items()
            if key not in {"model", "state_dict", "model_state", "char_to_idx", "idx_to_char", "hidden_size", "embedding_size", "n_layers"}
        }
    else:
        raise ValueError(f"Unsupported model file: {path}")

    model.to(device)
    model.eval()
    return model, char_to_idx, idx_to_char, meta
