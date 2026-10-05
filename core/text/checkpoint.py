from pathlib import Path

import torch
from torch import nn

from core.text.bert_gen import BertMLM
from core.text.deepseek import MiniDeepSeek
from core.text.lm import MiniGPT, SimpleTransformer
from core.text.rnn import TextRNN

ARCH_RNN = "rnn"
ARCH_TRANSFORMER = "transformer"
ARCH_GPT = "gpt"
ARCH_DEEPSEEK = "deepseek"
ARCH_BERT_GEN = "bert_gen"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _vocab_on_model(model, char_to_idx, idx_to_char):
    model.char_to_idx = dict(char_to_idx)
    model.idx_to_char = {int(key): value for key, value in idx_to_char.items()}


def _arch_of(model):
    if isinstance(model, BertMLM):
        return ARCH_BERT_GEN
    if isinstance(model, MiniDeepSeek):
        return ARCH_DEEPSEEK
    if isinstance(model, MiniGPT):
        return ARCH_GPT
    if isinstance(model, SimpleTransformer):
        return ARCH_TRANSFORMER
    return ARCH_RNN


def save_model(path, model, char_to_idx, idx_to_char, **meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    was_training = model.training
    _vocab_on_model(model, char_to_idx, idx_to_char)
    model.to("cpu")
    model.eval()
    arch = _arch_of(model)
    payload = {
        "arch": arch,
        "state_dict": model.state_dict(),
        "char_to_idx": model.char_to_idx,
        "idx_to_char": model.idx_to_char,
        **meta,
    }
    if arch == ARCH_RNN:
        payload["hidden_size"] = model.hidden_size
        payload["embedding_size"] = model.embedding_size
        payload["n_layers"] = model.n_layers
    else:
        payload["d_model"] = model.d_model
        payload["n_layers"] = model.n_layers
        payload["n_heads"] = model.n_heads
        payload["block_size"] = model.block_size
        if arch == ARCH_DEEPSEEK:
            payload["kv_lora_rank"] = model.kv_lora_rank
            payload["q_lora_rank"] = model.q_lora_rank
        if arch == ARCH_BERT_GEN:
            payload["mask_id"] = model.mask_id
    torch.save(payload, path)
    model.to(device)
    model.train(was_training)
    return path


def _build_lm(arch, vocab_size, payload):
    kwargs = {
        "vocab_size": vocab_size,
        "d_model": payload["d_model"],
        "n_layers": payload["n_layers"],
        "n_heads": payload["n_heads"],
        "block_size": payload["block_size"],
    }
    if arch == ARCH_BERT_GEN:
        kwargs["mask_id"] = payload.get("mask_id", vocab_size - 1)
        return BertMLM(**kwargs)
    if arch == ARCH_DEEPSEEK:
        kwargs["kv_lora_rank"] = payload.get("kv_lora_rank")
        kwargs["q_lora_rank"] = payload.get("q_lora_rank")
        return MiniDeepSeek(**kwargs)
    if arch == ARCH_GPT:
        return MiniGPT(**kwargs)
    return SimpleTransformer(**kwargs)


def _torch_load_local(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def load_model(path, device=None):
    if device is None:
        device = get_device()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    payload = _torch_load_local(path, device)
    if isinstance(payload, nn.Module):
        model = payload
        char_to_idx = dict(model.char_to_idx)
        idx_to_char = {int(key): value for key, value in model.idx_to_char.items()}
        meta = {"arch": _arch_of(model)}
    elif isinstance(payload, dict) and ("state_dict" in payload or "model_state" in payload):
        idx_to_char = {int(key): value for key, value in payload["idx_to_char"].items()}
        char_to_idx = dict(payload["char_to_idx"])
        arch = payload.get("arch", ARCH_RNN)
        if arch in {None, ARCH_RNN}:
            model = TextRNN(
                input_size=len(idx_to_char),
                hidden_size=payload["hidden_size"],
                embedding_size=payload["embedding_size"],
                n_layers=payload["n_layers"],
            )
        elif arch in {ARCH_TRANSFORMER, ARCH_GPT, ARCH_DEEPSEEK, ARCH_BERT_GEN}:
            vocab_size = len(idx_to_char)
            if arch == ARCH_BERT_GEN:
                vocab_size = int(payload.get("mask_id", vocab_size)) + 1
            model = _build_lm(arch, vocab_size, payload)
        else:
            raise ValueError(f"Unknown arch {arch!r} in {path}")
        model.load_state_dict(payload.get("state_dict", payload.get("model_state")))
        _vocab_on_model(model, char_to_idx, idx_to_char)
        skip = {
            "model",
            "state_dict",
            "model_state",
            "char_to_idx",
            "idx_to_char",
            "hidden_size",
            "embedding_size",
            "n_layers",
            "d_model",
            "n_heads",
            "block_size",
            "kv_lora_rank",
            "q_lora_rank",
            "mask_id",
        }
        meta = {key: value for key, value in payload.items() if key not in skip}
        meta["arch"] = arch if arch else ARCH_RNN
    else:
        raise ValueError(f"Unsupported model file: {path}")

    model.to(device)
    model.eval()
    return model, char_to_idx, idx_to_char, meta
