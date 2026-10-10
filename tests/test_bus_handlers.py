"""core/bus/handlers.py: обработчики заданий воркеров на крошечных моделях (CPU)."""
import base64

import numpy as np
import pytest
import torch

from core.bus import handlers
from core.image.ascii import HEADER
from core.mood.model import fit_mood
from core.text.lm import MiniGPT
from core.text.rnn import TextRNN

CHARS = list("абвгд \n")
C2I = {c: i for i, c in enumerate(CHARS)}
I2C = {i: c for c, i in C2I.items()}
CPU = torch.device("cpu")


def test_job_text_lstm():
    np.random.seed(0)
    torch.manual_seed(0)
    runtime = {
        "lstm_model": TextRNN(len(CHARS), 8, 8, 1),
        "char_to_idx": C2I,
        "idx_to_char": I2C,
        "device": CPU,
    }
    out = handlers.job_text_lstm({"start": "а", "length": 50, "lines": 2}, runtime)
    assert out["type"] == "message" and isinstance(out["content"], str)
    assert len(out["content"]) <= handlers.DISCORD_LIMIT


def test_job_text_gpt():
    np.random.seed(0)
    model = MiniGPT(len(CHARS), d_model=16, n_layers=1, n_heads=2, block_size=16)
    out = handlers.job_text_gpt({"start": "б", "length": 40}, {"gpt": (model, C2I, I2C), "device": CPU})
    assert out["type"] == "message" and out["content"].startswith("б")


@pytest.mark.parametrize("fn, key", [(handlers.job_text_gpt, "gpt"), (handlers.job_text_deepseek, "deepseek"),
                                     (handlers.job_text_transformer, "transformer"), (handlers.job_text_bert_gen, "bert_gen")])
def test_text_handlers_without_model(fn, key):
    with pytest.raises(RuntimeError, match="not loaded"):
        fn({}, {key: None})


def test_job_mood():
    model, vocab = fit_mood(epochs=1, seed=1)
    out = handlers.job_mood({"source": "мне весело " * 30}, {"mood": (model, vocab), "device": CPU})
    assert out["type"] == "message"
    assert out["content"].startswith("Настроение:") and "vector=(" in out["content"]
    assert out["content"].endswith("...")
    with pytest.raises(RuntimeError):
        handlers.job_mood({"source": "x"}, {"mood": None})


def test_job_pic_still_simple_chunks(small_png_bytes):
    payload = {"mode_key": "2", "data_b64": base64.b64encode(small_png_bytes).decode(), "filename": "a.png"}
    out = handlers.job_pic_still(payload, None)
    assert out["type"] == "chunks" and out["chunks"]
    assert out["chunks"][0].startswith(HEADER)
    assert all("```" in c for c in out["chunks"])


def test_job_pic_still_simple_file(small_png_bytes, tmp_path, monkeypatch):
    from core.image import render

    monkeypatch.setattr(render, "pick_device", lambda *a, **k: CPU)
    out_path = tmp_path / "out" / "pic.png"
    payload = {
        "mode_key": "2",
        "data_b64": base64.b64encode(small_png_bytes).decode(),
        "filename": "a.png",
        "out_path": str(out_path),
        "cells": 24,
        "scale": 1,
    }
    out = handlers.job_pic_still(payload, None)
    assert out["type"] == "file" and out_path.is_file()
    assert out_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_job_pic_still_errors(tmp_path):
    with pytest.raises(RuntimeError):
        handlers.job_pic_still({"mode_key": "2"}, None)
    with pytest.raises(RuntimeError, match="source not found"):
        handlers.job_pic_still({"mode_key": "3", "source_path": str(tmp_path / "nope.png")}, None)


def test_job_pic_video_rejects_simple():
    with pytest.raises(RuntimeError, match="simple"):
        handlers.job_pic_video({"mode_key": "2"}, None)
