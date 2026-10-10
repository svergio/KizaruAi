"""Cancel / MAX_RUNTIME interrupt generators; tokens_out matches model vocab."""
import base64
import io
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from core.bus import handlers
from core.bus.cancel_flag import (
    GenerationCancelled,
    check_cancelled,
    get_cancel_event,
    set_cancel_event,
)
from core.bus.queue import MAX_RUNTIME_SEC
from core.bus.routing import WORKER_KINDS, resolve_worker_kind
from core.bus.worker_loop import _token_counts, run_assigned_once
from core.image import anim
from core.mood import model as mood_mod
from core.text import deepseek as deepseek_mod
from core.text import sample as sample_mod
from core.text.bert_gen import BertMLM
from core.text.lm import MiniGPT, SimpleTransformer
from core.text.model_tokenizer import ModelVocabTokenizer
from core.text.rnn import TextRNN
from core.text.tokenize import TOKEN_NL, TOKEN_UNK

# 1x1 PNG
_PNG = base64.b64encode(
    (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
).decode("ascii")

_KIND_SPECS = [
    ("text_lstm", "text", {"model": "lstm", "start": "a", "length": 40, "lines": 12}),
    (
        "text_lstm_word",
        "text",
        {"model": "lstm-word", "start": "a", "length": 8, "lines": 12},
    ),
    (
        "text_transformer",
        "text",
        {"model": "transformer", "start": "a", "length": 40, "lines": 12},
    ),
    ("text_gpt", "text", {"model": "gpt", "start": "a", "length": 40, "lines": 12}),
    (
        "text_deepseek",
        "text",
        {"model": "deepseek", "start": "a", "length": 40, "lines": 12},
    ),
    (
        "text_bert_gen",
        "text",
        {"model": "bert-gen", "start": "a", "length": 40, "iters": 8, "lines": 12},
    ),
    ("mood", "mood", {"source": "злой"}),
    (
        "pic_still",
        "pic",
        {"mode_key": "2", "data_b64": _PNG, "filename": "a.png"},
    ),
    (
        "pic_video",
        "pic",
        {
            "mode_key": "1",
            "animated": True,
            "as_file": True,
            "data_b64": _PNG,
            "filename": "a.gif",
        },
    ),
]


@pytest.fixture(autouse=True)
def _clear_cancel_flag():
    yield
    set_cancel_event(None)


def _tiny_char_vocab():
    alphabet = "abc "
    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def _tiny_word_vocab():
    vocab = {TOKEN_UNK: 0, TOKEN_NL: 1, "a": 2, "b": 3}
    idx = {i: t for t, i in vocab.items()}
    return vocab, idx


def _set_flag_at_step(monkeypatch, event, steps, at=3):
    real = check_cancelled

    def counting():
        steps["n"] += 1
        if steps["n"] >= at:
            event.set()
        real()

    monkeypatch.setattr("core.bus.cancel_flag.check_cancelled", counting)


def _arm_mid_cancel(monkeypatch, job_store, job_id, uid, *, at=2):
    """Set DB cancelled + event on Nth check_cancelled inside a real handler."""
    steps = {"n": 0}
    real = check_cancelled

    def counting():
        steps["n"] += 1
        if steps["n"] >= at:
            job_store.cancel(job_id, uid)
            ev = get_cancel_event()
            if ev is not None:
                ev.set()
        real()

    monkeypatch.setattr("core.bus.cancel_flag.check_cancelled", counting)
    return steps


def _runtime_for_kind(worker_kind: str) -> dict:
    device = torch.device("cpu")
    if worker_kind == "text_lstm":
        chars, idx = _tiny_char_vocab()
        return {
            "lstm_model": TextRNN(len(chars), 8, 8),
            "char_to_idx": chars,
            "idx_to_char": idx,
            "device": device,
        }
    if worker_kind == "text_lstm_word":
        vocab, idx = _tiny_word_vocab()
        return {
            "lstm_model": TextRNN(len(vocab), 8, 8),
            "token_to_idx": vocab,
            "idx_to_token": idx,
            "device": device,
        }
    if worker_kind == "text_transformer":
        chars, idx = _tiny_char_vocab()
        model = SimpleTransformer(
            len(chars), d_model=16, n_layers=1, n_heads=2, block_size=16, dropout=0.0
        )
        return {"transformer": (model, chars, idx), "level": "char", "device": device}
    if worker_kind == "text_gpt":
        chars, idx = _tiny_char_vocab()
        model = MiniGPT(
            len(chars), d_model=16, n_layers=1, n_heads=2, block_size=16, dropout=0.0
        )
        return {"gpt": (model, chars, idx), "level": "char", "device": device}
    if worker_kind == "text_deepseek":
        chars, idx = _tiny_char_vocab()
        model = deepseek_mod.MiniDeepSeek(
            len(chars),
            d_model=16,
            n_layers=1,
            n_heads=2,
            block_size=16,
            dropout=0.0,
            kv_lora_rank=4,
            q_lora_rank=4,
        )
        return {"deepseek": (model, chars, idx), "level": "char", "device": device}
    if worker_kind == "text_bert_gen":
        chars, idx = _tiny_char_vocab()
        # mask id = last vocab slot
        model = BertMLM(
            len(chars), d_model=16, n_layers=1, n_heads=2, block_size=48, dropout=0.0
        )
        return {"bert_gen": (model, chars, idx), "level": "char", "device": device}
    if worker_kind == "mood":
        model, vocab = mood_mod.fit_mood(epochs=1, seed=0)
        return {"mood": (model, vocab), "device": device}
    return {"device": device}


def _handler_for_kind(worker_kind: str):
    return {
        "text_lstm": handlers.job_text_lstm,
        "text_lstm_word": handlers.job_text_lstm_word,
        "text_transformer": handlers.job_text_transformer,
        "text_gpt": handlers.job_text_gpt,
        "text_deepseek": handlers.job_text_deepseek,
        "text_bert_gen": handlers.job_text_bert_gen,
        "mood": handlers.job_mood,
        "pic_still": handlers.job_pic_still,
        "pic_video": handlers.job_pic_video,
    }[worker_kind]


@pytest.mark.parametrize("worker_kind,bot_kind,payload", _KIND_SPECS)
def test_cancel_flag_stops_each_worker_kind(
    job_store, worker_kind, bot_kind, payload, monkeypatch
):
    uid = f"owner-{worker_kind}"
    job_id = job_store.enqueue(uid, bot_kind, payload)
    job_store.route_one(resolve_worker_kind)
    assert job_store.get_job(job_id)["worker_kind"] == worker_kind

    runtime = _runtime_for_kind(worker_kind)
    handler = _handler_for_kind(worker_kind)
    steps = _arm_mid_cancel(monkeypatch, job_store, job_id, uid, at=2)

    if worker_kind == "text_deepseek":
        real_stop = deepseek_mod.CancelFlagStoppingCriteria.__call__

        def counting_stop(self, input_ids, scores, **kwargs):
            steps["n"] += 1
            if steps["n"] >= 2:
                job_store.cancel(job_id, uid)
                ev = get_cancel_event()
                if ev is not None:
                    ev.set()
            return real_stop(self, input_ids, scores, **kwargs)

        monkeypatch.setattr(
            deepseek_mod.CancelFlagStoppingCriteria, "__call__", counting_stop
        )

    if worker_kind == "pic_still":

        def fake_render_pic(*_a, **_k):
            job_store.cancel(job_id, uid)
            ev = get_cancel_event()
            if ev is not None:
                ev.set()
            return "ok\n"

        monkeypatch.setattr(handlers, "render_pic", fake_render_pic)

    if worker_kind == "pic_video":

        def fake_anim(source, out, **_kwargs):
            Path(out).write_bytes(b"GIF89a")
            job_store.cancel(job_id, uid)
            ev = get_cancel_event()
            if ev is not None:
                ev.set()

        monkeypatch.setattr(handlers, "render_anim_gif", fake_anim)

    assert run_assigned_once(
        job_store,
        worker_kind,
        lambda p: handler(p, runtime),
        runtime=runtime,
    )
    assert job_store.get_job(job_id)["status"] == "cancelled"
    assert steps["n"] >= 1


def test_sample_evaluate_stops_by_step_4(monkeypatch):
    char_to_idx, idx_to_char = _tiny_char_vocab()
    model = TextRNN(len(char_to_idx), 8, 8)
    event = threading.Event()
    set_cancel_event(event)
    steps = {"n": 0}
    try:
        _set_flag_at_step(monkeypatch, event, steps, at=3)
        with pytest.raises(GenerationCancelled):
            sample_mod.evaluate(
                model,
                char_to_idx,
                idx_to_char,
                start_text="a",
                prediction_len=20,
                temp=1.0,
            )
        assert steps["n"] <= 4
    finally:
        set_cancel_event(None)


def test_deepseek_stopping_criteria_stops_by_step_4(monkeypatch):
    char_to_idx, idx_to_char = _tiny_char_vocab()
    model = deepseek_mod.MiniDeepSeek(
        len(char_to_idx),
        d_model=16,
        n_layers=1,
        n_heads=2,
        block_size=16,
        dropout=0.0,
        kv_lora_rank=4,
        q_lora_rank=4,
    )
    event = threading.Event()
    set_cancel_event(event)
    steps = {"n": 0}
    real_stop = deepseek_mod.CancelFlagStoppingCriteria.__call__

    def counting_stop(self, input_ids, scores, **kwargs):
        steps["n"] += 1
        if steps["n"] >= 3:
            event.set()
        return real_stop(self, input_ids, scores, **kwargs)

    try:
        monkeypatch.setattr(
            deepseek_mod.CancelFlagStoppingCriteria, "__call__", counting_stop
        )
        with pytest.raises(GenerationCancelled):
            deepseek_mod.generate(
                model,
                char_to_idx,
                idx_to_char,
                start_text="a",
                prediction_len=20,
                temp=1.0,
                level="char",
            )
        assert steps["n"] <= 4
    finally:
        set_cancel_event(None)


def test_mood_predict_checks_cancel(monkeypatch):
    model, vocab = mood_mod.fit_mood(epochs=1, seed=0)
    event = threading.Event()
    set_cancel_event(event)
    steps = {"n": 0}
    real = check_cancelled

    def counting():
        steps["n"] += 1
        if steps["n"] >= 1:
            event.set()
        real()

    try:
        monkeypatch.setattr("core.bus.cancel_flag.check_cancelled", counting)
        with pytest.raises(GenerationCancelled):
            mood_mod.predict_mood(model, vocab, "злой")
        assert steps["n"] <= 2
    finally:
        set_cancel_event(None)


class _FakePipe:
    def __init__(self, chunks=None):
        self._buf = io.BytesIO(b"".join(chunks or []))
        self._writes = []

    def read(self, n=-1):
        return self._buf.read(n)

    def write(self, data):
        self._writes.append(bytes(data))
        return len(data)

    def close(self):
        return None


class _FakeProc:
    def __init__(self, stdout=None, returncode=0):
        self.stdout = stdout
        self.stderr = io.BytesIO(b"")
        self.stdin = _FakePipe()
        self.returncode = returncode
        self._alive = True
        self.killed = 0

    def poll(self):
        return None if self._alive else self.returncode

    def wait(self):
        self._alive = False
        return self.returncode

    def kill(self):
        self.killed += 1
        self._alive = False


def test_stream_anim_cancel_kills_ffmpeg_and_drops_out(tmp_path, monkeypatch):
    monkeypatch.setattr(anim, "_VAR_DIR", tmp_path)
    anim._NVENC = None
    w, h = 2, 2
    frame = np.zeros((h, w, 3), dtype=np.uint8).tobytes()
    frames = [frame, frame, frame]
    decoder = _FakeProc(stdout=_FakePipe(frames))
    encoder = _FakeProc()
    out = tmp_path / "cancel_me.mp4"
    event = threading.Event()
    set_cancel_event(event)
    batches = {"n": 0}
    gray = torch.zeros((1, 4, 4), dtype=torch.uint8)

    def fake_popen(cmd, **kwargs):
        return decoder if "pipe:1" in cmd else encoder

    def fake_render(*args, **kwargs):
        batches["n"] += 1
        if batches["n"] >= 1:
            event.set()
        return gray

    try:
        with patch("core.image.anim.require_tool"), patch(
            "core.image.anim.probe_video",
            return_value={"width": w, "height": h, "fps": 30.0, "frames": 3},
        ), patch("core.image.anim.pick_device", return_value=torch.device("cpu")), patch(
            "core.image.anim.pick_gpu_batch", return_value=1
        ), patch("core.image.anim._has_nvenc", return_value=False), patch(
            "core.image.anim.render_rgb_batch", side_effect=fake_render
        ), patch(
            "core.image.anim._needs_file_line_mask", return_value=False
        ), patch(
            "core.image.anim.subprocess.Popen", side_effect=fake_popen
        ):
            with pytest.raises(GenerationCancelled):
                anim.stream_anim(
                    "src.mp4",
                    out,
                    mode="line",
                    cells=1,
                    discord=True,
                )
        assert decoder.killed >= 1
        assert encoder.killed >= 1
        assert not out.is_file()
    finally:
        set_cancel_event(None)


# Fixed strings + hand-counted tokens per tokenizer (not derived from encode() in assert).
_TOKEN_CASES = [
    # char vocab "abc ": "ab c" -> a,b,space,c = 4
    ("text_lstm", "ab c", 4, "char"),
    ("text_transformer", "ab c", 4, "char"),
    ("text_gpt", "ab c", 4, "char"),
    ("text_deepseek", "ab c", 4, "char"),
    ("text_bert_gen", "ab c", 4, "char"),
    # word vocab: "я на" -> ["я", "на"] = 2
    ("text_lstm_word", "я на", 2, "word"),
    # whitespace split: 6 tokens
    (
        "mood",
        "Настроение: злой 0.10 | грустный 0.20",
        6,
        "wordsplit",
    ),
    # default counter = len chars; chunks "aa"+"bbb" = 5
    ("pic_still", None, 5, "chunks"),
    ("pic_video", None, 5, "chunks"),
]


@pytest.mark.parametrize("worker_kind,text,expected,mode", _TOKEN_CASES)
def test_tokens_out_matches_generated_sequence(worker_kind, text, expected, mode):
    if mode == "char":
        char_to_idx, idx_to_char = _tiny_char_vocab()
        tok = ModelVocabTokenizer(char_to_idx, level="char", idx_to_token=idx_to_char)
        runtime = {"hf_tokenizer": tok, "checkpoint": worker_kind}
        result = {"type": "message", "content": text}
        _tin, tout = _token_counts({"start": "a"}, result, runtime)
    elif mode == "word":
        vocab = {TOKEN_UNK: 0, TOKEN_NL: 1, "я": 2, "на": 3}
        tok = ModelVocabTokenizer(vocab, level="word")
        runtime = {"hf_tokenizer": tok, "checkpoint": worker_kind}
        result = {"type": "message", "content": text}
        _tin, tout = _token_counts({"start": "я"}, result, runtime)
    elif mode == "wordsplit":
        runtime = {"count_tokens": lambda t: max(1, len(str(t).split()))}
        result = {"type": "message", "content": text}
        _tin, tout = _token_counts({"source": "злой"}, result, runtime)
    else:
        result = {"type": "chunks", "chunks": ["aa", "bbb"]}
        _tin, tout = _token_counts({}, result, {})
    assert tout == expected


def test_pic_video_max_runtime_above_text():
    text_caps = [
        MAX_RUNTIME_SEC[k]
        for k in WORKER_KINDS
        if k.startswith("text_") or k == "mood"
    ]
    assert MAX_RUNTIME_SEC["pic_video"] > max(text_caps)


def test_gpt_sample_cancel_real_loop(monkeypatch):
    char_to_idx, idx_to_char = _tiny_char_vocab()
    model = MiniGPT(
        len(char_to_idx),
        d_model=16,
        n_layers=1,
        n_heads=2,
        block_size=16,
        dropout=0.0,
    )
    event = threading.Event()
    set_cancel_event(event)
    steps = {"n": 0}
    try:
        _set_flag_at_step(monkeypatch, event, steps, at=3)
        with pytest.raises(GenerationCancelled):
            sample_mod.evaluate_lm(
                model,
                char_to_idx,
                idx_to_char,
                start_text="a",
                prediction_len=20,
                temp=1.0,
            )
        assert steps["n"] <= 4
    finally:
        set_cancel_event(None)
