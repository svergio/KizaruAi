"""core/text: архитектуры, сохранение/загрузка чекпоинтов, генерация. Крошечные модели на CPU."""
import numpy as np
import pytest
import torch

from core.text.bert_gen import BertMLM, apply_mlm_mask, generate_mlm
from core.text.checkpoint import (
    ARCH_BERT_GEN,
    ARCH_DEEPSEEK,
    ARCH_GPT,
    ARCH_RNN,
    ARCH_TRANSFORMER,
    cuda_for_job,
    get_device,
    load_model,
    save_model,
)
from core.text.deepseek import MiniDeepSeek
from core.text.lm import MiniGPT, SimpleTransformer
from core.text.rnn import TextRNN
from core.text.sample import _sample_index, evaluate, evaluate_lm, evaluate_lm_words, evaluate_words

CHARS = list("абвгд \n")
C2I = {c: i for i, c in enumerate(CHARS)}
I2C = {i: c for c, i in C2I.items()}
V = len(CHARS)
CPU = torch.device("cpu")


def _build(arch):
    torch.manual_seed(0)
    kw = dict(d_model=16, n_layers=1, n_heads=2, block_size=16)
    if arch == ARCH_RNN:
        return TextRNN(V, hidden_size=8, embedding_size=8, n_layers=1)
    if arch == ARCH_TRANSFORMER:
        return SimpleTransformer(V, **kw)
    if arch == ARCH_GPT:
        return MiniGPT(V, **kw)
    if arch == ARCH_DEEPSEEK:
        return MiniDeepSeek(V, **kw)
    if arch == ARCH_BERT_GEN:
        return BertMLM(V + 1, mask_id=V, **kw)
    raise AssertionError(arch)


def test_get_device_is_cpu_in_tests():
    assert get_device().type == "cpu"


@pytest.mark.parametrize("arch", [ARCH_TRANSFORMER, ARCH_GPT, ARCH_DEEPSEEK, ARCH_BERT_GEN])
def test_lm_forward_shape(arch):
    model = _build(arch)
    out = model(torch.zeros((2, 10), dtype=torch.long))
    assert out.shape == (2, 10, model.head.out_features)
    with pytest.raises(ValueError):
        model(torch.zeros(10, dtype=torch.long))


def test_rnn_step_shapes():
    model = _build(ARCH_RNN)
    hidden = model.init_hidden(1)
    out, hidden = model.step(torch.LongTensor([1]), hidden)
    assert out.shape == (1, 1, V)
    with pytest.raises(ValueError):
        model.step(torch.zeros((1, 1, 1, 1), dtype=torch.long), hidden)


@pytest.mark.parametrize("arch", [ARCH_RNN, ARCH_TRANSFORMER, ARCH_GPT, ARCH_DEEPSEEK, ARCH_BERT_GEN])
def test_save_load_roundtrip(tmp_path, arch):
    model = _build(arch)
    path = save_model(tmp_path / f"{arch}.pt", model, C2I, I2C, epoch=3, mean_loss=1.5, level="char")
    loaded, c2i, i2c, meta = load_model(path, device=CPU)
    assert meta["arch"] == arch and meta["epoch"] == 3 and meta["mean_loss"] == 1.5
    assert c2i == C2I and i2c == I2C
    a, b = model.state_dict(), loaded.state_dict()
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key].cpu(), b[key].cpu()), key


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.pt", device=CPU)


def test_load_model_unsupported_payload(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"hello": 1}, path)
    with pytest.raises(ValueError):
        load_model(path, device=CPU)


def test_cuda_for_job_cpu_is_noop():
    model = _build(ARCH_GPT)
    with cuda_for_job(model, CPU):
        pass
    assert next(model.parameters()).device.type == "cpu"


def test_evaluate_rnn_generates_text():
    np.random.seed(0)
    torch.manual_seed(0)
    text = evaluate(_build(ARCH_RNN), C2I, I2C, start_text="аб", prediction_len=20, temp=0.5, device=CPU)
    assert isinstance(text, str) and text.startswith("аб")
    assert set(text) <= set(CHARS)


def test_evaluate_lm_generates_text():
    np.random.seed(0)
    text = evaluate_lm(_build(ARCH_GPT), C2I, I2C, start_text="в", prediction_len=30, temp=1.0, device=CPU)
    assert text.startswith("в") and set(text) <= set(CHARS)


def test_evaluate_lm_max_lines_stops():
    np.random.seed(1)
    text = evaluate_lm(_build(ARCH_GPT), C2I, I2C, start_text="а", prediction_len=300, temp=1.0, device=CPU, max_lines=1)
    non_empty_closed = [ln for ln in text.split("\n")[:-1] if ln.strip()]
    assert len(non_empty_closed) <= 1


def test_word_level_generation():
    from core.text.tokenize import TOKEN_NL, TOKEN_UNK

    t2i = {TOKEN_UNK: 0, TOKEN_NL: 1, "я": 2, "иду": 3, "домой": 4}
    i2t = {i: t for t, i in t2i.items()}
    np.random.seed(0)
    torch.manual_seed(0)
    rnn = TextRNN(len(t2i), hidden_size=8, embedding_size=8, n_layers=1)
    assert isinstance(evaluate_words(rnn, t2i, i2t, start_text="я", prediction_len=10, device=CPU), str)
    gpt = MiniGPT(len(t2i), d_model=16, n_layers=1, n_heads=2, block_size=8)
    out = evaluate_lm_words(gpt, t2i, i2t, start_text="я", prediction_len=10, device=CPU)
    assert TOKEN_UNK not in out


def test_generate_mlm_char_level():
    torch.manual_seed(0)
    model = _build(ARCH_BERT_GEN)
    text = generate_mlm(model, C2I, I2C, start_text="аб", seq_len=12, n_iters=4, temp=1.0, device=CPU)
    assert isinstance(text, str) and text.startswith("аб")


def test_apply_mlm_mask():
    torch.manual_seed(0)
    tokens = torch.randint(0, V, (2, 16))
    inputs, labels = apply_mlm_mask(tokens, mask_id=V, vocab_size=V + 1, mlm_prob=0.5)
    chosen = labels != -100
    assert chosen.any()
    assert torch.equal(labels[chosen], tokens[chosen])
    assert torch.equal(inputs[~chosen], tokens[~chosen])


def test_sample_index_respects_block_ids():
    np.random.seed(0)
    probs = np.array([0.5, 0.5, 0.0])
    for _ in range(20):
        assert _sample_index(probs, 3, block_ids=[0]) == 1
