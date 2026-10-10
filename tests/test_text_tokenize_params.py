"""core/text/tokenize.py, params.py, pretrained_vocab.py."""
import json

import pytest

from core.text import params
from core.text.pretrained_vocab import (
    filter_known,
    load_whitelist_file,
    load_wordlevel_vocab,
    merge_whitelist,
    seed_tokens,
)
from core.text.tokenize import (
    TOKEN_NL,
    TOKEN_UNK,
    build_word_vocab,
    detokenize,
    encode_words,
    text_to_word_seq,
    tokenize,
)


def test_tokenize_and_detokenize_roundtrip():
    tokens = tokenize("Я иду,\nдомой!")
    assert tokens == ["я", "иду", ",", TOKEN_NL, "домой", "!"]
    assert detokenize(tokens) == "я иду,\nдомой!"


def test_tokenize_handles_crlf_and_empty():
    assert tokenize("а\r\nб") == ["а", TOKEN_NL, "б"]
    assert tokenize("") == []


def test_build_word_vocab_min_freq_and_whitelist():
    t2i, i2t = build_word_vocab("кайф кайф ночь", min_freq=2, whitelist=["Огонь"])
    assert list(t2i)[:2] == [TOKEN_UNK, TOKEN_NL]
    assert "кайф" in t2i and "ночь" not in t2i and "огонь" in t2i
    assert all(i2t[i] == t for t, i in t2i.items())


def test_text_to_word_seq_and_encode():
    seq, t2i, _ = text_to_word_seq("а б а б", min_freq=1)
    assert len(seq) == 4
    assert encode_words("неизвестное", t2i) == [t2i[TOKEN_UNK]]
    assert encode_words("", t2i) == [t2i[TOKEN_NL]]
    with pytest.raises(ValueError):
        text_to_word_seq("", min_freq=1)


@pytest.mark.parametrize("model", ["lstm", "lstm-word", "transformer", "gpt", "deepseek", "bert-gen"])
def test_resolve_text_params_defaults(model):
    assert params.resolve_text_params(model) == params.DEFAULTS[model]


def test_resolve_text_params_clamps():
    out = params.resolve_text_params("gpt", {"temp": 99, "lines": 0, "iters": 10_000, "length": 99_999})
    assert out == {"temp": params.TEMP_MAX, "lines": params.LINES_MIN, "iters": params.ITERS_MAX, "length": params.CHAR_LENGTH_MAX}
    word = params.resolve_text_params("lstm-word", {"length": 1})
    assert word["length"] == params.WORD_LENGTH_MIN


def test_resolve_text_params_unknown_model():
    with pytest.raises(ValueError):
        params.resolve_text_params("gpt-5")


def test_collect_overrides_and_describe():
    assert params.collect_overrides(temp=0.5, lines=None) == {"temp": 0.5}
    d = params.describe("bert-gen", params.resolve_text_params("bert-gen"))
    assert "iters=" in d and "seq_len=" in d
    assert "tokens" in params.describe("lstm-word", params.resolve_text_params("lstm-word"))


def test_whitelist_helpers(tmp_path):
    wl = tmp_path / "wl.txt"
    wl.write_text("# комментарий\nКайф\nкайф\n<unk>\nночь\n", encoding="utf-8")
    words = load_whitelist_file(wl)
    assert words == ["кайф", "ночь"]
    merged, idx = merge_whitelist({TOKEN_UNK: 0, TOKEN_NL: 1, "ночь": 2}, words)
    assert merged["кайф"] == 3 and idx[3] == "кайф"
    assert seed_tokens(["Я и ты", "я"]) == ["я", "и", "ты"]
    assert filter_known(["а", "б"], {"а": 1}) == ["а"]


def test_load_wordlevel_vocab(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"vocab": {"а": 0, "б": 1}}}), encoding="utf-8")
    assert load_wordlevel_vocab(path) == {"а": 0, "б": 1}
    path.write_text(json.dumps({"model": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_wordlevel_vocab(path)
