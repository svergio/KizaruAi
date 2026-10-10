"""core/corpus/lyrics.py: теги секций Genius, нормализация символов, корпус -> последовательность."""
import numpy as np
import pytest
import torch

from core.corpus.lyrics import (
    clean_lyrics,
    get_batch,
    load_corpus,
    prepare_corpus_text,
    strip_genius_annotations,
    text_to_seq,
)


def test_strip_genius_section_tags():
    raw = "[Куплет 1: Kizaru]\nЯ на районе\n[Припев]\nДеньги не спят\n(Intro) Ночь"
    assert strip_genius_annotations(raw) == "Я на районе\nДеньги не спят\nНочь"


def test_strip_genius_keeps_one_blank_between_blocks():
    raw = "строка\n\n\n\nещё строка"
    assert strip_genius_annotations(raw) == "строка\n\nещё строка"


def test_strip_genius_drops_ui_noise():
    raw = "5 Contributors\nСтрока\nYou might also like\n12 Embed"
    assert strip_genius_annotations(raw) == "Строка"


def test_clean_lyrics_normalizes_spaces_and_newlines():
    assert clean_lyrics("Я\u00a0иду  домой\n\n\nНочь\u200b") == "Я иду домой\nНочь\n"
    assert clean_lyrics("") == "\n"


def test_prepare_corpus_text_genius_mode():
    raw = "[Припев]\nКайф и огонь\nEmbed\nProduced by X\nНочь не спит"
    out = prepare_corpus_text(raw, genius=True)
    assert out == "Кайф и огонь\nНочь не спит\n"
    assert prepare_corpus_text("   ", genius=True) == ""


def test_load_corpus_txt(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("  строка   один \n\n строка два\n", encoding="utf-8")
    assert load_corpus(path) == "строка один\nстрока два\n"


def test_load_corpus_empty_raises(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_corpus(path)


def test_load_corpus_csv(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("text\nпривет\n\nпока\n", encoding="utf-8")
    assert load_corpus(path) == "привет\n\nпока"


def test_text_to_seq_roundtrip():
    seq, c2i, i2c = text_to_seq("абба")
    assert seq.dtype == np.int64
    assert "".join(i2c[int(i)] for i in seq) == "абба"
    assert c2i["а"] in (0, 1) and len(c2i) == 2


def test_get_batch_shapes_and_shift():
    seq = np.arange(100, dtype=np.int64)
    np.random.seed(0)
    x, y = get_batch(seq, seq_len=10, batch_size=3)
    assert x.shape == (3, 9, 1) and y.shape == (3, 9, 1)
    assert x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_get_batch_too_short_raises():
    with pytest.raises(ValueError):
        get_batch(np.arange(5), seq_len=10, batch_size=1)
