"""core/mood/model.py: токенизация, обучение на 1-2 эпохах, сохранение/загрузка."""
import pytest
import torch

from core.mood import model as mood


def test_tokenize_lower_and_yo():
    assert mood.tokenize("Ёжик ЗЛОЙ!") == ["ежик", "злой"]


def test_build_vocab_and_encode_unknown():
    vocab = mood.build_vocab(["добрый кот", "злой кот"])
    assert vocab[mood.UNK] == 0 and len(vocab) == 4
    assert mood.encode("неизвестно", vocab).tolist() == [[0]]
    assert mood.encode("", vocab).tolist() == [[0]]


def test_vector_train_axes_are_valid():
    for text, target in mood.VECTOR_TRAIN.items():
        assert len(target) == 2, text
        assert all(bit in (0, 1, None) for bit in target), text


@pytest.fixture(scope="module")
def tiny_mood():
    return mood.fit_mood(epochs=2, seed=1)


def test_predict_mood_contract(tiny_mood):
    model, vocab = tiny_mood
    kind_name, kind, fun_name, fun = mood.predict_mood(model, vocab, "злой но веселый")
    assert kind_name in ("добрый", "злой") and fun_name in ("веселый", "грустный")
    assert 0.0 <= kind <= 1.0 and 0.0 <= fun <= 1.0


def test_accuracy_in_range(tiny_mood):
    model, vocab = tiny_mood
    assert 0.0 <= mood.accuracy(model, vocab, mood.train_phrases()) <= 1.0
    assert 0.0 <= mood.axis_accuracy(model, vocab, mood.train_phrases()) <= 1.0
    assert mood.accuracy(model, vocab, {}) == 1.0


def test_save_load_roundtrip(tmp_path, tiny_mood):
    model, vocab = tiny_mood
    path = mood.save_mood(tmp_path / "mood.pt", model, vocab, train_axis_acc=0.5)
    loaded, vocab2, meta = mood.load_mood(path)
    assert vocab2 == vocab and meta["train_axis_acc"] == 0.5
    assert mood.predict_mood(loaded, vocab2, "кайф") == mood.predict_mood(model, vocab, "кайф")


def test_load_mood_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        mood.load_mood(tmp_path / "nope.pt")
    old = tmp_path / "old.pt"
    torch.save({"kind": "legacy", "vocab": {}}, old)
    with pytest.raises(ValueError):
        mood.load_mood(old)


def test_mood_source_prefers_reply():
    assert mood.mood_source(" ответ ", "текст") == "ответ"
    assert mood.mood_source("", " текст ") == "текст"
    assert mood.mood_source(None, None) == ""
