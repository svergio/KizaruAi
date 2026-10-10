"""core/corpus/pack.py + training/pack_lyrics_dataset.py: упаковка кэшей в JSONL-датасет."""
import json
import sys
from pathlib import Path

import pytest

from core.corpus.pack import (
    assign_splits,
    pack_corpus,
    row_has_section_tag_line,
    slug_from_hf_path,
)


def test_slug_from_hf_path():
    assert slug_from_hf_path(Path("huggingartists_big_baby_tape.json")) == "big-baby-tape"
    with pytest.raises(ValueError):
        slug_from_hf_path(Path("genius_kizaru.json"))


def test_row_has_section_tag_line():
    assert row_has_section_tag_line("[Припев]\nтекст")
    assert row_has_section_tag_line("строка\nEmbed")
    assert not row_has_section_tag_line("просто строка\nи ещё одна")


def test_assign_splits_deterministic():
    rows = [{"id": str(i)} for i in range(10)]
    a = assign_splits(rows, train_ratio=0.8, seed=1)
    b = assign_splits(rows, train_ratio=0.8, seed=1)
    assert [r["split"] for r in a] == [r["split"] for r in b]
    assert sum(r["split"] == "train" for r in a) == 8
    assert sum(r["split"] == "valid" for r in a) == 2


def test_assign_splits_edge_cases():
    assert assign_splits([]) == []
    one = assign_splits([{"id": "x"}])
    assert one[0]["split"] == "train"
    two = assign_splits([{"id": "a"}, {"id": "b"}], train_ratio=0.98)
    assert sorted(r["split"] for r in two) == ["train", "valid"]


def _write_caches(corpus: Path):
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "huggingartists_kizaru.json").write_text(
        json.dumps({"train": ["Я на районе\nНочь не спит", "", "Деньги\nКайф"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (corpus / "genius_big-baby-tape.json").write_text(
        json.dumps(
            {
                "slug": "big-baby-tape",
                "name": "Big Baby Tape",
                "songs": [
                    {"title": "Gimme", "text": "[Куплет 1]\nДеньги текут\n[Припев]\nНочь горит"},
                    {"title": "Empty", "text": ""},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_pack_corpus_end_to_end(tmp_path):
    corpus = tmp_path / "corpus"
    out = tmp_path / "dataset"
    _write_caches(corpus)
    main_path, manifest = pack_corpus(corpus, out, train_ratio=0.5, seed=1)
    assert main_path == out / "lyrics.jsonl"
    rows = [json.loads(line) for line in main_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3 == manifest["rows_total"]
    assert manifest["train"] + manifest["valid"] == 3
    assert manifest["by_source"] == {"huggingartists": 2, "genius": 1}
    for name in ("train.jsonl", "valid.jsonl", "manifest.json", "splits/train_ids.txt"):
        assert (out / name).is_file()
    # Ни одна строка датасета не должна содержать тегов секций [Куплет]/[Припев].
    for row in rows:
        assert not row_has_section_tag_line(row["text"]), row


def test_pack_corpus_empty_dir_raises(tmp_path):
    (tmp_path / "corpus").mkdir()
    with pytest.raises(ValueError):
        pack_corpus(tmp_path / "corpus", tmp_path / "out")


def test_pack_lyrics_dataset_cli_empty_exits_1(tmp_path, monkeypatch, capsys):
    from training import pack_lyrics_dataset

    (tmp_path / "corpus").mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["pack_lyrics_dataset", "--corpus-dir", str(tmp_path / "corpus"), "--out-dir", str(tmp_path / "o")],
    )
    with pytest.raises(SystemExit) as exc:
        pack_lyrics_dataset.main()
    assert exc.value.code == 1


def test_pack_lyrics_dataset_cli_ok(tmp_path, monkeypatch, capsys):
    from training import pack_lyrics_dataset

    _write_caches(tmp_path / "corpus")
    monkeypatch.setattr(
        sys,
        "argv",
        ["pack_lyrics_dataset", "--corpus-dir", str(tmp_path / "corpus"), "--out-dir", str(tmp_path / "o")],
    )
    pack_lyrics_dataset.main()
    assert "rows=3" in capsys.readouterr().out
