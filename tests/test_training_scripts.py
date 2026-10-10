"""training/scripts: build_jef_dataset, import_verse_schema, download_poetry на крошечных локальных данных.
Скачивание подменяется, модели с HF не грузятся."""
import io
import json
import random
import sys
import types

import pytest

from conftest import ROOT, load_module_from_path

SCRIPTS = ROOT / "training" / "scripts"


@pytest.fixture(scope="module")
def jef():
    return load_module_from_path(SCRIPTS / "build_jef_dataset.py")


@pytest.fixture(scope="module")
def verse():
    return load_module_from_path(SCRIPTS / "import_verse_schema.py")


@pytest.fixture(scope="module")
def poetry():
    return load_module_from_path(SCRIPTS / "download_poetry.py")


QUATRAIN = "Я иду домой\nНочь стоит стеной\nГород за спиной\nСнова я с тобой"


def _make_sources(tmp_path):
    rifma = tmp_path / "rifma.json"
    rifma.write_text(
        json.dumps(
            [
                {"poem_text": QUATRAIN, "rhyme_scheme": "a b a b"},
                {"poem_text": "Строка одна\nСтрока вторая\nТретья тут\nЧетвёртая", "rhyme_scheme": "abba"},
                {"poem_text": "одна строка", "rhyme_scheme": "a"},
                {"poem_text": "Белый стих\nбез рифмы\nсовсем", "rhyme_scheme": "- - -"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    en = tmp_path / "en_raw"
    en.mkdir()
    (en / "poet.txt").write_text(
        "AUTHOR Poet\nTITLE Song\nRHYME a a b b\nthe cat\nthe hat\nthe dog\nthe log\n",
        encoding="utf-8",
    )
    genius = tmp_path / "genius"
    genius.mkdir()
    (genius / "genius_kizaru.json").write_text(
        json.dumps({"songs": [{"text": "[Куплет]\nДеньги\nНочь\nКайф\n[Припев]\nОгонь\nДым"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    ars = tmp_path / "ars.json"
    ars.write_text(
        json.dumps([{"poem_text": "Первая\nВторая\n\nТретья\nЧетвёртая", "author": "A", "title": "T"}], ensure_ascii=False),
        encoding="utf-8",
    )
    return rifma, en, genius, ars


# ------------------------------------------------------------- build_jef_dataset


@pytest.mark.parametrize(
    "raw, expected",
    [("a b a b", "ABAB"), ("abba", "ABBA"), ("a a b b", "AABB"), ("aaa", "AABB"), (None, "free"),
     ("- -", "free"), ("", "free"), ("a b c d", "other"), ("b a b a", "ABAB")],
)
def test_norm_scheme(jef, raw, expected):
    assert jef._norm_scheme(raw) == expected


def test_row_rules(jef):
    assert jef._row("одна", "abab", "ru", "x") is None
    long = jef._row("\n".join(f"l{i}" for i in range(20)), "abab", "ru", "x")
    assert long["lines"] == 16 and long["label_bin"] == "rhyme"
    free = jef._row("а\nб", None, "ru", "x")
    assert free["label_scheme"] == "free" and free["label_bin"] == "no_rhyme"
    unlabeled = jef._row("а\nб", None, "ru", "x", require_scheme=False)
    assert "label_bin" not in unlabeled


def test_make_negatives(jef):
    pos = [jef._row(QUATRAIN, "abab", "ru", "rifma"), jef._row("а\nб", "- -", "ru", "rifma")]
    neg = jef.make_negatives(pos, random.Random(0), per_pos=1.0)
    assert len(neg) == 1
    assert neg[0]["label_bin"] == "no_rhyme" and neg[0]["source"] == "rifma_shuffle"
    assert sorted(neg[0]["text"].split("\n")) == sorted(QUATRAIN.split("\n"))
    assert neg[0]["text"] != QUATRAIN


def test_split_rows(jef):
    rows = [{"i": i} for i in range(20)]
    train, valid, test = jef.split_rows(rows, 0.1, 0.1, seed=8)
    assert (len(train), len(valid), len(test)) == (16, 2, 2)
    assert sorted(r["i"] for r in train + valid + test) == list(range(20))
    assert jef.split_rows(rows, 0.1, 0.1, seed=8) == (train, valid, test)
    small = jef.split_rows([{"i": 0}, {"i": 1}, {"i": 2}], 0.1, 0.1, seed=1)
    assert sum(map(len, small)) == 3


def test_genius_stanza_chunks(jef):
    text = "[Куплет]\nа\nб\n[Припев]\nв\nг\nд\n[Outro]\nодна"
    assert jef._genius_stanza_chunks(text) == [["а", "б"], ["в", "г", "д"]]


def test_build_jef_main_end_to_end(jef, tmp_path, monkeypatch, capsys):
    rifma, en, genius, ars = _make_sources(tmp_path)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "jef"
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_jef_dataset", "--rifma", str(rifma), "--ars", str(ars), "--en-raw", str(en),
         "--genius-dir", str(genius), "--out-dir", str(out), "--neg-ratio", "1.0"],
    )
    jef.main()
    rows = []
    for name in ("train", "valid", "test"):
        path = out / f"{name}.jsonl"
        assert path.is_file()
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    sources = {r["source"] for r in rows}
    assert {"rifma", "chicago", "arspoetica", "genius_slang"} <= sources
    assert all("label_bin" not in r for r in rows if r["source"] in ("arspoetica", "genius_slang"))
    assert any(r["source"].endswith("_shuffle") for r in rows)
    notes = tmp_path / "var" / "jef_data_notes.md"
    assert notes.is_file() and "not requested" in notes.read_text(encoding="utf-8")


# ----------------------------------------------------------- import_verse_schema


def test_normalize_rhyme(verse):
    assert verse._normalize_rhyme("a b a b") == "ABAB"
    assert verse._normalize_rhyme("a a *") == "AA*"
    assert verse._normalize_rhyme("--") == "free"
    assert verse._normalize_rhyme(None) is None


@pytest.mark.parametrize("n, form", [(2, "couplet"), (4, "quatrain"), (6, "couplet"), (12, "freestyle")])
def test_guess_form(verse, n, form):
    assert verse._guess_form(n, None) == form


def test_import_verse_main(verse, tmp_path, monkeypatch, capsys):
    rifma, en, _genius, ars = _make_sources(tmp_path)
    out_jsonl = tmp_path / "schema" / "verse.jsonl"
    out_txt = tmp_path / "schema" / "verse_lm.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        ["import_verse_schema", "--rifma", str(rifma), "--ars", str(ars), "--en-raw", str(en),
         "--en-form", str(tmp_path / "missing"), "--out-jsonl", str(out_jsonl), "--out-txt", str(out_txt)],
    )
    verse.main()
    recs = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
    by_src = {}
    for r in recs:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    assert by_src == {"rifma": 3, "arspoetica": 2, "chicago_rhyme": 1}
    assert next(r for r in recs if r["source"] == "chicago_rhyme")["rhyme"] == "AABB"
    assert out_txt.read_text(encoding="utf-8").count("\n\n") == len(recs)


def test_import_verse_skip_flags(verse, tmp_path, monkeypatch, capsys):
    rifma, en, _genius, ars = _make_sources(tmp_path)
    out_jsonl = tmp_path / "v.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        ["import_verse_schema", "--rifma", str(rifma), "--ars", str(ars), "--en-raw", str(en),
         "--out-jsonl", str(out_jsonl), "--out-txt", str(tmp_path / "v.txt"), "--skip-ars-nc", "--skip-en"],
    )
    verse.main()
    recs = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
    assert {r["source"] for r in recs} == {"rifma"}


# ---------------------------------------------------------------- download_poetry

XML = (
    "<?xml version='1.0' encoding='utf-8'?><corpus>"
    "<item><text>Строка 1\n  Строка 2  \n\n</text></item>"
    "<item><text></text></item>"
    "<item><author>x</author></item>"
    "<item><text>Ещё стих</text></item></corpus>"
)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_poems_from_xml_and_write_corpus(poetry, tmp_path):
    xml = tmp_path / "all.xml"
    xml.write_text(XML, encoding="utf-8")
    poems = list(poetry.poems_from_xml(xml))
    assert len(poems) == 2
    n, chars = poetry.write_corpus(poems, tmp_path / "out" / "poetry.txt")
    assert n == 2
    assert (tmp_path / "out" / "poetry.txt").read_text(encoding="utf-8") == "Строка 1\nСтрока 2\nЕщё стих\n"


def test_download_is_mocked(poetry, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(poetry.urllib.request, "urlopen", lambda req, timeout=None: _Resp(XML.encode("utf-8")))
    dest = poetry.download("https://example.invalid/all.xml", tmp_path / "raw" / "all.xml")
    assert dest.read_text(encoding="utf-8") == XML
    assert "WARN sha mismatch" in capsys.readouterr().out


def test_try_hf_stihi_sample_with_fake_datasets(poetry, tmp_path, monkeypatch, capsys):
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: iter([{"text": "Стих\n\nвторой"}, {"text": ""}, {"text": "ещё"}])
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setattr(poetry, "OUT_DIR", tmp_path)
    poetry.try_hf_stihi_sample(limit=10)
    assert (tmp_path / "stihi_ru_sample.txt").read_text(encoding="utf-8") == "Стих\nвторой\nещё\n"


def test_download_poetry_main_offline(poetry, tmp_path, monkeypatch, capsys):
    raw = tmp_path / "raw" / "poetry_corpus.xml"

    def fake_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(XML, encoding="utf-8")
        return dest

    monkeypatch.setattr(poetry, "RAW_XML", raw)
    monkeypatch.setattr(poetry, "OUT_TXT", tmp_path / "classical.txt")
    monkeypatch.setattr(poetry, "download", fake_download)
    monkeypatch.setattr(poetry, "try_hf_stihi_sample", lambda limit: None)
    assert poetry.main() == 0
    assert (tmp_path / "classical.txt").read_text(encoding="utf-8").startswith("Строка 1")
