"""training: подготовка корпусов (prepare_subset, prepare_data) без скачиваний."""
import json

from training import prepare_data, prepare_subset


def test_norm_slug():
    assert prepare_subset._norm_slug("Guf-") == "guf"
    assert prepare_subset._norm_slug("Big-Baby-Tape") == "bigbabytape"


def test_genius_slug_duplicates_grouped(tmp_path):
    for name in ("genius_guf.json", "genius_guf-.json", "genius_kizaru.json"):
        (tmp_path / name).write_text(json.dumps({"songs": [{"text": "x"}]}), encoding="utf-8")
    caches = prepare_subset._index_caches(tmp_path)
    assert set(caches) == {"guf", "guf-", "kizaru"}
    families = prepare_subset._by_norm(caches)
    assert sorted(families["guf"]) == ["guf", "guf-"]
    assert families["kizaru"] == ["kizaru"]


def test_payload_weight(tmp_path):
    path = tmp_path / "genius_a.json"
    path.write_text(json.dumps({"songs": [{"text": "абв"}, {"text": " "}, {"text": "гд"}]}, ensure_ascii=False), encoding="utf-8")
    assert prepare_subset._payload_weight(path) == (2, 5)


def test_texts_from_hf_json(tmp_path):
    path = tmp_path / "hf.json"
    path.write_text(json.dumps({"train": ["  a ", "", "b"]}), encoding="utf-8")
    assert prepare_data.texts_from_hf_json(path) == ["a", "b"]
    path.write_text(json.dumps({"text": {"0": "x"}}), encoding="utf-8")
    assert prepare_data.texts_from_hf_json(path) == ["x"]


def test_load_artist_pools(tmp_path):
    (tmp_path / "01.txt").write_text("# comment\nkizaru\nguf\n", encoding="utf-8")
    (tmp_path / "02.txt").write_text("guf\nbasta\n", encoding="utf-8")
    assert prepare_data.load_artist_pools(tmp_path) == ("kizaru", "guf", "basta")
    assert prepare_data.load_artist_pools(tmp_path / "missing") == ()
