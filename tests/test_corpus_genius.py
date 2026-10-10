"""core/corpus/genius.py и genius_pool.py: парсинг страниц, кэш, ретраи. Сеть подменяется."""
import io
import json
import urllib.error

import pytest

from core.corpus import genius, genius_pool
from core.corpus.genius import (
    cache_path,
    extract_lyrics_html_from_song_page,
    html_to_plain_lyrics,
    load_cached_songs,
    lyrics_looks_clean,
    parse_artist_page,
    pick_artist_hit,
    salvage_cache_file,
    strip_page_tail,
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(genius.time, "sleep", lambda *_: None)
    monkeypatch.setattr(genius, "_sleep", lambda: None)


def test_html_to_plain_lyrics():
    html = "Я<br>иду &amp; <a href='x'>домой</a><i>!</i>"
    assert html_to_plain_lyrics(html) == "Я\nиду & домой!"


def test_extract_lyrics_from_container_div():
    page = '<html><div class="x" data-lyrics-container="true">Строка 1<br/>Строка 2</div></html>'
    assert "Строка 1" in html_to_plain_lyrics(extract_lyrics_html_from_song_page(page))


def test_extract_lyrics_from_embedded_json():
    page = 'xx "body":{"html":"<p>Привет\\nмир<\\/p>"} yy'
    assert extract_lyrics_html_from_song_page(page) == "<p>Привет\nмир</p>"


def test_extract_lyrics_empty_page():
    assert extract_lyrics_html_from_song_page("<html></html>") == ""


def test_strip_page_tail():
    assert strip_page_tail('Текст песни","children":[1,2]') == "Текст песни"
    assert strip_page_tail("") == ""


def test_lyrics_looks_clean():
    assert lyrics_looks_clean("нормальный текст")
    assert not lyrics_looks_clean("")
    assert not lyrics_looks_clean('x "children" y')
    assert not lyrics_looks_clean("<div>что-то</div>")
    assert not lyrics_looks_clean("\\n" * 31)


def test_pick_artist_hit_requires_exact_name():
    hits = [
        {"name": "Guffern", "slug": "guffern", "id": 1},
        {"name": "Guf", "slug": "Guf", "id": 2},
    ]
    assert pick_artist_hit(hits, "guf")["id"] == 2
    assert pick_artist_hit(hits[:1], "guf") is None


def test_parse_artist_page():
    page = '{"slug":"kizaru","id":123,"name":"Kizaru"}'
    assert parse_artist_page(page, "kizaru") == (123, "Kizaru")
    assert parse_artist_page(page, "other") is None


def test_fetch_song_text_permanent_error_no_retry(monkeypatch):
    calls = []

    def boom(url):
        calls.append(url)
        raise ValueError("no lyrics body on page")

    monkeypatch.setattr(genius, "fetch_song_lyrics", boom)
    text, err = genius.fetch_song_text("https://genius.invalid/x", tries=3)
    assert text is None and "no lyrics body" in err
    assert len(calls) == 1


def test_fetch_song_text_dirty_then_clean(monkeypatch):
    answers = iter(['<div>грязь</div>', "чистый текст"])
    monkeypatch.setattr(genius, "fetch_song_lyrics", lambda url: next(answers))
    assert genius.fetch_song_text("u", tries=3) == ("чистый текст", None)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_read_url_retries_on_429(monkeypatch):
    state = {"n": 0}

    def fake_urlopen(req, timeout=None):
        state["n"] += 1
        if state["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many", None, None)
        return _Resp(b'{"ok": true}')

    monkeypatch.setattr(genius.urllib.request, "urlopen", fake_urlopen)
    assert genius._request_json("https://genius.invalid/api") == {"ok": True}
    assert state["n"] == 3


def test_read_url_404_not_retried(monkeypatch):
    state = {"n": 0}

    def fake_urlopen(req, timeout=None):
        state["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(genius.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        genius._read_url("https://genius.invalid/x")
    assert state["n"] == 1


def test_cache_path_and_load_cached(tmp_path):
    assert cache_path(tmp_path, "a/b").name == "genius_a_b.json"
    cache_path(tmp_path, "kizaru").write_text(
        json.dumps({"slug": "kizaru", "songs": [{"text": "раз"}, {"text": ""}, {"text": "два"}]}),
        encoding="utf-8",
    )
    texts, stats = load_cached_songs(tmp_path)
    assert texts == ["раз", "два"]
    assert stats == [("kizaru", 2, 6)]


def test_salvage_cache_file_drops_dirty_and_trims_tail(tmp_path):
    path = tmp_path / "genius_x.json"
    path.write_text(
        json.dumps(
            {
                "complete": True,
                "songs": [
                    {"text": 'Хороший текст","children":[1]'},
                    {"text": "<script>bad</script>"},
                    {"text": "ок"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert salvage_cache_file(path) is True
    data = json.loads(path.read_text(encoding="utf-8"))
    assert [s["text"] for s in data["songs"]] == ["Хороший текст", "ок"]
    assert salvage_cache_file(path) is False


def test_fetch_artist_to_cache_mocked(tmp_path, monkeypatch):
    monkeypatch.setattr(genius, "resolve_artist", lambda slug: (7, "Kizaru"))
    monkeypatch.setattr(
        genius,
        "list_artist_songs",
        lambda aid: [{"id": 1, "title": "A", "url": "u1"}, {"id": 2, "title": "B", "url": "u2"}],
    )
    monkeypatch.setattr(
        genius,
        "fetch_song_text",
        lambda url: ("текст " + url, None) if url == "u1" else (None, "404 not found"),
    )
    payload = genius.fetch_artist_to_cache("kizaru", tmp_path)
    assert payload["complete"] is True  # 404 — постоянная ошибка, повтор не нужен
    saved = json.loads(cache_path(tmp_path, "kizaru").read_text(encoding="utf-8"))
    assert saved["songs"] == [{"title": "A", "url": "u1", "text": "текст u1"}]


# ------------------------------------------------------------------ genius_pool


def test_clamp_workers_bounds():
    assert genius_pool.MIN_WORKERS == 2
    assert genius_pool.clamp_workers(1) == 2  # --workers 1 -> минимум 2
    assert genius_pool.clamp_workers(3) == 3  # --workers 3 остаётся 3 (не 5)
    assert genius_pool.clamp_workers(10_000) == genius_pool.MAX_WORKERS
    mid = (genius_pool.MIN_WORKERS + genius_pool.MAX_WORKERS) // 2
    assert genius_pool.clamp_workers(mid) == mid


def _cache(dir_, slug, **payload):
    cache_path(dir_, slug).write_text(json.dumps({"slug": slug, **payload}, ensure_ascii=False), encoding="utf-8")


def test_cache_is_complete_and_pending(tmp_path):
    _cache(tmp_path, "done", complete=True, songs=[{"text": "ок"}])
    _cache(tmp_path, "partial", complete=False, songs=[])
    _cache(tmp_path, "dirty", complete=True, songs=[{"text": "<div>x</div>"}])
    assert genius_pool.cache_is_complete(cache_path(tmp_path, "done"))
    assert not genius_pool.cache_is_complete(cache_path(tmp_path, "partial"))
    assert not genius_pool.cache_is_complete(cache_path(tmp_path, "dirty"))
    assert not genius_pool.cache_is_complete(tmp_path / "missing.json")
    assert genius_pool.pending_slugs(["done", "partial", "dirty", "new"], tmp_path) == ["partial", "dirty", "new"]


def test_adopt_legacy_caches(tmp_path):
    _cache(tmp_path, "legacy", songs=[{"text": "текст"}])
    _cache(tmp_path, "empty", songs=[{"text": ""}])
    _cache(tmp_path, "part", songs=[{"text": "x"}])
    stamped, dropped = genius_pool.adopt_legacy_caches(tmp_path, partial_slugs=["part"])
    assert stamped == ["legacy"]
    assert sorted(dropped) == ["empty", "part"]
    assert json.loads(cache_path(tmp_path, "legacy").read_text(encoding="utf-8"))["complete"] is True
    assert not cache_path(tmp_path, "empty").exists()
