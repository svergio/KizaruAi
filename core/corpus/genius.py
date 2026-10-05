import html as html_lib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "KizaruAi/1.0"
REQUEST_DELAY_SEC = 1.0
SONGS_PER_PAGE = 50


def _sleep():
    time.sleep(REQUEST_DELAY_SEC)


def _read_url(url):
    # Пять воркеров бьют в один IP: 429/503 не должны ронять артиста.
    delay = 2.0
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 503) or attempt == 4:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)
        except urllib.error.URLError as exc:
            last = exc
            if attempt == 4:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise last


def _request_json(url):
    return json.loads(_read_url(url))


def _request_text(url):
    return _read_url(url).decode("utf-8", "replace")


def _read_json_string(source, start):
    out = []
    i = start
    while i < len(source):
        ch = source[i]
        if ch != "\\":
            if ch == '"':
                break
            out.append(ch)
            i += 1
            continue
        if i + 1 >= len(source):
            break
        esc = source[i + 1]
        if esc == '"':
            out.append('"')
            i += 2
        elif esc == "n":
            out.append("\n")
            i += 2
        elif esc == "r":
            out.append("\r")
            i += 2
        elif esc == "t":
            out.append("\t")
            i += 2
        elif esc == "\\":
            out.append("\\")
            i += 2
        elif esc == "/":
            out.append("/")
            i += 2
        elif esc == "u" and i + 5 < len(source):
            out.append(chr(int(source[i + 2 : i + 6], 16)))
            i += 6
        else:
            out.append(esc)
            i += 2
    return "".join(out), i


def html_to_plain_lyrics(html_body):
    text = re.sub(r"<br\s*/?>", "\n", html_body, flags=re.I)
    text = re.sub(r"<a[^>]*>", "", text, flags=re.I)
    text = re.sub(r"</a>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html_lib.unescape(text).strip()


def strip_page_tail(text):
    # Строка html на Genius экранирована дважды, парсер уезжает в "children" страницы.
    if not text:
        return ""
    cut = len(text)
    for marker in ('","children"', '"children":', "window.__", "<script"):
        idx = text.find(marker)
        if 0 <= idx < cut:
            cut = idx
    text = text[:cut]
    text = text.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
    return text.strip()


def extract_lyrics_html_from_song_page(page):
    needles = (
        '\\"body\\":{\\"html\\":\\"',
        '"body":{"html":"',
    )
    for needle in needles:
        pos = page.find(needle)
        if pos < 0:
            continue
        start = pos + len(needle)
        html_body, _ = _read_json_string(page, start)
        if html_body.strip():
            return html_body
    parts = re.findall(
        r'<div[^>]+data-lyrics-container="true"[^>]*>(.*?)</div>',
        page,
        re.S,
    )
    chunks = []
    for raw in parts:
        plain = html_to_plain_lyrics(raw)
        if plain and not re.match(r"(?i)^\d+\s+contributors", plain):
            chunks.append(plain)
    if chunks:
        return "<br>\n".join(chunks)
    return ""


def fetch_song_lyrics(song_url):
    page = _request_text(song_url)
    html_body = extract_lyrics_html_from_song_page(page)
    if not html_body:
        raise ValueError("no lyrics body on page")
    return strip_page_tail(html_to_plain_lyrics(html_body))


def lyrics_looks_clean(text):
    # В текст иногда протекает JSON страницы Genius. Такое в кэш не кладём.
    if not text or not text.strip():
        return False
    if text.count("\\n") > 30 or '"children"' in text or "window.__" in text:
        return False
    low = text.lower()
    if "<script" in low or "<div" in low:
        return False
    return True


def _error_is_permanent(message):
    low = (message or "").lower()
    return "no lyrics body" in low or "404" in low


def fetch_song_text(song_url, tries=3):
    last = "empty"
    for attempt in range(tries):
        try:
            text = fetch_song_lyrics(song_url)
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            if _error_is_permanent(last):
                return None, last
        else:
            if lyrics_looks_clean(text):
                return text, None
            last = "dirty lyrics"
        if attempt + 1 < tries:
            time.sleep(min(2 * (attempt + 1), 8))
    return None, last


def _norm_key(value):
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def pick_artist_hit(hits, slug):
    # «guf» внутри «guffern» — другой артист. Берём только полное совпадение имени или slug.
    wanted = _norm_key(slug)
    for result in hits:
        keys = {_norm_key(result.get("name")), _norm_key(result.get("slug"))}
        if wanted in keys and result.get("id") is not None:
            return result
    return None


def parse_artist_page(page, slug):
    wanted = _norm_key(slug)
    for match in re.finditer(r'\\"slug\\":\\"(.*?)\\"', page):
        if _norm_key(match.group(1)) != wanted:
            continue
        window = page[match.start():match.start() + 600]
        id_match = re.search(r'\\"id\\":(\d+)', window)
        if id_match is None:
            continue
        name_match = re.search(r'\\"name\\":\\"(.*?)\\"', window)
        name = name_match.group(1) if name_match else slug
        return int(id_match.group(1)), name
    for match in re.finditer(r'"slug":"(.*?)"', page):
        if _norm_key(match.group(1)) != wanted:
            continue
        window = page[match.start():match.start() + 600]
        id_match = re.search(r'"id":(\d+)', window)
        if id_match is None:
            continue
        name_match = re.search(r'"name":"(.*?)"', window)
        name = name_match.group(1) if name_match else slug
        return int(id_match.group(1)), name
    return None


def resolve_from_artist_page(slug):
    url = "https://genius.com/artists/" + urllib.parse.quote(slug)
    try:
        page = _request_text(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    return parse_artist_page(page, slug)


def resolve_artist(slug):
    query = urllib.parse.quote(slug.replace("-", " "))
    search_url = f"https://genius.com/api/search/multi?per_page=5&q={query}"
    _sleep()
    payload = _request_json(search_url)
    hits = []
    for section in payload.get("response", {}).get("sections", []):
        for hit in section.get("hits", []):
            if hit.get("type") != "artist":
                continue
            result = hit.get("result") or {}
            if result:
                hits.append(result)
    chosen = pick_artist_hit(hits, slug)
    if chosen is not None:
        return int(chosen["id"]), chosen.get("name") or slug
    _sleep()
    found = resolve_from_artist_page(slug)
    if found is None:
        raise ValueError(f"artist not found: {slug}")
    return found


def list_artist_songs(artist_id):
    page_num = 1
    songs = []
    while True:
        url = (
            f"https://genius.com/api/artists/{artist_id}/songs"
            f"?sort=popularity&per_page={SONGS_PER_PAGE}&page={page_num}"
        )
        _sleep()
        payload = _request_json(url)
        batch = payload.get("response", {}).get("songs") or []
        if not batch:
            break
        for item in batch:
            song_url = item.get("url")
            if song_url:
                songs.append({"id": item.get("id"), "title": item.get("title"), "url": song_url})
        if len(batch) < SONGS_PER_PAGE:
            break
        page_num += 1
    return songs


def salvage_cache_file(path):
    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("complete") is not True:
        return False
    songs = []
    changed = False
    for song in payload.get("songs") or []:
        raw = song.get("text") or ""
        text = strip_page_tail(raw)
        if not lyrics_looks_clean(text):
            changed = True
            continue
        if text != raw:
            changed = True
            song = dict(song)
            song["text"] = text
        songs.append(song)
    if not changed:
        return False
    payload["songs"] = songs
    if not songs:
        payload["complete"] = False
    tmp = file_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(file_path)
    return True


def salvage_caches(cache_dir):
    saved = []
    for path in sorted(Path(cache_dir).glob("genius_*.json")):
        if salvage_cache_file(path):
            saved.append(path.stem.replace("genius_", "", 1))
    return saved


def cache_path(cache_dir, slug):
    safe = slug.replace("/", "_")
    return Path(cache_dir) / f"genius_{safe}.json"


def _write_cache(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_artist_to_cache(slug, cache_dir, token=None):
    del token
    path = cache_path(cache_dir, slug)
    _sleep()
    artist_id, resolved_name = resolve_artist(slug)
    songs_meta = list_artist_songs(artist_id)
    songs_out = []
    errors = []
    payload = {
        "slug": slug,
        "name": resolved_name,
        "artist_id": artist_id,
        "songs": songs_out,
        "errors": errors,
        "complete": False,
    }
    _write_cache(path, payload)
    total = len(songs_meta)
    for index, meta in enumerate(songs_meta, start=1):
        _sleep()
        text, err = fetch_song_text(meta["url"])
        if text:
            songs_out.append({"title": meta["title"], "url": meta["url"], "text": text})
        elif err:
            errors.append(f"{meta.get('title')}: {err}")
        if index == 1 or index % 10 == 0 or index == total:
            _write_cache(path, payload)
            print(
                f"    {slug}: {index}/{total} songs ({len(songs_out)} ok)",
                flush=True,
            )
    transient = [item for item in errors if not _error_is_permanent(item)]
    # Дырявый или грязный прогон не закрываем: пул скачает артиста ещё раз.
    payload["complete"] = bool(songs_out) and not transient
    _write_cache(path, payload)
    return payload


def load_cached_songs(cache_dir):
    merged = []
    stats = []
    for path in sorted(Path(cache_dir).glob("genius_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        slug = payload.get("slug") or path.stem.replace("genius_", "", 1)
        songs = payload.get("songs") or []
        texts = [item["text"] for item in songs if (item.get("text") or "").strip()]
        merged.extend(texts)
        stats.append((slug, len(texts), sum(len(t) for t in texts)))
    return merged, stats


def try_lyricsgenius_fetch(slug, cache_dir, token):
    try:
        import lyricsgenius
    except ImportError:
        return None
    genius = lyricsgenius.Genius(
        token,
        verbose=False,
        remove_section_headers=False,
        skip_non_songs=True,
        retries=2,
        timeout=60,
    )
    genius.excluded_terms = ["(Remix)", "(Live)"]
    artist = genius.search_artist(slug.replace("-", " "), max_songs=500, sort="popularity")
    if artist is None:
        raise ValueError(f"artist not found: {slug}")
    songs_out = []
    for song in artist.songs:
        text = (song.lyrics or "").strip()
        if text:
            songs_out.append({"title": song.title, "url": getattr(song, "url", ""), "text": text})
    payload = {
        "slug": slug,
        "name": artist.name,
        "artist_id": artist.id,
        "songs": songs_out,
        "errors": [],
        "source": "lyricsgenius",
        "complete": True,
    }
    path = cache_path(cache_dir, slug)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def fetch_artist(slug, cache_dir):
    token = os.environ.get("GENIUS_ACCESS_TOKEN", "").strip()
    if token:
        try:
            result = try_lyricsgenius_fetch(slug, cache_dir, token)
            if result is not None:
                return result
        except Exception:
            pass
    return fetch_artist_to_cache(slug, cache_dir, token=token)
