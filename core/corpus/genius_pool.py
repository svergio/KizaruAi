import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from core.corpus.genius import cache_path, fetch_artist

MAX_WORKERS = 5
SLUG_TRIES = 3


def clamp_workers(workers):
    return max(1, min(MAX_WORKERS, int(workers)))


def cache_is_complete(path):
    from core.corpus.genius import lyrics_looks_clean

    file_path = Path(path)
    if not file_path.exists():
        return False
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("complete") is not True:
        return False
    songs = payload.get("songs") or []
    # complete=true со старого прогона, но внутри мусор страницы — качаем заново
    for song in songs:
        text = song.get("text") or ""
        if text.strip() and not lyrics_looks_clean(text):
            return False
    return True


def pending_slugs(artists, cache_dir):
    waiting = []
    for slug in artists:
        if not cache_is_complete(cache_path(cache_dir, slug)):
            waiting.append(slug)
    return waiting


def adopt_legacy_caches(cache_dir, partial_slugs=()):
    # Старый прогон не ставил complete. Готовый файл помечаем, оборванный удаляем.
    partial = set(partial_slugs)
    cache = Path(cache_dir)
    stamped = []
    dropped = []
    for path in sorted(cache.glob("genius_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        slug = payload.get("slug") or path.stem.replace("genius_", "", 1)
        if slug in partial or payload.get("complete") is False:
            if slug in partial:
                path.unlink()
                dropped.append(slug)
            continue
        if payload.get("complete") is True:
            continue
        songs = payload.get("songs") or []
        if not any((item.get("text") or "").strip() for item in songs):
            path.unlink()
            dropped.append(slug)
            continue
        payload["complete"] = True
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        stamped.append(slug)
    return stamped, dropped


def fetch_slug_job(slug, cache_dir):
    try:
        payload = fetch_artist(slug, str(cache_dir))
    except Exception as exc:
        return {"slug": slug, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if payload.get("complete") is not True:
        errs = payload.get("errors") or []
        tail = "; ".join(errs[:3])
        return {"slug": slug, "ok": False, "error": tail or "incomplete"}
    songs = payload.get("songs") or []
    docs = len([item for item in songs if (item.get("text") or "").strip()])
    return {"slug": slug, "ok": True, "docs": docs}


def run_genius_pool(artists, cache_dir, workers=MAX_WORKERS):
    width = clamp_workers(workers)
    queue = pending_slugs(artists, cache_dir)
    failures = []
    done = []
    tries = {}
    print(f"genius pool workers={width} pending={len(queue)}", flush=True)
    if not queue:
        return done, failures
    inflight = {}
    with ProcessPoolExecutor(max_workers=width) as pool:
        while queue or inflight:
            while queue and len(inflight) < width:
                slug = queue.pop(0)
                inflight[pool.submit(fetch_slug_job, slug, str(cache_dir))] = slug
            if not inflight:
                break
            future = next(as_completed(list(inflight)))
            slug = inflight.pop(future)
            try:
                info = future.result()
            except Exception as exc:
                info = {"slug": slug, "ok": False, "error": str(exc)}
            if info.get("ok"):
                done.append(info)
                print(f"  done {info['slug']} docs={info['docs']}", flush=True)
                continue
            attempt = tries.get(slug, 0) + 1
            tries[slug] = attempt
            if attempt < SLUG_TRIES:
                queue.append(slug)
                print(f"  retry {slug} {attempt}/{SLUG_TRIES}: {info.get('error')}", flush=True)
                continue
            failures.append(f"{slug}: {info.get('error')}")
            print(f"  fail {slug}: {info.get('error')}", flush=True)
    return done, failures
