import json
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from core.corpus.genius import cache_path, fetch_artist

MAX_WORKERS = 25
MIN_WORKERS = 2
WORKER_STEP = 1
SLUG_TRIES = 3
PRESSURE_WINDOW = 20
PRESSURE_HIT_LIMIT = 4
COOLDOWN_SEC = 3600
SCALE_UP_CLEAN = 12


def clamp_workers(workers):
    return max(MIN_WORKERS, min(MAX_WORKERS, int(workers)))


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


def _is_pressure_error(error: str) -> bool:
    text = (error or "").lower()
    if not text:
        return False
    if any(tok in text for tok in ("404", "not found", "dirty lyrics", "incomplete")):
        return False
    return any(
        tok in text
        for tok in (
            "429",
            "503",
            "timeout",
            "timed out",
            "urlerror",
            "connection",
            "reset",
            "temporarily",
            "rate limit",
            "too many",
            "503:",
            "http error 429",
            "http error 503",
        )
    )


def run_genius_pool(artists, cache_dir, workers=MAX_WORKERS):
    target = clamp_workers(workers)
    active = target
    cooldown_until = 0.0
    queue = pending_slugs(artists, cache_dir)
    failures = []
    done = []
    tries = {}
    recent = deque(maxlen=PRESSURE_WINDOW)
    print(
        f"genius pool workers_target={target} workers_active={active} "
        f"pending={len(queue)} adaptive=on step={WORKER_STEP} cooldown={COOLDOWN_SEC}s",
        flush=True,
    )
    if not queue:
        return done, failures

    inflight = {}
    with ProcessPoolExecutor(max_workers=target) as pool:
        while queue or inflight:
            now = time.time()
            if cooldown_until and now >= cooldown_until:
                cooldown_until = 0.0
                pressure = sum(1 for flag in recent if flag)
                if pressure < PRESSURE_HIT_LIMIT and active < target:
                    nxt = min(target, active + WORKER_STEP)
                    print(
                        f"  scale-up workers={active}->{nxt} "
                        f"(cooldown over, pressure={pressure}/{len(recent)})",
                        flush=True,
                    )
                    active = nxt

            while queue and len(inflight) < active:
                slug = queue.pop(0)
                inflight[pool.submit(fetch_slug_job, slug, str(cache_dir))] = slug

            if not inflight:
                break

            finished, _pending = wait(
                list(inflight), return_when=FIRST_COMPLETED, timeout=1.0
            )
            if not finished:
                continue

            for future in finished:
                slug = inflight.pop(future)
                try:
                    info = future.result()
                except Exception as exc:
                    info = {"slug": slug, "ok": False, "error": str(exc)}

                if info.get("ok"):
                    done.append(info)
                    recent.append(False)
                    print(f"  done {info['slug']} docs={info['docs']}", flush=True)
                else:
                    err = info.get("error") or ""
                    pressure = _is_pressure_error(err)
                    recent.append(pressure)
                    attempt = tries.get(slug, 0) + 1
                    tries[slug] = attempt
                    if attempt < SLUG_TRIES:
                        queue.append(slug)
                        tag = "pressure" if pressure else "retry"
                        print(
                            f"  {tag} {slug} {attempt}/{SLUG_TRIES}: {err}",
                            flush=True,
                        )
                    else:
                        failures.append(f"{slug}: {err}")
                        print(f"  fail {slug}: {err}", flush=True)

                hits = sum(1 for flag in recent if flag)
                now = time.time()
                if hits >= PRESSURE_HIT_LIMIT and active > MIN_WORKERS:
                    nxt = max(MIN_WORKERS, active - WORKER_STEP)
                    if nxt < active:
                        active = nxt
                        cooldown_until = now + COOLDOWN_SEC
                        print(
                            f"  throttle workers->{active} "
                            f"cooldown={COOLDOWN_SEC}s "
                            f"pressure={hits}/{len(recent)}",
                            flush=True,
                        )
                        # stop filling new slots until some inflight drain naturally
                elif (
                    not cooldown_until
                    and active < target
                    and len(recent) >= SCALE_UP_CLEAN
                    and hits == 0
                ):
                    nxt = min(target, active + WORKER_STEP)
                    if nxt > active:
                        print(
                            f"  scale-up workers={active}->{nxt} (clean window)",
                            flush=True,
                        )
                        active = nxt

    return done, failures
