"""Общие хелперы списков артистов (training/artist_pools/): ключ slug и исключения из пула ru_rap.

Без зависимостей от torch и от jef_challenge — можно импортировать из build_mix и тестов.

    from training.pools import pool_key, load_ru_rap_exclude, load_ru_rap_exclude_pools
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOLS_DIR = ROOT / "training" / "artist_pools"
RU_RAP_EXCLUDE_FILE = POOLS_DIR / "ru_rap_exclude.txt"
DEFAULT_EXCLUDE_POOL = "ru_other"

_NORM_RE = re.compile(r"[\W_]+", re.U)


def pool_key(slug: str) -> str:
    """Ключ slug из списков пулов: без регистра, дефисов и пунктуации.

    Отрезается только однозначный суффикс дубля — ровно одна цифра после последнего дефиса:
    Kizaru-2 -> kizaru, 25-17-2 -> 2517, I61-2 -> i61. Числа в имени не трогаются:
    25-17 -> 2517, ALBLAK-52 -> alblak52, 47-ak-47 -> 47ak47.
    """
    slug = (slug or "").strip().rstrip("-")
    slug = re.sub(r"-\d$", "", slug)
    return _NORM_RE.sub("", slug.lower())


def load_ru_rap_exclude_pools(path: Path = RU_RAP_EXCLUDE_FILE) -> dict[str, str]:
    """pool_key -> пул из ru_rap_exclude.txt (формат `slug<TAB>pool`).

    Пустые строки и комментарии (# ..., в том числе в конце строки) пропускаются.
    Строка без колонки pool (или с пустой) даёт DEFAULT_EXCLUDE_POOL ('ru_other'). Нет файла — {}.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        slug, _, pool = line.partition("\t")
        k = pool_key(slug)
        if k:
            out[k] = pool.strip() or DEFAULT_EXCLUDE_POOL
    return out


def load_ru_rap_exclude(path: Path = RU_RAP_EXCLUDE_FILE) -> frozenset[str]:
    """Ключи pool_key артистов, исключённых из ru_rap (колонка pool игнорируется)."""
    return frozenset(load_ru_rap_exclude_pools(path))
