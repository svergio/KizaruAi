"""Никаких путей под одну машину: C:\\Users\\..., E:\\Pet_Projects\\... в коде проекта.
Обычный тест (без xfail): пока пути не вычищены, он честно падает и показывает, где они."""
import re

import pytest

from conftest import ROOT

PATTERN = re.compile(r"(?i)\b[a-z]:[\\/]+(?:users|pet_projects)(?:[\\/]|\b)")
SCAN_DIRS = ("core", "app", "training", "scripts", "var")
SKIP_PARTS = {"tmp", "__pycache__"}


def _files():
    out = []
    for name in SCAN_DIRS:
        base = ROOT / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() not in (".py", ".ps1"):
                continue
            rel = path.relative_to(ROOT)
            if SKIP_PARTS & set(rel.parts[:-1]):
                continue
            out.append(path)
    start_all = ROOT / "start_all.ps1"
    if start_all.is_file():
        out.append(start_all)
    return out


FILES = _files()


def test_scan_found_files():
    assert len(FILES) > 20


@pytest.mark.parametrize("path", FILES, ids=[str(p.relative_to(ROOT)).replace("\\", "/") for p in FILES])
def test_no_machine_specific_paths(path):
    hits = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if PATTERN.search(line):
            hits.append(f"{lineno}: {line.strip()}")
    assert not hits, "захардкоженные пути:\n" + "\n".join(hits)
