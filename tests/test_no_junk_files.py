"""Запрет временных/одноразовых файлов в var/, training/, scripts/ (см. docs/CURSOR_MISTAKES.md)."""
from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECK_DIRS = ("var", "training", "scripts", "app")
JUNK_PATTERNS = ("_*.py", "*.disabled*", "*_bak*", "*_backup*", "*.py.bak", "*.ps1.bak", "*_once.py")
ALLOWED_NAMES = {"__init__.py", "__main__.py"}
ALLOWED_PATTERNS = ("*.log.bak",)  # логи можно
MESSAGE = ("Временные файлы запрещены (docs/CURSOR_MISTAKES.md). Удали файл, не меняй этот тест. "
           "Повторяемая проверка -> флаг в существующем CLI или тест.")


def is_junk(name: str) -> bool:
    low = name.lower()
    if name in ALLOWED_NAMES or any(fnmatchcase(low, p) for p in ALLOWED_PATTERNS):
        return False
    return any(fnmatchcase(low, p) for p in JUNK_PATTERNS)


def find_junk(root: Path, dirs=CHECK_DIRS) -> list[str]:
    found = []
    for d in dirs:
        base = root / d
        if not base.is_dir():
            continue
        for fp in base.rglob("*"):
            if "__pycache__" in fp.parts or not fp.is_file():
                continue
            if is_junk(fp.name):
                found.append(fp.relative_to(root).as_posix())
    return sorted(found)


@pytest.mark.parametrize("name", [
    "_resolve_slugs.py", "_probe.py", "build_mix.py.DISABLED", "x.DISABLED.py", "train_BAK.py",
    "notes_BAK", "data_backup.json", "cfg_Backup.yaml", "build_mix.py.bak", "run.ps1.bak",
    "fix_once.py", "BUILD.PY.BAK",
])
def test_is_junk_positive(name):
    assert is_junk(name)


@pytest.mark.parametrize("name", [
    "__init__.py", "__main__.py", "deepseek_v2_a3_train.log.bak", "train.LOG.BAK", "build_mix.py",
    "check_dataset.py", "once.py", "backup.md", "run.ps1", "notes.txt", "data.bak",
])
def test_is_junk_negative(name):
    assert not is_junk(name)


def test_find_junk_on_tmp_tree(tmp_path):
    (tmp_path / "var" / "sub").mkdir(parents=True)
    (tmp_path / "training" / "__pycache__").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "app").mkdir()
    (tmp_path / "other").mkdir()
    for rel in ("var/sub/_tmp.py", "var/run.log.bak", "training/__init__.py", "training/__pycache__/_x.py",
                "training/ok.py", "scripts/a.ps1.bak", "app/_junk.py", "other/_skip.py"):
        (tmp_path / rel).write_text("", encoding="utf-8")
    assert find_junk(tmp_path) == ["app/_junk.py", "scripts/a.ps1.bak", "var/sub/_tmp.py"]


def test_no_junk_files_in_repo():
    junk = find_junk(ROOT)
    assert not junk, MESSAGE + "\n" + "\n".join(f"  {p}" for p in junk)


def test_no_node_modules_or_package_json():
    bad = []
    for p in ROOT.rglob("package.json"):
        if any(part.startswith(".") for part in p.parts):
            continue
        if "node_modules" in p.parts:
            continue
        bad.append(p.relative_to(ROOT).as_posix())
    for p in ROOT.rglob("node_modules"):
        if p.is_dir() and not any(part.startswith(".") for part in p.parts):
            bad.append(p.relative_to(ROOT).as_posix() + "/")
    assert not bad, "node_modules/ и package.json запрещены:\n" + "\n".join(bad)


def test_db_files_only_under_data():
    bad = []
    for pattern in ("*.db", "*.db-shm", "*.db-wal"):
        for p in ROOT.rglob(pattern):
            if any(part.startswith(".") or part == "__pycache__" for part in p.parts):
                continue
            rel = p.relative_to(ROOT)
            if rel.parts[:1] != ("data",):
                bad.append(rel.as_posix())
    assert not bad, "*.db* только в data/:\n" + "\n".join(bad)


def test_app_web_no_bare_init_main():
    web = ROOT / "app" / "web"
    assert not (web / "init.py").exists()
    assert not (web / "main.py").exists()
    assert (web / "__init__.py").is_file()
    assert (web / "__main__.py").is_file()
