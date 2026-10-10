"""Ops-скрипты в var/: только smoke. Импорт по пути без запуска main(); --help только там, где есть argparse.
ВНИМАНИЕ: у большинства скриптов нет argparse, и `--help` запустил бы реальную работу
(слежение за обучением, остановка процессов), поэтому их не запускаем вообще."""
import ast
import os

import pytest

from conftest import ROOT, load_module_from_path, run_py

VAR = ROOT / "var"
OPS_SCRIPTS = sorted(p for p in VAR.glob("*.py"))
EXPECTED = {
    "deepseek_cutover_15k_rl.py",
    "deepseek_cutover_mlflow.py",
    "deepseek_rl_waiter.py",
    "mlflow_backfill_deepseek_v2.py",
    "night_train_babysit.py",
    "night_train_monitor.py",
    "pipeline_monitor.py",
}


def test_expected_ops_scripts_present():
    names = {p.name for p in OPS_SCRIPTS}
    missing = EXPECTED - names
    assert not missing, f"пропали ops-скрипты: {sorted(missing)}"


def _has_main_guard(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            left = node.test.left
            if isinstance(left, ast.Name) and left.id == "__name__":
                return True
    return False


@pytest.mark.parametrize("path", OPS_SCRIPTS, ids=[p.name for p in OPS_SCRIPTS])
def test_ops_script_has_main_guard(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert _has_main_guard(tree), f"{path.name}: нет if __name__ == '__main__' — импорт запустит работу"


@pytest.mark.parametrize("path", OPS_SCRIPTS, ids=[p.name for p in OPS_SCRIPTS])
def test_ops_script_imports_without_running(path, monkeypatch, tmp_path):
    # Некоторые скрипты при импорте делают os.chdir / os.environ.setdefault — monkeypatch всё вернёт.
    monkeypatch.chdir(tmp_path)
    for key in ("MLFLOW_TRACKING_URI", "MLFLOW_DISABLE_AGENT_HINT"):
        monkeypatch.delenv(key, raising=False)
    before = set(os.listdir(tmp_path))
    module = load_module_from_path(path, name=f"_kz_ops_{path.stem}")
    assert callable(getattr(module, "main", None)), f"{path.name}: нет main()"
    assert set(os.listdir(tmp_path)) == before


@pytest.mark.integration
def test_mlflow_backfill_help():
    proc = run_py([str(VAR / "mlflow_backfill_deepseek_v2.py"), "--help"], timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--cutover-step" in proc.stdout


def test_mlflow_backfill_parse_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in ("MLFLOW_TRACKING_URI", "MLFLOW_DISABLE_AGENT_HINT"):
        monkeypatch.delenv(key, raising=False)
    mod = load_module_from_path(VAR / "mlflow_backfill_deepseek_v2.py", name="_kz_ops_backfill_parse")
    log = tmp_path / "train.log"
    log.write_text("step=20 loss=1.5\nшум\nstep=10 loss=2.0\nstep=20 loss=9.9\n", encoding="utf-8")
    assert mod.parse_log(log) == [(10, 2.0), (20, 1.5)]
