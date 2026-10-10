"""Скрипты обучения: быстрые проверки независимых функций + smoke-прогоны (slow) на игрушечных данных, CPU."""
import json
import sys

import numpy as np
import pytest
import torch

from core.text.checkpoint import load_model

CPU = torch.device("cpu")


# ------------------------------------------------------------- train_jef (без модели)


@pytest.fixture(scope="module")
def train_jef():
    from training import train_jef

    return train_jef


def test_ece_perfect_and_empty(train_jef):
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert train_jef.expected_calibration_error(probs, np.array([0, 1])) == pytest.approx(0.0)
    assert train_jef.expected_calibration_error(probs, np.array([1, 0])) == pytest.approx(1.0)
    assert np.isnan(train_jef.expected_calibration_error(np.zeros((0, 2)), np.array([], dtype=int)))


def test_load_jsonl_and_labeled_rows(train_jef, tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        '{"text": "a", "label_bin": "rhyme"}\n\n{"text": "b"}\n{"text": "c", "label_bin": "???"}\n',
        encoding="utf-8",
    )
    rows = train_jef.load_jsonl(path)
    assert len(rows) == 3
    assert [r["text"] for r in train_jef.labeled_rows(rows)] == ["a"]


def test_jef_dataset_with_fake_tokenizer(train_jef):
    def fake_tok(text, truncation, max_length, padding, return_tensors):
        ids = torch.zeros((1, max_length), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    rows = [
        {"text": "a", "label_bin": "rhyme", "label_scheme": "ABAB", "soft": True},
        {"text": "b", "label_bin": "no_rhyme"},
    ]
    ds = train_jef.JefDataset(rows, fake_tok, max_len=8)
    first, second = ds[0], ds[1]
    assert len(ds) == 2 and first["input_ids"].shape == (8,)
    assert int(first["labels"]) == 1 and int(first["has_scheme"]) == 1
    assert int(first["scheme"]) == train_jef.SCHEME2ID["ABAB"]
    assert float(first["weight"]) == pytest.approx(train_jef.JefDataset.soft_weight)
    assert int(second["labels"]) == 0 and int(second["has_scheme"]) == 0 and float(second["weight"]) == 1.0


def test_write_results_logs_to_mlflow(train_jef, tmp_path, fake_mlflow, capsys):
    from training import mlflow_track

    mlflow_track.start_run("kizaru-jef")
    train_jef.write_results(tmp_path / "out", {"test_f1": 0.9, "test_ece": 0.05, "test_acc": 0.88, "gate": "pass"})
    data = json.loads((tmp_path / "out" / "RESULTS.json").read_text(encoding="utf-8"))
    assert data["gate"] == "pass"
    assert {"test_f1", "test_ece", "test_acc"} <= fake_mlflow.metric_names()
    assert fake_mlflow.tags["gate"] == "pass"


def test_write_results_without_mlflow(train_jef, tmp_path, capsys):
    train_jef.write_results(tmp_path, {"test_f1": 0.1, "test_ece": 0.5, "test_acc": 0.2, "gate": "fail"})
    assert (tmp_path / "RESULTS.json").is_file()


# -------------------------------------------------------- train_rl_deepseek (без модели)


@pytest.fixture(scope="module")
def train_rl():
    from training import train_rl_deepseek

    return train_rl_deepseek


def test_hard_gate(train_rl):
    good = "Я на районе ночью\nДеньги в кармане лежат\nГород не спит опять\nМы снова тут стоим"
    assert train_rl.hard_gate(good, max_lines=6) == 1.0
    assert train_rl.hard_gate("одна строка", max_lines=6) == 0.15
    assert train_rl.hard_gate("\n".join(["повтор строки тут"] * 4), max_lines=6) == 0.35
    assert train_rl.hard_gate("\n".join(f"строка номер {i}" for i in range(20)), max_lines=6) == 0.4
    assert train_rl.hard_gate("аа\nбб\nвв", max_lines=6) == 0.5


def test_load_seed_pool(train_rl, tmp_path):
    assert train_rl.load_seed_pool(tmp_path / "missing.txt", 40)
    corpus = tmp_path / "c.txt"
    corpus.write_text("первая строка\nвторая строка\n\nтретья\n", encoding="utf-8")
    pool = train_rl.load_seed_pool(corpus, max_chars=40, limit=10)
    assert pool[0] == "первая строка\nвторая строка"
    assert all(len(p) <= 40 for p in pool)


def test_write_status_atomic(train_rl, tmp_path):
    path = tmp_path / "var" / "status.json"
    train_rl.write_status(path, {"step": 1, "текст": "ок"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"step": 1, "текст": "ок"}
    assert not path.with_suffix(".json.tmp").exists()


# ----------------------------------------------------------------- smoke (slow)


def _run_main(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


def _write_val_siblings(corpus_path):
    text = corpus_path.read_text(encoding="utf-8")
    corpus_path.with_name("val.txt").write_text(text, encoding="utf-8")
    corpus_path.with_name("val_ru_rap.txt").write_text(text, encoding="utf-8")


def _tiny_lm_argv(arch, corpus, out, steps="2", **extra):
    argv = [
        "--arch", arch, "--corpus", str(corpus), "--out", str(out), "--steps", str(steps),
        "--batch-size", "2", "--d-model", "16", "--n-layers", "1", "--n-heads", "2",
        "--block-size", "16", "--eval-every", "1", "--gen-len", "8", "--val-batches", "2",
        "--no-mlflow",
    ]
    for key, value in extra.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is False or value is None:
            continue
        else:
            argv.extend([flag, str(value)])
    return argv


@pytest.mark.slow
@pytest.mark.parametrize("arch", ["gpt", "transformer", "deepseek", "bert_gen"])
def test_train_lm_smoke(arch, tiny_corpus, tmp_path, monkeypatch, capsys):
    from training import train_lm

    monkeypatch.chdir(tmp_path)
    _write_val_siblings(tiny_corpus)
    out = tmp_path / "ckpt" / f"{arch}.pt"
    _run_main(train_lm, _tiny_lm_argv(arch, tiny_corpus, out), monkeypatch)
    best = out.with_name(out.stem + "_best" + out.suffix)
    model, c2i, _i2c, meta = load_model(best, device=CPU)
    assert meta["arch"] == arch and meta["epoch"] == 2 and meta["level"] == "char"
    assert "done, best=" in capsys.readouterr().out


@pytest.mark.slow
def test_train_lm_word_level(tiny_corpus, tmp_path, monkeypatch, capsys):
    from training import train_lm

    monkeypatch.chdir(tmp_path)
    _write_val_siblings(tiny_corpus)
    out = tmp_path / "gpt_word.pt"
    _run_main(
        train_lm,
        _tiny_lm_argv("gpt", tiny_corpus, out, level="word", block_size=8, min_freq=1, gen_len=5),
        monkeypatch,
    )
    best = out.with_name(out.stem + "_best" + out.suffix)
    assert load_model(best, device=CPU)[3]["level"] == "word"


@pytest.mark.slow
def test_train_lm_logs_metrics_to_mlflow(tiny_corpus, tmp_path, monkeypatch, fake_mlflow, capsys):
    from training import train_lm

    monkeypatch.chdir(tmp_path)
    _write_val_siblings(tiny_corpus)
    out = tmp_path / "g.pt"
    argv = _tiny_lm_argv("gpt", tiny_corpus, out, steps="3", gen_len=5)
    argv = [a for a in argv if a != "--no-mlflow"]
    argv.extend(["--mlflow-experiment", "kizaru-test"])
    _run_main(train_lm, argv, monkeypatch)
    assert fake_mlflow.experiments == ["kizaru-test"]
    assert {"loss", "val_focus", "val_all", "val_unk_rate", "bpc"} <= fake_mlflow.metric_names()
    assert fake_mlflow.params["arch"] == "gpt"
    assert fake_mlflow.ended == ["FINISHED"]
    assert "mlflow_run=fake-run-1" in capsys.readouterr().out


@pytest.mark.slow
def test_train_lm_survives_mlflow_down(tiny_corpus, tmp_path, monkeypatch, broken_mlflow, capsys):
    from training import train_lm

    monkeypatch.chdir(tmp_path)
    _write_val_siblings(tiny_corpus)
    out = tmp_path / "g.pt"
    argv = [a for a in _tiny_lm_argv("gpt", tiny_corpus, out, gen_len=5) if a != "--no-mlflow"]
    _run_main(train_lm, argv, monkeypatch)
    assert out.with_name("g_best.pt").is_file()
    assert "mlflow_warn start_run failed" in capsys.readouterr().out


@pytest.mark.slow
def test_train_lm_rejects_navec_for_char(tiny_corpus, tmp_path, monkeypatch):
    from training import train_lm

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        _run_main(train_lm, ["--arch", "gpt", "--corpus", str(tiny_corpus), "--init-navec", "--no-mlflow"], monkeypatch)


@pytest.mark.slow
def test_train_rnn_smoke(tiny_corpus, tmp_path, monkeypatch, capsys):
    from training import train

    monkeypatch.chdir(tmp_path)
    out = tmp_path / "kizaru.pt"
    _run_main(
        train,
        ["--corpus", str(tiny_corpus), "--out", str(out), "--epochs", "2", "--seq-len", "16", "--batch-size", "2",
         "--hidden-size", "16", "--embedding-size", "8", "--n-layers", "1", "--bptt", "8", "--eval-every", "1",
         "--gen-len", "10"],
        monkeypatch,
    )
    meta = load_model(out, device=CPU)[3]
    assert meta["arch"] == "rnn" and meta["epoch"] == 2


@pytest.mark.slow
def test_train_mood_smoke(tmp_path, monkeypatch, fake_mlflow, capsys):
    from core.mood.model import load_mood
    from training import train_mood

    out = tmp_path / "mood.pt"
    _run_main(train_mood, ["--out", str(out), "--epochs", "1"], monkeypatch)
    _model, _vocab, meta = load_mood(out)
    assert 0.0 <= meta["train_axis_acc"] <= 1.0
    assert {"train_phrase_acc", "train_axis_acc"} <= fake_mlflow.metric_names()
    assert fake_mlflow.ended == ["FINISHED"]
