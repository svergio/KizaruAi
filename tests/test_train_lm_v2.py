"""train_lm v2: val_focus/best/patience, BPE, tie, init-from, bpc, shared mix split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from core.text.checkpoint import load_model, save_model
from core.text.lm import MiniGPT

CPU = torch.device("cpu")


def _run_main(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


def _corpus_pair(tmp_path: Path, train_lines: list[str], val_lines: list[str], focus_lines=None):
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    focus = tmp_path / "val_ru_rap.txt"
    train.write_text("\n".join(train_lines * 20) + "\n", encoding="utf-8")
    val.write_text("\n".join(val_lines * 20) + "\n", encoding="utf-8")
    focus.write_text("\n".join((focus_lines or val_lines) * 20) + "\n", encoding="utf-8")
    return train, val, focus


def _tiny_argv(train, out, *, steps=4, eval_every=1, patience=0, level="word", **extra):
    argv = [
        "--arch", "gpt",
        "--level", level,
        "--corpus", str(train),
        "--out", str(out),
        "--steps", str(steps),
        "--batch-size", "2",
        "--d-model", "16",
        "--n-layers", "1",
        "--n-heads", "2",
        "--block-size", "8",
        "--min-freq", "1",
        "--eval-every", str(eval_every),
        "--val-batches", "2",
        "--gen-len", "4",
        "--no-mlflow",
        "--vocab-size", "200",
    ]
    if patience:
        argv.extend(["--patience", str(patience)])
    for key, value in extra.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is False or value is None:
            continue
        else:
            argv.extend([flag, str(value)])
    return argv


def test_best_and_patience_on_val_focus(tmp_path, monkeypatch):
    from training import train_lm

    train, _val, _focus = _corpus_pair(
        tmp_path,
        ["я на районе ночь опять не спит и город нам кричит"],
        ["деньги в кармане сердце не болит каждый новый день"],
    )
    out = tmp_path / "m.pt"
    vals = iter([3.0, 2.0, 2.5, 2.6])

    def fake_eval(model, focus_batches, all_batches, criterion, device, arch):
        v = float(next(vals))
        return v, v + 10.0, 1.0

    monkeypatch.setattr(train_lm, "eval_val_metrics", fake_eval)
    monkeypatch.chdir(tmp_path)
    _run_main(
        train_lm,
        _tiny_argv(train, out, steps=10, patience=2),
        monkeypatch,
    )
    best = out.with_name("m_best.pt")
    last = out.with_name("m_last.pt")
    assert best.is_file() and last.is_file()
    _m, _c, _i, meta = load_model(best, device=CPU)
    assert meta["step"] == 2
    assert meta["val_focus"] == pytest.approx(2.0)
    assert last.stat().st_mtime >= best.stat().st_mtime


def test_patience_stops_after_fourth_eval(tmp_path, monkeypatch, capsys):
    from training import train_lm

    train, _val, _focus = _corpus_pair(
        tmp_path,
        ["я на районе ночь опять не спит и город нам кричит"],
        ["деньги в кармане сердце не болит каждый новый день"],
    )
    out = tmp_path / "p.pt"
    seq = [3.0, 2.0, 2.5, 2.6]

    def fake_eval(model, focus_batches, all_batches, criterion, device, arch):
        return float(seq.pop(0)), 9.0, 1.0

    monkeypatch.setattr(train_lm, "eval_val_metrics", fake_eval)
    monkeypatch.chdir(tmp_path)
    _run_main(train_lm, _tiny_argv(train, out, steps=20, patience=2), monkeypatch)
    out_txt = capsys.readouterr().out
    assert "early_stop patience=2" in out_txt
    assert "best_step=2" in out_txt
    assert load_model(out.with_name("p_best.pt"), device=CPU)[3]["step"] == 2


def test_best_uses_val_focus_not_val_all(tmp_path, monkeypatch):
    from training import train_lm

    train, _val, _focus = _corpus_pair(
        tmp_path,
        ["я на районе ночь опять не спит и город нам кричит"],
        ["деньги в кармане сердце не болит каждый новый день"],
    )
    out = tmp_path / "f.pt"
    # focus worsens after step1; all improves — best must stay at step1
    focus_seq = iter([2.0, 3.0, 4.0])
    all_seq = iter([9.0, 1.0, 0.5])

    def fake_eval(model, focus_batches, all_batches, criterion, device, arch):
        return float(next(focus_seq)), float(next(all_seq)), 1.0

    monkeypatch.setattr(train_lm, "eval_val_metrics", fake_eval)
    monkeypatch.chdir(tmp_path)
    _run_main(train_lm, _tiny_argv(train, out, steps=3, patience=0), monkeypatch)
    meta = load_model(out.with_name("f_best.pt"), device=CPU)[3]
    assert meta["step"] == 1
    assert meta["val_focus"] == pytest.approx(2.0)


def test_val_on_val_eval_mode_weights_frozen(tmp_path):
    from training import train_lm

    model = MiniGPT(vocab_size=20, d_model=16, n_layers=1, n_heads=2, block_size=8).to(CPU)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    seq = np.arange(64, dtype=np.int64) % 20
    batches = train_lm.make_fixed_batches(
        seq, seq_len=9, batch_size=2, n_batches=2, seed=0, level="char", idx_to_token={i: chr(65 + i) for i in range(20)}
    )
    model.train()
    crit = torch.nn.CrossEntropyLoss()
    v1 = train_lm.mean_ce_on_batches(model, batches, crit, CPU, "gpt")
    v2 = train_lm.mean_ce_on_batches(model, batches, crit, CPU, "gpt")
    after = model.state_dict()
    assert model.training
    assert v1 == pytest.approx(v2)
    for k in before:
        assert torch.equal(before[k], after[k])


def test_vocab_from_train_only_unk_on_val(tmp_path, monkeypatch):
    from training import train_lm
    from core.text.tokenize import text_to_word_seq, tokenize, TOKEN_UNK

    train_text = ("я на районе ночь опять не спит\n" * 40)
    val_text = ("эксклюзивноесловодлявалидации xyzzyunique\n" * 40)
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    focus = tmp_path / "val_ru_rap.txt"
    train.write_text(train_text, encoding="utf-8")
    val.write_text(val_text, encoding="utf-8")
    focus.write_text(val_text, encoding="utf-8")
    seq, token_to_idx, _ = text_to_word_seq(train_text, min_freq=1)
    assert "эксклюзивноесловодлявалидации" not in token_to_idx
    assert "xyzzyunique" not in token_to_idx
    rate = train_lm.unk_rate(val_text, token_to_idx, "word")
    assert rate > 0.0
    unk = token_to_idx[TOKEN_UNK]
    assert any(token_to_idx.get(t, unk) == unk for t in tokenize(val_text))


def test_vocab_from_byte_identical(tmp_path, monkeypatch):
    from training import train_lm
    from core.text.bpe import train_bpe, tokenizer_path_for_ckpt

    train, val, focus = _corpus_pair(
        tmp_path,
        ["я на районе ночь опять не спит и город нам кричит кайф"],
        ["деньги в кармане сердце не болит каждый новый день хит"],
    )
    src_out = tmp_path / "src.pt"
    monkeypatch.chdir(tmp_path)
    _run_main(
        train_lm,
        _tiny_argv(train, src_out, steps=1, level="bpe", vocab_size=120),
        monkeypatch,
    )
    src_tok = tokenizer_path_for_ckpt(src_out.with_name("src_best.pt"))
    assert src_tok.is_file()
    dst_out = tmp_path / "dst.pt"
    _run_main(
        train_lm,
        _tiny_argv(
            train,
            dst_out,
            steps=1,
            level="bpe",
            vocab_from=str(src_out.with_name("src_best.pt")),
        ),
        monkeypatch,
    )
    dst_tok = tokenizer_path_for_ckpt(dst_out.with_name("dst_best.pt"))
    assert src_tok.read_bytes() == dst_tok.read_bytes()


def test_init_from_weights_and_shape_error(tmp_path, monkeypatch):
    from training import train_lm

    train, _val, _focus = _corpus_pair(
        tmp_path,
        ["я на районе ночь опять не спит и город нам кричит"],
        ["деньги в кармане сердце не болит каждый новый день"],
    )
    src = tmp_path / "init_src.pt"
    monkeypatch.chdir(tmp_path)
    _run_main(train_lm, _tiny_argv(train, src, steps=1), monkeypatch)
    best = src.with_name("init_src_best.pt")
    model, c2i, i2c, meta = load_model(best, device=CPU)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    donor = torch.load(best, map_location="cpu", weights_only=False)
    m2 = MiniGPT(
        vocab_size=len(i2c),
        d_model=model.d_model,
        n_layers=model.n_layers,
        n_heads=model.n_heads,
        block_size=model.block_size,
        embedding_size=model.embedding_size,
    )
    train_lm._assert_shape_match(m2, donor["state_dict"], best)
    m2.load_state_dict(donor["state_dict"])
    for k in before:
        assert torch.equal(before[k], m2.state_dict()[k])

    bad = MiniGPT(vocab_size=len(i2c) + 5, d_model=16, n_layers=1, n_heads=2, block_size=8)
    with pytest.raises(ValueError, match="shape mismatch"):
        train_lm._assert_shape_match(bad, donor["state_dict"], best)


def test_tie_weights_same_tensor():
    model = MiniGPT(
        vocab_size=50, d_model=16, n_layers=1, n_heads=2, block_size=8, tie_weights=True
    )
    assert model.head.weight.data_ptr() == model.tok.weight.data_ptr()


def test_bpc_same_helper_word_and_bpe():
    from training import train_lm

    assert train_lm.bits_per_char(10.0, 5) == pytest.approx(2.0)
    assert train_lm.bits_per_char(10.0, 5) == train_lm.bits_per_char(10.0, 5)


def test_worker_loads_word_and_bpe(tmp_path, monkeypatch):
    from training import train_lm
    from core.text.sample import evaluate_lm_words, evaluate_lm_bpe
    from core.text.bpe import load_bpe, resolve_tokenizer_file

    train, _val, _focus = _corpus_pair(
        tmp_path,
        ["я на районе ночь опять не спит и город нам кричит кайф огонь"],
        ["деньги в кармане сердце не болит каждый новый день как хит"],
    )
    monkeypatch.chdir(tmp_path)
    word_out = tmp_path / "w.pt"
    _run_main(train_lm, _tiny_argv(train, word_out, steps=1, level="word"), monkeypatch)
    word_best = word_out.with_name("w_best.pt")
    wm, wc, wi, wmeta = load_model(word_best, device=CPU)
    assert wmeta["level"] == "word"
    text_w = evaluate_lm_words(wm, wc, wi, start_text="Я", prediction_len=4, device=CPU, max_lines=2)
    assert isinstance(text_w, str) and len(text_w) > 0

    bpe_out = tmp_path / "b.pt"
    _run_main(
        train_lm,
        _tiny_argv(train, bpe_out, steps=1, level="bpe", vocab_size=120),
        monkeypatch,
    )
    bpe_best = bpe_out.with_name("b_best.pt")
    bm, bc, bi, bmeta = load_model(bpe_best, device=CPU)
    assert bmeta["level"] == "bpe"
    tok = load_bpe(resolve_tokenizer_file(bmeta.get("tokenizer_path") or bpe_best))
    text_b = evaluate_lm_bpe(bm, tok, start_text="Я", prediction_len=4, device=CPU, max_lines=2)
    assert isinstance(text_b, str) and len(text_b) > 0


def test_shared_split_pretrain_finetune(tmp_path, monkeypatch):
    import training.build_mix as bm

    world_corpus = tmp_path / "corpus"
    world_poetry = tmp_path / "poetry"
    pools = tmp_path / "pools"
    for d in (world_corpus, world_poetry, pools):
        d.mkdir()
    pools.joinpath("00_existing.txt").write_text("rap-one\nrap-two\n", encoding="utf-8")
    pools.joinpath("en_rap.txt").write_text("en-one\n", encoding="utf-8")

    def song(text, artist="rap-one"):
        lines = text.count("\n") + 1
        return {
            "id": artist + text[:8],
            "pool": "ru_rap",
            "hash": bm.split_hash(text),
            "source": "genius",
            "artist": artist,
            "title": "t",
            "tokens": bm.count_tokens(text),
            "lines": lines,
            "lat_lines": 0,
            "script_lines": lines,
            "text": text,
            "split": bm.split_of(bm.split_hash(text), 0.2, 0.2),
        }

    texts = [f"уникальная песня номер {i} про район и ночь" * 3 for i in range(30)]
    songs = [song(t) for t in texts]
    for s in songs[:5]:
        s["pool"] = "ru_poetry"
        s["source"] = "ru_poetry_txt"
    for s in songs[5:8]:
        s["pool"] = "en_rap"
        s["artist"] = "en-one"

    table = bm.pool_table(songs)
    pre_recipe = {
        "name": "pre",
        "seed": 1,
        "total_tokens": "max",
        "splits": {"val": 0.2, "test": 0.2},
        "pools": {"ru_rap": 0.7, "ru_poetry": 0.2, "en_rap": 0.1},
        "fill_all": [],
        "max_pool_share": {},
    }
    # normalize like load_recipe
    total = sum(pre_recipe["pools"].values())
    pre_recipe["pools"] = {k: v / total for k, v in pre_recipe["pools"].items()}
    ft_recipe = {
        "name": "ft",
        "seed": 1,
        "total_tokens": "max",
        "splits": {"val": 0.2, "test": 0.2},
        "pools": {"ru_rap": 0.85, "ru_poetry": 0.10, "en_rap": 0.05},
        "fill_all": [],
        "max_pool_share": {},
    }
    total = sum(ft_recipe["pools"].values())
    ft_recipe["pools"] = {k: v / total for k, v in ft_recipe["pools"].items()}

    pre_plan = bm.plan_targets(pre_recipe, table)
    ft_plan = bm.plan_targets(ft_recipe, table)
    pre_ch = bm.sample(songs, pre_plan["targets"], 1)
    ft_ch = bm.sample(songs, ft_plan["targets"], 1)
    pre_hold = {s["hash"] for s in pre_ch["val"] + pre_ch["test"]}
    ft_train = {s["hash"] for s in ft_ch["train"]}
    ft_hold = {s["hash"] for s in ft_ch["val"] + ft_ch["test"]}
    pre_train = {s["hash"] for s in pre_ch["train"]}
    assert pre_hold.isdisjoint(ft_train)
    assert ft_hold.isdisjoint(pre_train)


def test_mlflow_gets_val_metrics(tmp_path, monkeypatch, fake_mlflow):
    from training import train_lm

    train, _val, _focus = _corpus_pair(
        tmp_path,
        ["я на районе ночь опять не спит и город нам кричит"],
        ["деньги в кармане сердце не болит каждый новый день"],
    )
    out = tmp_path / "ml.pt"
    monkeypatch.chdir(tmp_path)
    argv = _tiny_argv(train, out, steps=2)
    argv = [a for a in argv if a != "--no-mlflow"]
    argv.extend(["--mlflow-experiment", "kizaru-test"])
    _run_main(train_lm, argv, monkeypatch)
    names = fake_mlflow.metric_names()
    assert {"val_focus", "val_all", "val_unk_rate", "bpc"} <= names


def test_train_lm_writes_meta_json_sidecar(tmp_path, monkeypatch):
    from training import train_lm

    train, _val, _focus = _corpus_pair(
        tmp_path,
        ["я на районе ночь опять не спит и город нам кричит"],
        ["деньги в кармане сердце не болит каждый новый день"],
    )
    out = tmp_path / "meta_out.pt"
    monkeypatch.chdir(tmp_path)
    _run_main(train_lm, _tiny_argv(train, out, steps=2, eval_every=1), monkeypatch)
    best = out.with_name("meta_out_best.pt")
    last = out.with_name("meta_out_last.pt")
    for ckpt in (best, last):
        side = Path(str(ckpt) + ".meta.json")
        assert side.is_file(), side
        meta = json.loads(side.read_text(encoding="utf-8"))
        for key in ("name", "tokenizer", "vocab_size", "step", "val_ru_rap", "bpc", "date"):
            assert key in meta
        _m, _c, _i, ckpt_meta = load_model(ckpt, device=CPU)
        assert meta["step"] == ckpt_meta.get("step") or meta["step"] == ckpt_meta.get("epoch")
        assert meta["val_ru_rap"] == pytest.approx(float(ckpt_meta.get("val_focus")))


def test_ckpt_meta_cli_writes_sidecar(tmp_path):
    from training import ckpt_meta

    model = MiniGPT(vocab_size=32, d_model=16, n_layers=1, n_heads=2, block_size=8)
    c2i = {chr(97 + i): i for i in range(26)}
    i2c = {i: ch for ch, i in c2i.items()}
    path = tmp_path / "tiny.pt"
    save_model(
        path,
        model,
        c2i,
        i2c,
        step=7,
        val_focus=1.25,
        val_ru_rap=1.25,
        bpc=2.5,
        level="char",
    )
    meta = ckpt_meta.extract_and_write(path)
    side = ckpt_meta.sidecar_path(path)
    assert side.is_file()
    loaded = json.loads(side.read_text(encoding="utf-8"))
    assert loaded["step"] == 7
    assert loaded["val_ru_rap"] == pytest.approx(1.25)
    assert loaded["bpc"] == pytest.approx(2.5)
    assert loaded["vocab_size"] == 26
    assert loaded["name"] == "tiny.pt"
    assert meta["step"] == 7
