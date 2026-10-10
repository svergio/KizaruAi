"""training.check_dataset: проверки noise / brackets / rhyme / dupes на крошечных корпусах в tmp_path."""
import hashlib
import json
from pathlib import Path

import pytest

cd = pytest.importorskip("training.check_dataset", reason="training.check_dataset ещё не собран")

from conftest import run_py  # noqa: E402


# ------------------------------------------------------------------ фикстуры

def _song_file(path: Path, texts, key="text", wrap=True):
    songs = [{key: t, "title": f"s{i}"} for i, t in enumerate(texts)]
    data = {"songs": songs} if wrap else songs
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


NOISY_SONG = (
    "12 Contributors\n"
    "[Intro]\n"
    "я еду по району\n"
    "[Chorus: Kizaru]\n"
    "деньги на кармане\n"
    "You might also like\n"
    "[Produced by Somebody]\n"
)
CLEAN_SONG = "просто строка без мусора\nи вторая строка рядом\n"


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    _song_file(d / "genius_noisy.json", [NOISY_SONG, CLEAN_SONG])
    _song_file(d / "genius_guf-.json", [CLEAN_SONG])                 # дубль с дефисом, меньше песен
    _song_file(d / "genius_guf.json", [CLEAN_SONG, NOISY_SONG])      # нормальный
    _song_file(d / "genius_Big-Baby-Tape.json", [CLEAN_SONG], wrap=False)
    _song_file(d / "genius_big_baby_tape-.json", [CLEAN_SONG], wrap=False)  # тот же slug, тот же размер
    _song_file(d / "genius_solo-.json", [CLEAN_SONG])                 # только с дефисом
    (d / "genius_broken.json").write_text("{not json", encoding="utf-8")
    (d / "other.json").write_text("[]", encoding="utf-8")            # не genius_* — не учитывается
    return d


RHYMED = "\n".join(["i am making", "you are taking", "we are baking", "they are shaking",
                    "she is waking", "he is raking", "it is quaking", "all are faking"]) + "\n"
BROKEN = "\n".join(["i am making", "you are flowing", "we can stand", "they will fly",
                    "she is mute", "he got cold", "it is red", "all are glum"]) + "\n"

LYRICS = (
    "[Chorus]\n"
    "строка раз\n"
    "[Chorus]\n"
    "[Verse 1]\n"
    '{"apiPath": "/songs/1", "lyricsState": "complete"}\n'
    + "а" * 301 + "\n"
    "\n"
    "обычная строка\n"
)


@pytest.fixture
def lyrics(tmp_path):
    p = tmp_path / "lyrics.txt"
    p.write_text(LYRICS + RHYMED, encoding="utf-8")
    return p


def _hash_tree(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir"
            for p in sorted(root.rglob("*"))}


# ------------------------------------------------------------------ helpers

def test_norm_slug_matches_prepare_subset():
    assert cd.norm_slug("guf-") == "guf"
    assert cd.norm_slug("Big-Baby-Tape") == cd.norm_slug("big_baby_tape-") == "bigbabytape"
    ps = pytest.importorskip("training.prepare_subset")
    if hasattr(ps, "_norm_slug"):
        for s in ("guf-", "Big-Baby-Tape", "kizaru_2", "a--b-"):
            assert cd.norm_slug(s) == ps._norm_slug(s)


def test_song_texts_formats(tmp_path):
    a = _song_file(tmp_path / "a.json", ["x", "y"])
    b = _song_file(tmp_path / "b.json", ["z"], key="lyrics", wrap=False)
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert cd.song_texts(a) == ["x", "y"]
    assert cd.song_texts(b) == ["z"]
    assert cd.song_texts(bad) == []
    assert cd.song_texts(tmp_path / "missing.json") == []


@pytest.mark.parametrize("word,zone", [
    ("making", "aking"), ("кот", "от"), ("машина", "ина"), ("", None), (None, None), ("брр", None),
])
def test_rhyme_zone(word, zone):
    assert cd.rhyme_zone(word) == zone


# ------------------------------------------------------------------ noise

def test_noise_counts_raw_and_after_with_fake_prepare(corpus):
    calls = []

    def fake_prepare(text):
        calls.append(text)
        return "" if "Contributors" in text else text

    r = cd.check_noise(corpus, sample=0, seed=1, prepare=fake_prepare)
    # genius_* файлов 7 (other.json не в счёт), битый даёт 0 песен
    assert r["files"] == 7
    assert r["songs"] == 8 and len(calls) == 8
    assert r["dropped"] == 2                       # NOISY_SONG дважды
    assert r["raw"]["contributors"] == 2 and r["after"]["contributors"] == 0
    assert r["raw"]["section_tag"] == 2
    assert r["raw"]["genius_ui"] == 2
    assert r["raw"]["credits"] == 2
    assert r["raw"]["any_bracket"] == 2
    assert r["raw"]["json_leak"] == 0


def test_noise_with_real_prepare_removes_tags(corpus):
    r = cd.check_noise(corpus, sample=0, seed=1)
    assert r["songs"] == 8
    for key in cd.NOISE_PATTERNS:
        assert r["after"].get(key, 0) <= r["raw"].get(key, 0), key
    assert r["after"].get("section_tag", 0) == 0
    assert r["after"].get("contributors", 0) == 0


def test_noise_skips_empty_bodies(tmp_path):
    _song_file(tmp_path / "genius_e.json", ["", "   \n", CLEAN_SONG])
    r = cd.check_noise(tmp_path, sample=0, seed=0, prepare=lambda t: t)
    assert r["songs"] == 1 and r["dropped"] == 0


def test_noise_sample_and_seed_determinism(tmp_path, monkeypatch):
    for i in range(10):
        _song_file(tmp_path / f"genius_a{i}.json", [CLEAN_SONG])
    seen = []
    real = cd.song_texts
    monkeypatch.setattr(cd, "song_texts", lambda p: (seen.append(p.name), real(p))[1])

    def picked(sample, seed):
        seen.clear()
        r = cd.check_noise(tmp_path, sample=sample, seed=seed, prepare=lambda t: t)
        return r["files"], tuple(seen)

    n1, a = picked(3, 7)
    n2, b = picked(3, 7)
    assert n1 == n2 == 3 and a == b and len(set(a)) == 3
    others = {picked(3, s)[1] for s in range(20)}
    assert len(others) > 1                         # другой seed — другая выборка
    n_all, all_files = picked(0, 7)
    assert n_all == 10 and len(all_files) == 10   # 0 = все
    n_big, _ = picked(100, 7)
    assert n_big == 10                             # sample больше числа файлов


# ------------------------------------------------------------------ brackets

def test_brackets():
    r = cd.check_brackets(LYRICS)
    assert r["lines"] == 7                         # пустая строка не считается
    assert r["json_lines"] == 1
    assert r["long_lines"] == 1
    assert r["bracket_total"] == 3
    assert r["top_tags"][0] == ("chorus", 2)
    assert ("verse 1", 1) in r["top_tags"]


def test_brackets_top_limit_and_empty():
    text = "\n".join(f"[t{i}]" for i in range(30))
    assert len(cd.check_brackets(text, top=5)["top_tags"]) == 5
    r = cd.check_brackets("")
    assert r == {"lines": 0, "json_lines": 0, "long_lines": 0, "bracket_total": 0, "top_tags": []}


# ------------------------------------------------------------------ rhyme

def test_rhyme_detects_rhymes_vs_broken():
    good = cd.check_rhyme(RHYMED, seed=0)
    bad = cd.check_rhyme(BROKEN, seed=0)
    assert good["words"] == bad["words"] == 8
    assert good["by_offset"][1] == 100.0
    assert good["rhyme_in_next3"] == 100.0
    assert bad["by_offset"][1] == 0.0
    assert bad["rhyme_in_next3"] == 0.0
    assert set(good["by_offset"]) == {1, 2, 3, 4}


def test_rhyme_drops_junk_lines_and_is_seed_deterministic():
    junk = '{"apiPath": "x"} making\n' + "b" * 301 + " taking\n"
    r = cd.check_rhyme(junk + BROKEN, seed=3)
    assert r["words"] == 8                         # JSON- и длинная строка выкинуты
    mixed = RHYMED + BROKEN
    assert cd.check_rhyme(mixed, seed=5) == cd.check_rhyme(mixed, seed=5)


def test_rhyme_same_word_is_not_rhyme():
    r = cd.check_rhyme("a making\nb making\nc making\nd making\n")
    assert r["by_offset"][1] == 0.0 and r["rhyme_in_next3"] == 0.0


def test_rhyme_empty():
    r = cd.check_rhyme("")
    assert r["words"] == 0 and r["baseline"] == 0.0 and r["rhyme_in_next3"] == 0.0


# ------------------------------------------------------------------ dupes

def test_dupes(corpus):
    before = _hash_tree(corpus)
    r = cd.check_dupes(corpus)
    by_norm = {d["norm"]: d for d in r["dupes"]}
    assert set(by_norm) == {"guf", "bigbabytape"}
    guf = by_norm["guf"]
    assert {n for n, _, _ in guf["files"]} == {"genius_guf-.json", "genius_guf.json"}
    assert guf["keep"] == "genius_guf.json"        # больше песен
    assert dict((n, s) for n, s, _ in guf["files"]) == {"genius_guf-.json": 1, "genius_guf.json": 2}
    bbt = by_norm["bigbabytape"]
    assert bbt["keep"] == "genius_Big-Baby-Tape.json"  # при равенстве — без дефиса
    assert r["dash_only"] == ["genius_solo-.json"]
    # noisy, guf, bigbabytape, solo, broken
    assert r["groups"] == 5
    assert _hash_tree(corpus) == before            # ничего не удалено


def test_dupes_bigger_dash_file_wins(tmp_path):
    _song_file(tmp_path / "genius_x-.json", [CLEAN_SONG, CLEAN_SONG, CLEAN_SONG])
    _song_file(tmp_path / "genius_x.json", [CLEAN_SONG])
    assert cd.check_dupes(tmp_path)["dupes"][0]["keep"] == "genius_x-.json"


def test_dupes_empty_dir(tmp_path):
    assert cd.check_dupes(tmp_path) == {"groups": 0, "dupes": [], "dash_only": []}


def test_norm_slug_keeps_cyrillic():
    assert cd.norm_slug("Баста") == cd.norm_slug("баста-2") == cd.norm_slug("Баста-") == "баста"
    assert cd.norm_slug("Каста") == "каста" != cd.norm_slug("Баста")
    assert cd.norm_slug("25-17") == "2517"       # не суффикс дубля
    assert cd.norm_slug("Кровосток") != ""


def test_dupes_cyrillic_slugs(tmp_path):
    _song_file(tmp_path / "genius_Баста.json", [CLEAN_SONG, NOISY_SONG])
    _song_file(tmp_path / "genius_баста-2.json", [CLEAN_SONG])
    _song_file(tmp_path / "genius_Каста.json", [CLEAN_SONG])
    _song_file(tmp_path / "genius_Кровосток.json", [CLEAN_SONG])
    _song_file(tmp_path / "genius_Скриптонит-.json", [CLEAN_SONG])
    r = cd.check_dupes(tmp_path)
    by_norm = {d["norm"]: d for d in r["dupes"]}
    assert set(by_norm) == {"баста"}                       # Каста/Кровосток не склеены с Бастой и друг с другом
    assert sorted(f[0] for f in by_norm["баста"]["files"]) == ["genius_Баста.json", "genius_баста-2.json"]
    assert by_norm["баста"]["keep"] == "genius_Баста.json"  # больше песен
    assert r["groups"] == 4                                 # баста, каста, кровосток, скриптонит
    assert "" not in by_norm
    assert r["dash_only"] == ["genius_Скриптонит-.json"]


# ------------------------------------------------------------------ render / CLI

def test_render_sections(corpus, lyrics):
    text = lyrics.read_text(encoding="utf-8")
    md = cd.render({
        "noise": cd.check_noise(corpus, 0, 1, prepare=lambda t: t),
        "brackets": cd.check_brackets(text),
        "rhyme": cd.check_rhyme(text),
        "dupes": cd.check_dupes(corpus),
    })
    assert md.startswith("# ")
    assert md.count("\n## ") == 4
    assert "`[chorus]`: 2" in md
    assert "genius_guf.json" in md
    for key in cd.NOISE_PATTERNS:
        assert f"| {key} |" in md
    assert cd.render({}).startswith("# ")


def _main(capsys, *argv):
    code = cd.main(list(argv))
    return code, capsys.readouterr()


def test_main_all_writes_report(corpus, lyrics, tmp_path, capsys):
    out = tmp_path / "reports" / "sub" / "report.md"
    code, cap = _main(capsys, "all", "--corpus-dir", str(corpus), "--lyrics", str(lyrics),
                      "--sample", "0", "--out", str(out))
    assert code == 0
    assert out.is_file()
    md = out.read_text(encoding="utf-8")
    assert md.count("\n## ") == len(cd.CHECKS)
    assert md.strip() == cap.out.strip()
    assert "genius_guf.json" in md


def test_main_single_checks_and_dedup(corpus, lyrics, capsys):
    code, cap = _main(capsys, "dupes", "dupes", "--corpus-dir", str(corpus), "--lyrics", str(lyrics))
    assert code == 0 and cap.out.count("\n## ") == 1
    code, cap = _main(capsys, "brackets", "rhyme", "--lyrics", str(lyrics), "--corpus-dir", str(corpus))
    assert code == 0 and cap.out.count("\n## ") == 2


def test_main_missing_lyrics_returns_2(corpus, tmp_path, capsys):
    out = tmp_path / "r.md"
    code, cap = _main(capsys, "rhyme", "--lyrics", str(tmp_path / "nope.txt"),
                      "--corpus-dir", str(corpus), "--out", str(out))
    assert code == 2
    assert "nope.txt" in cap.err
    assert not out.exists()


def test_main_noise_seed_determinism(tmp_path, capsys):
    for i in range(8):
        _song_file(tmp_path / f"genius_s{i}.json", [NOISY_SONG] * (i + 1))
    args = ["noise", "--corpus-dir", str(tmp_path), "--sample", "3"]
    outs = [_main(capsys, *args, "--seed", "11")[1].out for _ in range(2)]
    assert outs[0] == outs[1]
    variants = {_main(capsys, *args, "--seed", str(s))[1].out for s in range(10)}
    assert len(variants) > 1


def test_main_dry_run_changes_nothing(corpus, lyrics, tmp_path, capsys):
    out = tmp_path / "reports" / "report.md"
    before = _hash_tree(tmp_path)
    code, cap = _main(capsys, "all", "--corpus-dir", str(corpus), "--lyrics", str(lyrics),
                      "--sample", "0", "--out", str(out), "--dry-run")
    assert code == 0
    assert "\n## " in cap.out                      # отчёт всё равно в stdout
    assert not out.exists() and not out.parent.exists()
    assert _hash_tree(tmp_path) == before


def test_main_rejects_unknown_check(capsys):
    with pytest.raises(SystemExit) as e:
        cd.main(["bogus"])
    assert e.value.code == 2


@pytest.mark.integration
def test_cli_subprocess_help_and_dry_run(corpus, lyrics, tmp_path):
    proc = run_py(["-m", "training.check_dataset", "--help"], timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    for flag in ("--corpus-dir", "--lyrics", "--sample", "--seed", "--out", "--dry-run"):
        assert flag in proc.stdout
    out = tmp_path / "r.md"
    before = _hash_tree(tmp_path)
    proc = run_py(["-m", "training.check_dataset", "all", "--corpus-dir", str(corpus),
                   "--lyrics", str(lyrics), "--out", str(out), "--dry-run"], timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "## " in proc.stdout
    assert _hash_tree(tmp_path) == before


# ------------------------------------------------------------------ textdupes / script

needs_script = pytest.mark.skipif(not all(hasattr(cd, n) for n in ("text_hash", "check_textdupes", "line_script",
                                                                  "check_script")),
                                  reason="в check_dataset ещё нет textdupes/script")


@needs_script
def test_text_hash_normalization():
    base = cd.text_hash("Я на районе\nДеньги в кармане")
    assert base == cd.text_hash("я  на   районе\n\nденьги в кармане ")
    assert base == cd.text_hash("Я НА РАЙОНЕ,\r\nденьги — в кармане!!!")
    assert base != cd.text_hash("Я на районе\nДеньги в тумане")
    assert len(base) == 40 and base == cd.text_hash("Я на районе\nДеньги в кармане")


@needs_script
def test_textdupes_across_different_filenames(tmp_path):
    song = "Я на районе снова ночью\nДеньги в кармане, всё как хочешь\n"
    _song_file(tmp_path / "genius_alpha.json", [song, CLEAN_SONG * 3])
    _song_file(tmp_path / "genius_totally_other.json", [song.upper().replace(",", " ")])
    _song_file(tmp_path / "genius_third.json", ["коротко", "строка " * 20])
    r = cd.check_textdupes(tmp_path)
    assert r["songs"] == 4                                     # "коротко" < 40 символов
    assert r["unique"] == 3 and r["extra_copies"] == 1
    assert r["dupe_share"] == pytest.approx(25.0)
    assert r["top_pairs"] == [(("genius_alpha.json", "genius_totally_other.json"), 1)]
    assert cd.check_textdupes(tmp_path / "empty_missing")["dupe_share"] == 0.0


@pytest.mark.parametrize("line,expected", [
    ("Я на районе", "cyr"),
    ("money on my mind", "lat"),
    ("سلام دنیا", "arab"),
    ("я в big city", "lat"),             # латиницы больше
    ("деньги money", "cyr"),             # кириллицы больше
    ("ok, это мой район", "cyr"),
    ("123 !!! ...", None),
    ("", None),
])
def test_line_script(line, expected):
    if not hasattr(cd, "line_script"):
        pytest.skip("нет line_script")
    assert cd.line_script(line) == expected


def _script_file(path, cyr, lat, arab=0):
    text = "\n".join(["я на районе"] * cyr + ["money on my mind"] * lat + ["سلام دنیا"] * arab)
    _song_file(path, [text])


@needs_script
def test_check_script_per_artist_shares(tmp_path):
    _script_file(tmp_path / "genius_ru.json", 2, 1, 1)
    _script_file(tmp_path / "genius_fa.json", 0, 0, 5)
    (tmp_path / "genius_empty.json").write_text(json.dumps({"songs": [{"text": "123"}]}), encoding="utf-8")
    r = cd.check_script(tmp_path)
    per = {name: (n, s) for name, n, s in r["per_artist"]}
    assert set(per) == {"genius_ru.json", "genius_fa.json"}     # файл без букв пропущен
    assert per["genius_ru.json"][0] == 4
    assert per["genius_ru.json"][1] == pytest.approx({"cyr": 50.0, "lat": 25.0, "arab": 25.0})
    assert per["genius_fa.json"][1]["arab"] == 100.0
    assert r["lines"] == 9
    assert r["share"]["arab"] == pytest.approx(600 / 9)
    assert sorted(r["arab_artists"]) == ["genius_fa.json", "genius_ru.json"]
    assert r["en_artists"] == []


@needs_script
def test_check_script_en_threshold_boundary(tmp_path, capsys):
    _script_file(tmp_path / "genius_p69.json", 31, 69)
    _script_file(tmp_path / "genius_p70.json", 30, 70)
    _script_file(tmp_path / "genius_p71.json", 29, 71)
    assert cd.check_script(tmp_path)["en_artists"] == ["genius_p71.json"]              # порог строгий, 70%
    assert cd.check_script(tmp_path, en_threshold=69.5)["en_artists"] == ["genius_p70.json", "genius_p71.json"]
    assert cd.check_script(tmp_path, en_threshold=68.9)["en_artists"] == [
        "genius_p69.json", "genius_p70.json", "genius_p71.json"]
    assert cd.check_script(tmp_path, en_threshold=71)["en_artists"] == []
    code, cap = _main(capsys, "script", "--corpus-dir", str(tmp_path), "--en-threshold", "70")
    assert code == 0 and "Артистов с латиницей больше порога: 1." in cap.out
