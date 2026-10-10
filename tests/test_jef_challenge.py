"""training.jef_challenge: сборка контрольного набора для гейта JEF и оценка с фейковым судьёй."""
import json
import re

import pytest

jc = pytest.importorskip("training.jef_challenge")

PREFIXES = ["я снова вижу этот", "мы гоняем через", "в голове опять", "на районе тихо",
            "мой брат не верит в", "по карманам только", "она не знает про", "в этом городе"]
RHYME_AABB = ["ночь", "дочь", "тень", "день"]
RHYME_ABAB = ["сон", "дом", "звон", "лом"]
RHYME_PAIR = ["машина", "вершина"]
FREE = [["асфальт", "кровать", "бумаги", "стекло"],
        ["мотор", "письмо", "гараж", "трамвай"],
        ["кулак", "вино", "фонарь", "бетон"]]
EN_LINES = ["i was walking down the empty street tonight", "money on my mind and the city never sleeps",
            "pull up in the foreign with the windows down", "we been grinding every single day for this",
            "tell my mama that i made it out the mud", "all these diamonds dancing on my wrist again",
            "never trust a soul that only shows up late", "running through the bag like it was nothing"]
MARKER = "это особая строчка которой нет нигде"


def _song(k: int, marker: bool = False) -> str:
    ends = RHYME_AABB + FREE[0] + RHYME_ABAB + FREE[1] + RHYME_PAIR + FREE[2]
    lines = [f"{PREFIXES[(k + j) % len(PREFIXES)]} {_num(k)} {w}" for j, w in enumerate(ends)]
    if marker:
        lines.insert(5, MARKER)
    return "\n".join(lines)


NUMS = ["первый", "второй", "третий", "четвёртый", "пятый", "шестой", "седьмой", "восьмой",
        "девятый", "десятый", "новый", "старый", "громкий", "тихий"]


def _num(k):
    return NUMS[k % len(NUMS)]


@pytest.fixture
def setup(tmp_path):
    pools = tmp_path / "pools"
    pools.mkdir()
    (pools / "00_existing.txt").write_text("# рэп\nKizaru\nBig-Baby-Tape-2\nCanis\n", encoding="utf-8")
    (pools / "ru_expand.txt").write_text("Kizaru-and-Big-Baby-Tape\n", encoding="utf-8")
    (pools / "ru_rap_exclude.txt").write_text("# wrong Genius artist\nCanis\n", encoding="utf-8")
    rows = []
    for k in range(12):
        artist = "Kizaru" if k % 2 else "Big Baby Tape"
        rows.append({"id": f"genius:{artist}:{k}", "artist": artist, "text": _song(k, marker=(k == 0))})
    rows.append({"id": "dup", "artist": "Kizaru", "text": _song(3).upper()})          # дубль по text_hash
    rows.append({"id": "rock", "artist": "Кино", "text": _song(12)})                   # не рэп
    rows.append({"id": "canis", "artist": "Canis", "text": _song(13)})                 # в deny
    for k in range(6):
        en = "\n".join(EN_LINES[(k + j) % len(EN_LINES)] + f" number {k}" for j in range(6))
        rows.append({"id": f"genius:Gunna:{k}", "artist": "Gunna", "text": en})
    corpus = tmp_path / "lyrics.jsonl"
    corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    jef = tmp_path / "jef"
    jef.mkdir()
    (jef / "train.jsonl").write_text(json.dumps({"text": f"что-то\n{MARKER}", "label_bin": "rhyme"},
                                                ensure_ascii=False) + "\n", encoding="utf-8")
    (jef / "valid.jsonl").write_text(json.dumps({"text": _song(4)}, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"corpus": corpus, "jef": jef, "pools": pools, "tmp": tmp_path, "rows": rows}


def _build(s, out_name="out", seed=1, per=3):
    out = s["tmp"] / out_name
    m = jc.build(s["corpus"], out, seed, per, jef_dir=s["jef"], pools_dir=s["pools"])
    recs = [json.loads(l) for l in (out / jc.CHALLENGE_FILE).read_text(encoding="utf-8").splitlines()]
    return m, recs, out


def _zone(line):
    return jc.rhyme_zone(jc.last_word(line))


def test_pool_parsing_and_artist_keys(tmp_path):
    assert jc.artist_keys("ЗАМАЙ (ZAMAY)") == {"замайzamay", "замай", "zamay"}
    (tmp_path / "00_existing.txt").write_text("# c\nKizaru-and-Guf\nCanis\nBig-Baby-Tape\n", encoding="utf-8")
    assert jc.load_rap_pool(tmp_path) == {"kizaru", "guf", "bigbabytape", "canis"}   # без файла исключений
    (tmp_path / "ru_rap_exclude.txt").write_text("Canis\n", encoding="utf-8")
    pool = jc.load_rap_pool(tmp_path)
    assert pool == {"kizaru", "guf", "bigbabytape"}
    assert jc.load_rap_pool(tmp_path, deny={"guf"}) == {"kizaru", "bigbabytape", "canis"}
    assert jc.is_rap_artist("Big Baby Tape", pool)
    assert not jc.is_rap_artist("Кино", pool)


def test_build_categories_labels_counts(setup):
    m, recs, out = _build(setup)
    counts = {c: sum(r["category"] == c for r in recs) for c in jc.CATEGORIES}
    assert counts == {c: 3 for c in jc.CATEGORIES}, m["available"]
    assert m["counts"] == counts and m["total"] == len(recs) == 33 and m["short"] == []
    for r in recs:
        group, label = jc.CATEGORIES[r["category"]]
        assert r["group"] == group and r["label_bin"] == label
        assert r["song_hash"] and r["song_hash"] in r["song_hashes"]
        assert r["id"].startswith(r["category"])
    assert json.loads((out / jc.MANIFEST_FILE).read_text(encoding="utf-8"))["counts"] == counts
    assert len({jc.text_hash(r["text"]) for r in recs}) == len(recs)


def test_category_structure(setup):
    _, recs, _ = _build(setup, per=5)
    by = {}
    for r in recs:
        by.setdefault(r["category"], []).append(r["text"].split("\n"))
    for a, b in by["pos_couplet"]:
        assert _zone(a) == _zone(b) and jc.last_word(a) != jc.last_word(b)
    for q in by["pos_quatrain"]:
        z = [_zone(x) for x in q]
        assert z[0] == z[1] and z[2] == z[3] or z[0] == z[2] and z[1] == z[3] or z[0] == z[3] and z[1] == z[2]
    for cat in ("neg_unrhymed", "neg_shift"):
        for q in by[cat]:
            zones = [_zone(x) for x in q]
            assert len(q) == 4 and len(set(zones)) == 4
    for a, b in by["neg_near_rhyme"]:
        assert _zone(a) != _zone(b)
    for q in by["trap_repeat_word"]:
        assert len({jc.last_word(x) for x in q}) == 1 and len(q) == 4
    for q in by["trap_prompt_copy"]:
        assert q[:2] == q[2:]
    for q in by["trap_tags"]:
        assert any("[" in x for x in q)
    for q in by["trap_yeah"]:
        assert any(x in jc.YEAH_FILLERS for x in q)
    for q in by["trap_fragment"]:
        assert 1 <= len(q) <= 3 and all(len(x.split()) <= 2 for x in q)
    for r in recs:
        if r["category"] == "trap_english":
            assert not re.search(r"[а-яё]", r["text"], re.I)
            assert len(set(r["song_hashes"])) == 4 and r["lang"] == "en"
        elif r["category"] not in ("trap_yeah", "trap_tags"):
            assert r["lang"] == "ru"


def test_only_rap_artists_and_jef_exclusion(setup):
    m, recs, _ = _build(setup, per=50)
    ru_artists = {r["artist"] for r in recs if r["lang"] == "ru"}
    assert ru_artists <= {"Kizaru", "Big Baby Tape"}
    excluded = {jc.text_hash(setup["rows"][0]["text"]), jc.text_hash(_song(4))}
    hashes = {h for r in recs for h in r["song_hashes"]}
    assert not hashes & excluded
    assert all(MARKER not in r["text"] for r in recs)
    assert all("громкий" not in r["text"] and "тихий" not in r["text"] for r in recs)  # песни Кино и Canis
    assert m["songs"]["ru_excluded_jef"] == 2
    assert m["songs"]["ru_not_rap"] >= 1                         # Кино
    assert m["songs"]["dupe"] >= 1


def test_no_song_from_rock_or_deny(setup):
    _, recs, _ = _build(setup, per=50)
    bad_ids = {"rock", "canis", "dup"}
    assert not {r["source_id"] for r in recs} & bad_ids


def test_deterministic_by_seed(setup):
    _, _, o1 = _build(setup, "a", seed=7)
    _, _, o2 = _build(setup, "b", seed=7)
    assert (o1 / jc.CHALLENGE_FILE).read_bytes() == (o2 / jc.CHALLENGE_FILE).read_bytes()
    assert (o1 / jc.MANIFEST_FILE).read_text(encoding="utf-8").replace(str(o1), "") == \
        (o2 / jc.MANIFEST_FILE).read_text(encoding="utf-8").replace(str(o2), "")
    base = (o1 / jc.CHALLENGE_FILE).read_bytes()
    others = {(_build(setup, f"s{s}", seed=s)[2] / jc.CHALLENGE_FILE).read_bytes() for s in (1, 2, 3)}
    assert any(o != base for o in others)


def test_missing_pools_fails(setup, tmp_path):
    with pytest.raises(SystemExit):
        jc.build(setup["corpus"], tmp_path / "x", 1, 3, jef_dir=setup["jef"], pools_dir=tmp_path / "nope")


# ---------------------------------------------------------------- eval

def _rows(spec):
    rows = []
    for cat, n in spec.items():
        for i in range(n):
            rows.append({"category": cat, "text": f"{cat}|{i}", "label_bin": jc.CATEGORIES[cat][1]})
    return rows


ALL3 = {c: 3 for c in jc.CATEGORIES}


def test_eval_perfect_and_bad_judges():
    rows = _rows(ALL3)
    ideal = lambda texts: [0.95 if t.startswith("pos_") else 0.05 for t in texts]  # noqa: E731
    rep = jc.evaluate(rows, ideal)
    assert rep["gate"] == "PASS"
    assert all(c["pass"] and c["frac_ok"] == 1.0 and c["n"] == 3 for c in rep["categories"].values())
    rep = jc.evaluate(rows, lambda texts: [0.5] * len(texts))
    assert rep["gate"] == "FAIL"
    assert all(c["frac_ok"] == 0.0 for c in rep["categories"].values())
    # судья, который любит повторы, валит только trap_repeat_word
    hacky = lambda texts: [0.95 if t.startswith(("pos_", "trap_repeat")) else 0.05 for t in texts]  # noqa: E731
    rep = jc.evaluate(rows, hacky)
    assert rep["gate"] == "FAIL"
    assert [c for c, r in rep["categories"].items() if not r["pass"]] == ["trap_repeat_word"]


def test_eval_thresholds_are_strict_and_fraction():
    rows = _rows(ALL3)
    table = {"pos_couplet": [0.9, 0.81, 0.8], "neg_shift": [0.1, 0.19, 0.2]}

    def judge(texts):
        out = []
        for t in texts:
            cat, i = t.split("|")
            out.append(table[cat][int(i)] if cat in table else (0.99 if cat.startswith("pos_") else 0.0))
        return out

    rep = jc.evaluate(rows, judge, min_frac=0.6)
    pc, ns = rep["categories"]["pos_couplet"], rep["categories"]["neg_shift"]
    assert pc["frac_ok"] == pytest.approx(2 / 3) and pc["pass"]          # 0.8 не > 0.8
    assert ns["frac_ok"] == pytest.approx(2 / 3) and ns["pass"]          # 0.2 не < 0.2
    assert pc["mean"] == pytest.approx((0.9 + 0.81 + 0.8) / 3)
    assert rep["gate"] == "PASS"
    rep = jc.evaluate(rows, judge, min_frac=0.7)
    assert rep["gate"] == "FAIL" and not rep["categories"]["pos_couplet"]["pass"]


def test_eval_missing_category_fails_and_limit():
    spec = dict(ALL3)
    spec.pop("trap_english")
    rep = jc.evaluate(_rows(spec), lambda t: [0.95 if x.startswith("pos_") else 0.0 for x in t])
    assert rep["gate"] == "FAIL" and rep["categories"]["trap_english"]["n"] == 0
    rep = jc.evaluate(_rows(ALL3), lambda t: [0.95 if x.startswith("pos_") else 0.0 for x in t], limit=2)
    assert all(c["n"] == 2 for c in rep["categories"].values())
    assert "ИТОГ: PASS" in jc.render_report(rep)


def test_cli_build_and_eval_with_fake_judge(setup, monkeypatch, capsys):
    out = setup["tmp"] / "cli"
    assert jc.main(["build", "--out", str(out), "--seed", "3", "--corpus", str(setup["corpus"]),
                    "--jef-dir", str(setup["jef"]), "--pools-dir", str(setup["pools"]),
                    "--per-category", "2"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["total"] == 22

    calls = {}

    def fake_load(checkpoint, **kw):
        calls["ckpt"], calls["kw"] = str(checkpoint), kw
        return lambda texts: [0.0] * len(texts)          # никогда не видит рифму

    monkeypatch.setattr(jc, "load_judge", fake_load)
    report = setup["tmp"] / "rep" / "r.json"
    code = jc.main(["eval", "--judge", "ckpt/x", "--data", str(out), "--report", str(report), "--device", "cpu"])
    assert code == 1
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["gate"] == "FAIL" and rep["judge"].endswith("x")
    assert not rep["categories"]["pos_couplet"]["pass"] and rep["categories"]["trap_yeah"]["pass"]
    assert calls["kw"]["device"] == "cpu"
    assert "ИТОГ: FAIL" in capsys.readouterr().out

    rows = jc.load_challenge(out)
    by_text = {r["text"]: r["group"] for r in rows}
    monkeypatch.setattr(jc, "load_judge",
                        lambda c, **kw: (lambda t: [0.99 if by_text[x] == "positive" else 0.01 for x in t]))
    assert jc.main(["eval", "--judge", "ckpt/x", "--data", str(out / jc.CHALLENGE_FILE)]) == 0


def test_near_rhyme_word_avoids_assonance():
    vocab = {"2:им": ["говорим", "одиноким", "любим"], "1:м": ["дом"]}
    rng = __import__("random").Random(0)
    w = jc.near_rhyme_word("говорим", {jc.rhyme_zone("говорим")}, vocab, rng)
    assert w is not None and w != "одиноким"            # та же ударная «о» — это ассонанс, не берём
    assert jc.rhyme_zone(w)[0] != "о"


def test_pools_reexported_for_back_compat():
    import training.pools as pools
    for name in ("pool_key", "load_ru_rap_exclude", "load_ru_rap_exclude_pools",
                 "RU_RAP_EXCLUDE_FILE", "DEFAULT_EXCLUDE_POOL"):
        assert getattr(jc, name) is getattr(pools, name)
    assert jc.DENY_ARTISTS == pools.load_ru_rap_exclude()


def test_arabic_song_not_used_as_english(tmp_path):
    pools = tmp_path / "pools"
    pools.mkdir()
    (pools / "00_existing.txt").write_text("Kizaru\n", encoding="utf-8")
    rows = [{"id": f"k{k}", "artist": "Kizaru", "text": _song(k)} for k in range(6)]
    rows += [{"id": f"fa{k}", "artist": "Someone", "text": "\n".join(["سلام دنیا من اینجا هستم"] * 6) + " " + "ب" * (k + 1)}
             for k in range(5)]
    rows += [{"id": f"en{k}", "artist": "Gunna",
              "text": "\n".join(EN_LINES[(k + j) % len(EN_LINES)] + f" number {k}" for j in range(6))}
             for k in range(6)]
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    m = jc.build(corpus, tmp_path / "o", 1, 3, jef_dir=tmp_path / "nojef", pools_dir=pools)
    assert m["songs"]["en_songs"] == 6 and m["songs"]["mixed"] == 5
    recs = jc.load_challenge(tmp_path / "o")
    assert not any(jc.ARAB_RE.search(r["text"]) for r in recs)
