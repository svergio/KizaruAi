"""Тесты training/build_mix.py (сборка микса по пулам).

Всё на крошечных пулах в tmp_path: списки артистов, genius_*.json, рецепт yaml.
Реальный файл используется только один: training/artist_pools/ru_rap_exclude.txt
(Anna-Asti -> ru_pop). Сеть, GPU и модели не нужны.

Регрессии для исправленных багов: пустой ключ '' в карте артистов, суффикс дубля -2,
доля 0 в рецепте, сплит по нормализованному тексту, dry-run с нехваткой -> код 3.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from training import build_mix as bm  # noqa: E402
from training.check_dataset import text_hash  # noqa: E402
from training.pools import pool_key  # noqa: E402

from conftest import run_py  # noqa: E402

RU_WORDS = ["город", "ночь", "дым", "бит", "район", "улица", "свет", "рифма",
            "ветер", "дорога", "звезда", "огонь", "дождь", "крыша", "мост", "пламя"]
EN_WORDS = ["city", "night", "smoke", "beat", "block", "street", "light", "rhyme",
            "wind", "road", "star", "fire", "rain", "roof", "bridge", "flame"]


def _tag(i: int, words: list[str], width: int = 4) -> str:
    """Уникальная «подпись» песни словами (цифры не годятся: важны буквы)."""
    out = []
    for _ in range(width):
        out.append(words[i % len(words)])
        i //= len(words)
    return " ".join(out)


def ru_song(i: int, lines: int = 6) -> str:
    return "\n".join(f"{_tag(i * 31 + k, RU_WORDS)} снова {RU_WORDS[k % 16]} внутри" for k in range(lines))


def en_song(i: int, lines: int = 6) -> str:
    return "\n".join(f"{_tag(i * 31 + k, EN_WORDS)} again {EN_WORDS[k % 16]} inside" for k in range(lines))


ARABIC_SONG = "\n".join(["مرحبا بك في المدينة الليلة", "الليل طويل والنجوم بعيدة",
                         "نحن هنا من اجل الموسيقى", "صوت الشارع في قلبي دائما"])


def write_genius(corpus: Path, slug: str, texts: list[str]) -> None:
    payload = {"slug": slug, "songs": [{"title": f"t{i}", "text": t} for i, t in enumerate(texts)]}
    (corpus / f"genius_{slug}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def quiet(_msg):
    return None


# --------------------------------------------------------------------- фикстуры


@pytest.fixture
def hermetic_lists(tmp_path, monkeypatch):
    """Списки артистов только из tmp: SOLO_RETRY_LIST и POETS_LIST - фиксированные пути, глушим."""
    monkeypatch.setattr(bm, "SOLO_RETRY_LIST", tmp_path / "no_solo_retry.txt")
    monkeypatch.setattr(bm, "POETS_LIST", tmp_path / "no_poets.txt")
    pools = tmp_path / "artist_pools"
    pools.mkdir()
    return pools


@pytest.fixture
def world(tmp_path, hermetic_lists, monkeypatch):
    """Мини-корпус: два рэпера (ru_rap), Anna-Asti (в списке рэпа, но exclude -> ru_pop),
    англ. рэпер (en_rap), неизвестный кириллический артист (ru_other), дубли и мусор."""
    pools = hermetic_lists
    (pools / "00_existing.txt").write_text("# рэп\ntst-rap-one\ntst-rap-two\nAnna-Asti\n", encoding="utf-8")
    (pools / "en_rap.txt").write_text("tst-en-one\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    poetry = tmp_path / "poetry"
    poetry.mkdir()

    rap_one = [ru_song(i) for i in range(40)] + [en_song(900), "коротко"]   # 1 латинская, 1 короткая
    rap_two = [ru_song(i) for i in range(100, 140)] + [ru_song(0).upper() + " !!!", ARABIC_SONG]
    asti = [ru_song(i) for i in range(200, 220)] + [ru_song(1)]             # дубль песни tst-rap-one
    en_one = [en_song(i) for i in range(300, 320)] + [ru_song(301)]         # 1 кириллическая
    nobody = [ru_song(i) for i in range(400, 410)]
    write_genius(corpus, "tst-rap-one", rap_one)
    write_genius(corpus, "tst-rap-two", rap_two)
    write_genius(corpus, "Anna-Asti", asti)
    write_genius(corpus, "tst-en-one", en_one)
    write_genius(corpus, "tst-nobody", nobody)
    # huggingartists дублирует genius-песню: genius читается первым -> копия hf выпадает
    (corpus / "huggingartists_tst_hf.json").write_text(
        json.dumps({"train": [ru_song(5), ru_song(500)]}, ensure_ascii=False), encoding="utf-8")

    real_load = bm.load_artist_pools
    monkeypatch.setattr(bm, "load_artist_pools", lambda pools_dir=pools: real_load(pools_dir))
    monkeypatch.setattr(bm, "DEFAULT_OUT_ROOT", tmp_path / "default_out")
    return {"root": tmp_path, "pools": pools, "corpus": corpus, "poetry": poetry}


@pytest.fixture
def rap_exclude():
    rx = bm.load_rap_exclude(log=quiet)
    assert rx, "реальный training/artist_pools/ru_rap_exclude.txt пуст или не найден"
    return rx


def collect(world, val=0.1, test=0.1, rx=None):
    """(songs, stats). collect_songs может вернуть и третий элемент (BigPool stihi) - он не нужен."""
    res = bm.collect_songs(world["corpus"], world["poetry"], bm.load_artist_pools(),
                            val, test, keep_text=True, log=quiet,
                            rap_exclude=rx if rx is not None else bm.load_rap_exclude(log=quiet))
    return res[0], res[1]


def write_recipe(path: Path, pools: dict, total="max", val=0.1, test=0.1, seed=1, name="tst_mix") -> Path:
    path.write_text(yaml.safe_dump({"name": name, "seed": seed, "total_tokens": total,
                                    "splits": {"val": val, "test": test}, "pools": pools},
                                   allow_unicode=True), encoding="utf-8")
    return path


def tree_snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*")}


# -------------------------------------------------------------------- song_filter


@pytest.mark.parametrize("pool, cnt, why", [
    ("ru_rap", Counter(), "no_letters"),
    ("ru_rap", Counter(cyr=10, arab=1), "arabic_script"),
    ("en_rap", Counter(lat=10, arab=1), "arabic_script"),
    ("other", Counter(arab=3), "arabic_script"),
    ("ru_rap", Counter(cyr=3, lat=3), None),            # ровно 0.5 - порог строгий
    ("ru_rap", Counter(cyr=3, lat=4), "latin_in_ru"),
    ("ru_pop", Counter(cyr=3, lat=4), "latin_in_ru"),   # пулы из exclude тоже ru_*
    ("ru_poetry", Counter(cyr=1, lat=9), "latin_in_ru"),
    ("en_rap", Counter(lat=3, cyr=3), None),
    ("en_rap", Counter(lat=3, cyr=4), "cyrillic_in_en"),
    ("en_poetry", Counter(lat=1, cyr=9), "cyrillic_in_en"),
    ("other", Counter(lat=10), None),
    ("uk_other", Counter(lat=9, cyr=1), None),          # не ru_* и не en_* - не режем
    ("ru_rap", Counter(cyr=10), None),
    ("en_rap", Counter(lat=10), None),
])
def test_song_filter(pool, cnt, why):
    assert bm.song_filter(pool, cnt) == why


# ------------------------------------------------------------------- resolve_pool


@pytest.mark.parametrize("base, lat, pool", [
    (None, 0.0, "ru_other"),
    (None, 0.70, "ru_other"),      # порог ARTIST_EN_SHARE строгий
    (None, 0.71, "other"),
    ("ru_rap", 0.2, "ru_rap"),
    ("ru_rap", 0.70, "ru_rap"),
    ("ru_rap", 0.9, "en_rap"),
    ("ru_poetry", 0.1, "ru_poetry"),
    ("ru_poetry", 0.9, "en_poetry"),
    ("ru_other", 0.9, "other"),
    ("ru_other", 0.1, "ru_other"),
    ("en_rap", 0.0, "en_rap"),     # en_rap из списка не переводится обратно
    ("en_poetry", 0.0, "en_poetry"),
])
def test_resolve_pool(base, lat, pool):
    assert bm.resolve_pool(base, lat) == pool


def test_resolve_pool_threshold_is_constant():
    assert bm.ARTIST_EN_SHARE == pytest.approx(0.70)
    assert bm.SONG_LAT_DROP_RU == pytest.approx(0.50)
    assert bm.SONG_CYR_DROP_EN == pytest.approx(0.50)


# -------------------------------------------------------------------- assign_pool


def test_real_exclude_has_asti_as_ru_pop(rap_exclude):
    assert rap_exclude[pool_key("Anna-Asti")] == "ru_pop"
    assert "ru_rap" not in set(rap_exclude.values())


@pytest.mark.parametrize("source, slug", [
    ("genius", "Anna-Asti"), ("lyricsgenius", "Anna-Asti"), ("huggingartists", "anna-asti"),
    ("genius", "anna-asti-2"),
])
def test_assign_pool_asti_is_ru_pop_not_rap(rap_exclude, source, slug):
    artist_map = {"annaasti": "ru_rap"}  # даже если артист стоит в рэп-списке
    assert bm.assign_pool(source, slug, 0.05, artist_map, rap_exclude) == "ru_pop"
    # явный пул из exclude не зависит от доли латиницы
    assert bm.assign_pool(source, slug, 0.95, artist_map, rap_exclude) == "ru_pop"


@pytest.mark.parametrize("source, slug, lat, artist_map, pool", [
    ("genius", "tst-unknown", 0.05, {}, "ru_other"),
    ("genius", "tst-unknown", 0.95, {}, "other"),
    ("genius", "tst-rapper", 0.05, {"tstrapper": "ru_rap"}, "ru_rap"),
    ("genius", "tst-rapper", 0.95, {"tstrapper": "ru_rap"}, "en_rap"),
    ("genius", "tst-en", 0.95, {"tsten": "en_rap"}, "en_rap"),
    ("genius", "tst-pop", 0.05, {"tstpop": "ru_other"}, "ru_other"),
    ("huggingartists", "tst-anything", 0.05, {}, "ru_rap"),
    ("huggingartists", "tst-anything", 0.95, {}, "en_rap"),
    ("ru_poetry_txt", "classical_poetry", 0.0, {}, "ru_poetry"),
    ("en_poetry_form", "Some Poet", 1.0, {}, "en_poetry"),
    ("en_rhymedata", "Other Poet", 1.0, {}, "en_poetry"),
])
def test_assign_pool_cases(rap_exclude, source, slug, lat, artist_map, pool):
    assert bm.assign_pool(source, slug, lat, artist_map, rap_exclude) == pool


def test_assign_pool_exclude_only_for_song_sources(rap_exclude):
    # стихи с именем, совпадающим с exclude, остаются стихами
    assert bm.assign_pool("ru_poetry_txt", "Anna-Asti", 0.0, {}, rap_exclude) == "ru_poetry"


# ------------------------------------------------------------- load_artist_pools


def test_load_artist_pools_priority(hermetic_lists):
    pools = hermetic_lists
    (pools / "00_existing.txt").write_text("tst-a\ntst-dup\n# tst-commented\n", encoding="utf-8")
    (pools / "ru_expand.txt").write_text("tst-b\n", encoding="utf-8")
    (pools / "ru_pop_rock_chanson.txt").write_text("tst-pop\ntst-dup\n", encoding="utf-8")
    (pools / "en_rap.txt").write_text("tst-en\ntst-dup\n", encoding="utf-8")
    m = bm.load_artist_pools(pools)
    assert m == {"tsta": "ru_rap", "tstb": "ru_rap", "tstpop": "ru_other",
                 "tsten": "en_rap", "tstdup": "en_rap"}


def test_load_artist_pools_missing_dir_is_empty(hermetic_lists, tmp_path):
    assert bm.load_artist_pools(tmp_path / "nope") == {}


def test_artist_slug_with_dup_suffix_matches_list(hermetic_lists, rap_exclude):
    (hermetic_lists / "00_existing.txt").write_text("tst-rapper\n", encoding="utf-8")
    m = bm.load_artist_pools(hermetic_lists)
    assert bm.assign_pool("genius", "tst-rapper-2", 0.05, m, rap_exclude) == "ru_rap"


def test_cyrillic_slug_in_list_does_not_create_empty_key(hermetic_lists, rap_exclude):
    (hermetic_lists / "00_existing.txt").write_text("Каста\n", encoding="utf-8")
    m = bm.load_artist_pools(hermetic_lists)
    assert "" not in m
    assert bm.assign_pool("genius", "Кто-то-другой", 0.05, m, rap_exclude) == "ru_other"
    assert m == {"каста": "ru_rap"}
    assert bm.assign_pool("genius", "Каста", 0.05, m, rap_exclude) == "ru_rap"
    assert bm.assign_pool("genius", "каста-2", 0.05, m, rap_exclude) == "ru_rap"


# ----------------------------------------------------------------------- split_of


HASHES = [hashlib.sha1(str(i).encode()).hexdigest() for i in range(5000)]


def test_split_of_boundaries():
    assert bm.split_of("00000000" + "0" * 32, 0.02, 0.02) == "test"
    assert bm.split_of("ffffffff" + "0" * 32, 0.02, 0.02) == "train"
    assert bm.split_of("ffffffff" + "0" * 32, 0.0, 0.0) == "train"
    assert bm.split_of("00000000" + "0" * 32, 0.0, 0.0) == "train"


def test_split_of_deterministic_and_proportional():
    a = [bm.split_of(h, 0.1, 0.1) for h in HASHES]
    b = [bm.split_of(h, 0.1, 0.1) for h in HASHES]
    assert a == b
    c = Counter(a)
    assert set(c) == {"train", "val", "test"}
    assert 0.08 < c["test"] / len(a) < 0.12
    assert 0.08 < c["val"] / len(a) < 0.12


def test_split_of_stable_when_proportions_change():
    small = {h: bm.split_of(h, 0.02, 0.02) for h in HASHES}
    big = {h: bm.split_of(h, 0.05, 0.05) for h in HASHES}
    # test при росте долей остаётся test; train при уменьшении долей остаётся train
    assert all(big[h] == "test" for h, s in small.items() if s == "test")
    assert all(small[h] == "train" for h, s in big.items() if s == "train")
    # изменение только val не трогает test
    assert all((bm.split_of(h, 0.3, 0.02) == "test") == (small[h] == "test") for h in HASHES)


def test_split_does_not_depend_on_recipe_pool_shares(world, tmp_path):
    r1 = bm.load_recipe(str(write_recipe(tmp_path / "r1.yaml", {"ru_rap": 85, "ru_pop": 15})))
    r2 = bm.load_recipe(str(write_recipe(tmp_path / "r2.yaml", {"ru_rap": 10, "en_rap": 90})))
    s1, _ = collect(world, r1["splits"]["val"], r1["splits"]["test"])
    s2, _ = collect(world, r2["splits"]["val"], r2["splits"]["test"])
    assert {s["hash"]: s["split"] for s in s1} == {s["hash"]: s["split"] for s in s2}


# ------------------------------------------------------------------ collect_songs


def test_collect_songs_pools_filters_and_dedup(world, rap_exclude):
    songs, stats = collect(world, rx=rap_exclude)
    by_artist = Counter((s["artist"], s["pool"]) for s in songs)
    # 42 песни: латинская, короткая и ru_song(1) (её раньше забрала Anna-Asti) выпали
    assert by_artist[("tst-rap-one", "ru_rap")] == 39
    assert by_artist[("Anna-Asti", "ru_pop")] == 21             # Asti -> ru_pop, не ru_rap
    assert not any(s["artist"] == "Anna-Asti" and s["pool"] == "ru_rap" for s in songs)
    assert by_artist[("tst-en-one", "en_rap")] == 20            # кириллическая выпала
    assert by_artist[("tst-nobody", "ru_other")] == 10          # неизвестный кириллический
    assert by_artist[("tst-rap-two", "ru_rap")] == 40           # дубль и арабская выпали
    assert by_artist[("tst-hf", "ru_rap")] == 1                 # дубль genius выпал
    assert stats["drop_short"] >= 1
    assert stats["drop_latin_in_ru"] == 1
    assert stats["drop_cyrillic_in_en"] == 1
    assert stats["drop_arabic_script"] == 1
    # дубли: ru_song(0) у rap-two (регистр/пунктуация), ru_song(1) у Asti, ru_song(5) у hf
    assert stats["drop_duplicate"] == 3
    assert stats["kept_songs"] == len(songs)


def test_dedup_across_pools_first_source_wins(world, rap_exclude):
    songs, _ = collect(world, rx=rap_exclude)
    hashes = [s["hash"] for s in songs]
    assert len(hashes) == len(set(hashes))
    # ru_song(1): genius_Anna-Asti.json идёт раньше genius_tst-rap-one.json (сортировка имён)
    owner = {s["hash"]: s for s in songs}
    dup = owner[text_hash(bm.prepare_corpus_text(ru_song(1), genius=True).strip())]
    assert (dup["artist"], dup["pool"]) == ("Anna-Asti", "ru_pop")
    dup0 = owner[text_hash(bm.prepare_corpus_text(ru_song(0), genius=True).strip())]
    assert dup0["artist"] == "tst-rap-one"


def test_collect_songs_one_split_per_song(world, rap_exclude):
    songs, _ = collect(world, 0.1, 0.1, rx=rap_exclude)
    for s in songs:
        assert s["split"] == bm.split_of(bm.split_hash(s["text"]), 0.1, 0.1)
        assert s["hash"] == text_hash(s["text"])
        assert s["split"] in bm.SPLITS
        assert s["tokens"] == bm.count_tokens(s["text"]) > 0


# -------------------------------------------------------------- max_feasible_total


def test_max_feasible_total():
    assert bm.max_feasible_total({"a": 1000, "b": 100}, {"a": 0.5, "b": 0.5}) == (200, "b")
    assert bm.max_feasible_total({"a": 850, "b": 100}, {"a": 0.85, "b": 0.15}) == (666, "b")
    assert bm.max_feasible_total({"a": 1000}, {"a": 0.9, "b": 0.1}) == (0, "b")  # пула нет вовсе
    assert bm.max_feasible_total({"a": 1000}, {"a": 1.0}) == (1000, "a")


def test_max_feasible_total_zero_share():
    assert bm.max_feasible_total({"a": 100, "b": 0}, {"a": 1.0, "b": 0.0}) == (100, "a")


# ----------------------------------------------------------- plan_targets / sample


def synthetic_songs(n_by_pool: dict[str, int], val=0.02, test=0.02, seed=0) -> list[dict]:
    rnd = random.Random(seed)
    songs = []
    for pool, n in n_by_pool.items():
        for i in range(n):
            h = hashlib.sha1(f"{pool}:{i}".encode()).hexdigest()
            songs.append({"id": f"{pool}:{i}", "pool": pool, "hash": h, "split": bm.split_of(h, val, test),
                          "source": "genius", "artist": f"{pool}-a{i % 7}", "title": None,
                          "tokens": rnd.randint(20, 60), "lines": 8, "lat_lines": 0, "script_lines": 8})
    return songs


RECIPE = {"pools": {"ru_rap": 0.85, "ru_poetry": 0.10, "en_rap": 0.05},
          "splits": {"val": 0.02, "test": 0.02}, "total_tokens": "max"}


@pytest.fixture(scope="module")
def big_pool():
    return synthetic_songs({"ru_rap": 6000, "ru_poetry": 3000, "en_rap": 600})


def _shares(rows):
    tok = Counter()
    for s in rows:
        tok[s["pool"]] += s["tokens"]
    total = sum(tok.values())
    return {p: t / total for p, t in tok.items()}, total


def test_plan_targets_shares_within_one_percent(big_pool):
    plan = bm.plan_targets(dict(RECIPE), bm.pool_table(big_pool))
    assert plan["short"] == []
    assert plan["limiting_pool"].split()[0] in RECIPE["pools"]
    chosen = bm.sample(big_pool, plan["targets"], seed=1)
    shares, total = _shares(chosen["train"])
    for pool, want in RECIPE["pools"].items():
        assert abs(shares[pool] - want) <= 0.01, (pool, shares[pool], want)
    assert total >= plan["total_train_tokens"]
    assert total <= plan["total_train_tokens"] + 3 * 60  # перебор не больше одной песни на пул
    for split in ("val", "test"):
        sh, _ = _shares(chosen[split])
        for pool, want in RECIPE["pools"].items():
            assert abs(sh[pool] - want) <= 0.03, (split, pool, sh[pool], want)


def test_plan_targets_numeric_total(big_pool):
    recipe = dict(RECIPE, total_tokens=20000)
    plan = bm.plan_targets(recipe, bm.pool_table(big_pool))
    assert plan["total_train_tokens"] == 20000 and plan["short"] == []
    chosen = bm.sample(big_pool, plan["targets"], seed=3)
    shares, total = _shares(chosen["train"])
    assert 20000 <= total <= 20000 + 3 * 60
    for pool, want in RECIPE["pools"].items():
        assert abs(shares[pool] - want) <= 0.01


def test_sample_same_seed_same_output_no_repeats(big_pool):
    plan = bm.plan_targets(dict(RECIPE), bm.pool_table(big_pool))
    a = bm.sample(big_pool, plan["targets"], seed=7)
    b = bm.sample(list(reversed(big_pool)), plan["targets"], seed=7)  # порядок входа не важен
    c = bm.sample(big_pool, plan["targets"], seed=8)
    assert {k: [s["id"] for s in v] for k, v in a.items()} == {k: [s["id"] for s in v] for k, v in b.items()}
    assert [s["id"] for s in a["train"]] != [s["id"] for s in c["train"]]
    all_hashes = [s["hash"] for rows in a.values() for s in rows]
    assert len(all_hashes) == len(set(all_hashes))
    for split, rows in a.items():
        assert all(s["split"] == split for s in rows)


def test_short_pool_reported_and_not_repeated(big_pool):
    table = bm.pool_table(big_pool)
    recipe = dict(RECIPE, total_tokens=10 ** 9)
    plan = bm.plan_targets(recipe, table)
    short = {(x["split"], x["pool"]) for x in plan["short"]}
    assert ("train", "ru_rap") in short and ("train", "en_rap") in short
    for (split, pool), target in plan["targets"].items():
        assert target <= table[pool]["by_split"][split]
    chosen = bm.sample(big_pool, plan["targets"], seed=1)
    hashes = [s["hash"] for rows in chosen.values() for s in rows]
    assert len(hashes) == len(set(hashes))
    assert len(hashes) <= len(big_pool)


def test_load_recipe_normalizes(tmp_path):
    r = bm.load_recipe(str(write_recipe(tmp_path / "r.yaml", {"ru_rap": 85, "ru_poetry": 10, "en_rap": 5},
                                        val=0.02, test=0.03)))
    assert sum(r["pools"].values()) == pytest.approx(1.0)
    assert r["pools"]["ru_rap"] == pytest.approx(0.85)
    assert r["splits"] == {"val": 0.02, "test": 0.03}
    with pytest.raises(ValueError):
        bm.load_recipe(str(write_recipe(tmp_path / "z.yaml", {"ru_rap": 0})))


def test_real_recipes_load():
    recipes = sorted(bm.MIXES_DIR.glob("*.yaml"))
    if not recipes:
        pytest.skip("в training/mixes нет рецептов")
    for fp in recipes:
        r = bm.load_recipe(str(fp))
        assert sum(r["pools"].values()) == pytest.approx(1.0)
        assert 0 <= r["splits"]["val"] + r["splits"]["test"] < 0.5


# ------------------------------------------------------------------------- main()


def test_main_writes_outputs(world, capsys):
    recipe = write_recipe(world["root"] / "mix.yaml", {"ru_rap": 80, "ru_pop": 20})
    out = world["root"] / "out"
    rc = bm.main(["--mix", str(recipe), "--out", str(out), "--corpus-dir", str(world["corpus"]),
                  "--poetry-dir", str(world["poetry"])])
    assert rc == 0, capsys.readouterr()
    for split in bm.SPLITS:
        assert (out / f"{split}.jsonl").is_file() and (out / f"{split}.txt").is_file()
    rows = [json.loads(ln) for split in bm.SPLITS
            for ln in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and len({r["hash"] for r in rows}) == len(rows)
    assert {r["pool"] for r in rows} <= {"ru_rap", "ru_pop"}
    assert not any(r["artist"] == "Anna-Asti" and r["pool"] != "ru_pop" for r in rows)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["short"] == [] and report["mix"] == "tst_mix"
    assert not (world["root"] / "default_out").exists()


def test_main_dry_run_writes_nothing(world, capsys):
    recipe = write_recipe(world["root"] / "mix.yaml", {"ru_rap": 80, "ru_pop": 20})
    out = world["root"] / "out"
    before = tree_snapshot(world["root"])
    rc = bm.main(["--mix", str(recipe), "--out", str(out), "--dry-run",
                  "--corpus-dir", str(world["corpus"]), "--poetry-dir", str(world["poetry"])])
    assert rc == 0
    assert "dry-run" in capsys.readouterr().out
    assert not out.exists()
    assert tree_snapshot(world["root"]) == before


def test_main_insufficient_pool_exit_3(world, capsys):
    recipe = write_recipe(world["root"] / "mix.yaml", {"ru_rap": 80, "ru_pop": 20}, total=10 ** 9)
    out = world["root"] / "out"
    before = tree_snapshot(world["root"])
    rc = bm.main(["--mix", str(recipe), "--out", str(out), "--corpus-dir", str(world["corpus"]),
                  "--poetry-dir", str(world["poetry"])])
    err = capsys.readouterr().err
    assert rc == 3
    assert "ERROR" in err and "--allow-short" in err
    assert "WARN short pool ru_rap/train" in err
    assert not out.exists()
    assert tree_snapshot(world["root"]) == before


def test_main_allow_short_does_not_repeat(world, capsys):
    recipe = write_recipe(world["root"] / "mix.yaml", {"ru_rap": 80, "ru_pop": 20}, total=10 ** 9)
    out = world["root"] / "out"
    rc = bm.main(["--mix", str(recipe), "--out", str(out), "--allow-short",
                  "--corpus-dir", str(world["corpus"]), "--poetry-dir", str(world["poetry"])])
    assert rc == 0
    songs, _ = collect(world)
    avail = Counter(s["hash"] for s in songs if s["pool"] in ("ru_rap", "ru_pop"))
    rows = [json.loads(ln) for split in bm.SPLITS
            for ln in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    hashes = [r["hash"] for r in rows]
    assert len(hashes) == len(set(hashes))          # без повторов
    assert set(hashes) <= set(avail)
    assert len(hashes) == len(avail)                # взято всё, что есть, и не больше
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["short"]


def test_cli_dry_run_subprocess(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_genius(corpus, "tst-nobody", [ru_song(i) for i in range(20)])
    recipe = write_recipe(tmp_path / "mix.yaml", {"ru_other": 1})
    before = tree_snapshot(tmp_path)
    res = run_py(["-m", "training.build_mix", "--mix", str(recipe), "--dry-run",
                  "--out", str(tmp_path / "out"), "--corpus-dir", str(corpus), "--poetry-dir", str(tmp_path)],
                 timeout=120)
    assert res.returncode == 0, res.stderr
    assert "dry-run" in res.stdout and "ru_other" in res.stdout
    assert tree_snapshot(tmp_path) == before


def test_cli_help():
    res = run_py(["-m", "training.build_mix", "--help"], timeout=60)
    assert res.returncode == 0
    assert "--allow-short" in res.stdout and "--dry-run" in res.stdout


# ------------------------------------------- сплит по нормализованному тексту песни


BASE_LINES = ["город ночь дым бит снова внутри", "район улица свет рифма опять",
              "ветер дорога звезда огонь горит", "дождь крыша мост пламя до утра"]
BASE = "\n".join(BASE_LINES)
VARIANTS = {
    "plain": BASE,
    "upper_punct": "\n".join(ln.upper().replace(" ", ", ", 1) + "!!!" for ln in BASE_LINES),
    "tags": "[Куплет 1: Кто-то]\n" + "\n".join(BASE_LINES[:2]) + "\n[Припев]\n"
            + "\n".join(BASE_LINES[2:]) + "\n[Аутро]",
    "spaces": "\n\n".join("  " + ln + "  " for ln in BASE_LINES),
}


@pytest.mark.parametrize("genius", [True, False])
def test_split_hash_same_for_text_variants(genius):
    keys = {name: bm.split_hash(bm.prepare_corpus_text(t, genius=genius).strip()) for name, t in VARIANTS.items()}
    assert len(set(keys.values())) == 1, keys
    # и сырой split_hash снимает [..]-теги сам
    assert bm.split_hash(VARIANTS["tags"]) == bm.split_hash(BASE)
    for val, test in ((0.02, 0.02), (0.1, 0.1), (0.3, 0.2)):
        assert len({bm.split_of(k, val, test) for k in keys.values()}) == 1


def test_same_text_different_artist_id_live_title_one_split(tmp_path, hermetic_lists, monkeypatch):
    """Один текст у разных артистов/источников/названий (в т.ч. '(Live)'): одна копия, один сплит."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i, (name, text) in enumerate(VARIANTS.items()):
        payload = {"slug": f"tst-artist-{name}", "songs": [
            {"title": "Песня (Live)" if i % 2 else "Песня", "text": text}]}
        (corpus / f"genius_tst-artist-{name}.json").write_text(json.dumps(payload, ensure_ascii=False),
                                                               encoding="utf-8")
    (corpus / "huggingartists_tst_hf.json").write_text(
        json.dumps({"train": [VARIANTS["tags"], VARIANTS["upper_punct"]]}, ensure_ascii=False), encoding="utf-8")
    songs, stats, *_ = bm.collect_songs(corpus, tmp_path, {}, 0.1, 0.1, log=quiet, rap_exclude={})
    assert len(songs) == 1 and stats["drop_duplicate"] == len(VARIANTS) + 2 - 1
    assert songs[0]["split"] == bm.split_of(bm.split_hash(BASE), 0.1, 0.1)


def test_split_ignores_seed_and_recipe(world, tmp_path, rap_exclude):
    a, _ = collect(world, 0.1, 0.1, rx=rap_exclude)
    b_, _ = collect(world, 0.1, 0.1, rx={})  # другой состав пулов (без exclude)
    sa = {s["hash"]: s["split"] for s in a}
    sb = {s["hash"]: s["split"] for s in b_}
    assert all(sb[h] == sp for h, sp in sa.items() if h in sb)


# -------------------------------------------------------- битый huggingartists JSON


def test_broken_huggingartists_json_skipped_with_warning(world, rap_exclude, capsys):
    corpus = world["corpus"]
    (corpus / "huggingartists_tst_bad.json").write_text("{not json", encoding="utf-8")
    (corpus / "huggingartists_tst_list.json").write_text("[1, 2, 3]", encoding="utf-8")
    (corpus / "huggingartists_tst_bin.json").write_bytes(b"\xff\xfe\x00garbage")
    songs, stats = collect(world, rx=rap_exclude)
    err = capsys.readouterr().err
    for name in ("huggingartists_tst_bad.json", "huggingartists_tst_list.json", "huggingartists_tst_bin.json"):
        assert "WARN" in err and name in err, err
    assert not any(s["artist"].startswith("tst-bad") or s["artist"].startswith("tst-list") for s in songs)
    assert any(s["artist"] == "tst-hf" for s in songs)  # нормальный hf-файл читается дальше


def test_main_survives_broken_huggingartists(world, capsys):
    (world["corpus"] / "huggingartists_tst_bad.json").write_text("{not json", encoding="utf-8")
    recipe = write_recipe(world["root"] / "mix.yaml", {"ru_rap": 80, "ru_pop": 20})
    rc = bm.main(["--mix", str(recipe), "--dry-run", "--corpus-dir", str(world["corpus"]),
                  "--poetry-dir", str(world["poetry"])])
    assert rc == 0
    assert "huggingartists_tst_bad.json" in capsys.readouterr().err


# ---------------------------------------------------------------- пул с долей 0


def test_zero_share_pool_skipped_in_plan(big_pool):
    recipe = {"pools": {"ru_rap": 0.9, "en_rap": 0.1, "ru_poetry": 0.0, "en_poetry": 0.0},
              "splits": {"val": 0.02, "test": 0.02}, "total_tokens": "max"}
    plan = bm.plan_targets(recipe, bm.pool_table(big_pool))
    assert plan["short"] == []
    assert "ru_poetry" not in plan["caps"] and "en_poetry" not in plan["caps"]
    assert plan["limiting_pool"].split()[0] in ("ru_rap", "en_rap")
    assert all(plan["targets"][(sp, p)] == 0 for sp in bm.SPLITS for p in ("ru_poetry", "en_poetry"))
    chosen = bm.sample(big_pool, plan["targets"], seed=1)
    assert not any(s["pool"] in ("ru_poetry", "en_poetry") for rows in chosen.values() for s in rows)


def test_max_feasible_total_all_zero():
    assert bm.max_feasible_total({"a": 100}, {"a": 0.0}) == (0, None)


def test_main_zero_share_pool_missing_from_corpus(world, capsys):
    recipe = write_recipe(world["root"] / "mix.yaml", {"ru_rap": 80, "ru_pop": 20, "en_poetry": 0})
    out = world["root"] / "out"
    rc = bm.main(["--mix", str(recipe), "--out", str(out), "--corpus-dir", str(world["corpus"]),
                  "--poetry-dir", str(world["poetry"])])
    assert rc == 0, capsys.readouterr().err
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["short"] == []


def test_main_dry_run_shortage_exit_3(world, capsys):
    recipe = write_recipe(world["root"] / "mix.yaml", {"ru_rap": 80, "ru_pop": 20}, total=10 ** 9)
    out = world["root"] / "out"
    before = tree_snapshot(world["root"])
    rc = bm.main(["--mix", str(recipe), "--out", str(out), "--dry-run",
                  "--corpus-dir", str(world["corpus"]), "--poetry-dir", str(world["poetry"])])
    err = capsys.readouterr().err
    assert rc == 3
    assert "ERROR" in err and "WARN short pool" in err
    assert not out.exists()
    assert tree_snapshot(world["root"]) == before
    # с --allow-short dry-run снова 0 и тоже ничего не пишет
    rc = bm.main(["--mix", str(recipe), "--out", str(out), "--dry-run", "--allow-short",
                  "--corpus-dir", str(world["corpus"]), "--poetry-dir", str(world["poetry"])])
    assert rc == 0
    assert tree_snapshot(world["root"]) == before


# ------------------------------------------------------------------ stihi_ru (full)


def _poem(pid: str, n_lines: int, base: int, author="Автор", latin=False) -> dict:
    mk = en_song if latin else ru_song
    lines = [ln for k in range(n_lines) for ln in [mk(base + k, 1)]]
    return {"id": pid, "author": author, "title": f"стих {pid}", "text": "\n".join(lines)}


@pytest.fixture
def stihi_world(world):
    zstd = pytest.importorskip("zstandard")
    pytest.importorskip("numpy")
    raw = world["poetry"] / "raw"
    raw.mkdir()
    poems = [_poem(f"p{i}", 20, 1000 + i * 40) for i in range(30)]      # по 2 куска (16 + 4 строки)
    poems.append(_poem("lat", 16, 5000, latin=True))                     # латиница -> drop_latin_in_ru
    poems.append({"id": "short", "author": "x", "title": "", "text": "коротко"})
    dup_song = ru_song(7)  # такая песня уже есть у tst-rap-one -> кусок-дубль выпадает
    poems.append({"id": "dup", "author": "y", "title": "", "text": dup_song})
    data = "\n".join(json.dumps(p, ensure_ascii=False) for p in poems) + "\nnot json\n"
    src = raw / bm.STIHI_ZST.name
    src.write_bytes(zstd.ZstdCompressor().compress(data.encode("utf-8")))
    return dict(world, stihi_src=src, n_poems=len(poems))


def test_stihi_poem_chunks():
    poem = _poem("p1", 20, 0)
    chunks, short = bm.stihi_poem_chunks(poem)
    assert short == 0 and len(chunks) == 2
    assert [c["id"] for c in chunks] == ["stihi:p1:0", "stihi:p1:1"]
    assert chunks[0]["lines"] == bm.POEM_CHUNK_LINES and chunks[1]["lines"] == 4
    for c in chunks:
        assert c["hash"] == text_hash(c["text"]) and c["tokens"] == bm.count_tokens(c["text"])
        assert c["poem"] == "p1" and c["artist"] == "Автор" and c["cyr"] == c["lines"]
    assert bm.stihi_poem_chunks({"id": "s", "text": "коротко"}) == ([], 1)


def test_stihi_cache_build_and_validity(stihi_world):
    src = stihi_world["stihi_src"]
    data_path, meta_path = bm.stihi_cache_paths(src, stihi_world["poetry"] / "cache")
    assert not bm.stihi_cache_valid(src, data_path, meta_path)
    meta = bm.build_stihi_cache(src, data_path, meta_path, workers=1, log=quiet)
    assert meta["poems"] == stihi_world["n_poems"]           # строка "not json" пропущена
    assert meta["chunks"] == 30 * 2 + 1 + 1 and meta["drop_short"] == 1
    assert bm.stihi_cache_valid(src, data_path, meta_path)
    recs = list(bm.iter_stihi_cache(data_path))
    assert len(recs) == meta["chunks"]
    assert [r["id"] for r in recs] == [r["id"] for r in bm.iter_stihi_cache(data_path)]  # порядок стабилен
    src.write_bytes(src.read_bytes() + b"\x00")              # источник изменился -> кеш невалиден
    assert not bm.stihi_cache_valid(src, data_path, meta_path)


def test_stihi_collect_dedup_and_poem_split(stihi_world, rap_exclude):
    src = stihi_world["stihi_src"]
    data_path, meta_path = bm.stihi_cache_paths(src, stihi_world["poetry"] / "cache")
    bm.build_stihi_cache(src, data_path, meta_path, workers=1, log=quiet)
    songs, stats, big = bm.collect_songs(stihi_world["corpus"], stihi_world["poetry"], bm.load_artist_pools(),
                                         0.2, 0.2, log=quiet, rap_exclude=rap_exclude, stihi_path=data_path)
    assert big is not None and big.pool == "ru_poetry" and len(big) == 60
    assert stats["stihi_chunks"] == 62
    assert stats["drop_latin_in_ru"] >= 2                     # 1 из genius + 1 из stihi
    recs = list(bm.iter_stihi_cache(data_path))
    split_by_poem: dict[str, set] = {}
    for j in range(len(big)):
        rec = recs[big.idx[j]]
        split_by_poem.setdefault(rec["poem"], set()).add(bm.SPLITS[big.split[j]])
    assert "dup" not in split_by_poem and "lat" not in split_by_poem
    assert all(len(v) == 1 for v in split_by_poem.values())   # куски стиха в одном сплите
    song_h64 = {bm.h64_of(s["hash"]) for s in songs}
    assert not song_h64 & set(big.h64)
    table = bm.pool_table(songs, big)
    assert table["ru_poetry"]["songs"] == 60 and table["ru_poetry"]["artists"] == 1
    assert sum(table["ru_poetry"]["by_split"].values()) == table["ru_poetry"]["tokens"]

    recipe = {"pools": {"ru_rap": 0.5, "ru_poetry": 0.5}, "splits": {"val": 0.2, "test": 0.2},
              "total_tokens": "max"}
    plan = bm.plan_targets(recipe, table)
    a = bm.sample(songs, plan["targets"], 3, big)
    b_ = bm.sample(songs, plan["targets"], 3, big)
    assert a == b_
    assert any("_big" in s for rows in a.values() for s in rows)
    bm.hydrate_big(a, big, log=quiet)
    rows = [s for r in a.values() for s in r]
    assert not any("_big" in s for s in rows)
    assert len({s["hash"] for s in rows}) == len(rows)
    stihi_rows = [s for s in rows if s.get("source") == "stihi_ru"]
    assert stihi_rows and all(s["text"] and s["pool"] == "ru_poetry" for s in stihi_rows)


def test_main_stihi_full_end_to_end(stihi_world, capsys):
    recipe = write_recipe(stihi_world["root"] / "mix.yaml", {"ru_rap": 50, "ru_poetry": 50}, val=0.2, test=0.2)
    out = stihi_world["root"] / "out"
    args = ["--mix", str(recipe), "--out", str(out), "--workers", "1",
            "--corpus-dir", str(stihi_world["corpus"]), "--poetry-dir", str(stihi_world["poetry"])]
    assert bm.main(args) == 0, capsys.readouterr().err
    first = capsys.readouterr().out
    assert "собираю кеш" in first
    rows = [json.loads(ln) for sp in bm.SPLITS for ln in (out / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(r["source"] == "stihi_ru" and r["text"] for r in rows)
    assert len({r["hash"] for r in rows}) == len(rows)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["filters"]["stihi_poems"] == stihi_world["n_poems"]
    # второй запуск берёт готовый кеш и даёт тот же микс
    snap = {sp: (out / f"{sp}.jsonl").read_text(encoding="utf-8") for sp in bm.SPLITS}
    assert bm.main(args) == 0
    assert "актуален" in capsys.readouterr().out
    assert {sp: (out / f"{sp}.jsonl").read_text(encoding="utf-8") for sp in bm.SPLITS} == snap


def test_main_stihi_sample_mode_reads_txt(world, capsys):
    (world["poetry"] / "stihi_ru_sample.txt").write_text(
        "\n".join(ru_song(7000 + i, 1) for i in range(64)), encoding="utf-8")
    recipe = write_recipe(world["root"] / "mix.yaml", {"ru_rap": 50, "ru_poetry": 50})
    rc = bm.main(["--mix", str(recipe), "--dry-run", "--stihi", "sample",
                  "--corpus-dir", str(world["corpus"]), "--poetry-dir", str(world["poetry"])])
    out = capsys.readouterr().out
    assert rc == 0
    assert not (world["poetry"] / "cache").exists()
    poetry_row = [ln for ln in out.splitlines() if ln.startswith("ru_poetry")]
    assert poetry_row and poetry_row[0].split()[1] == "4"      # 64 строки / 16 = 4 куска


# ------------------------------------------- незавершённые и битые кэши Genius


INCOMPLETE_SONGS = [ru_song(i) for i in range(600, 606)]
GENIUS_FILES = {
    # имя -> (содержимое, читается ли)
    "tst-incomplete": ({"slug": "tst-incomplete", "complete": False, "songs": INCOMPLETE_SONGS}, False),
    "tst-complete": ({"slug": "tst-complete", "complete": True, "songs": [ru_song(i) for i in range(610, 616)]}, True),
    "tst-nokey": ({"slug": "tst-nokey", "songs": [ru_song(i) for i in range(620, 626)]}, True),
    # текущий код пропускает только complete is False; null/0/"false" считаются полными
    "tst-null": ({"slug": "tst-null", "complete": None, "songs": [ru_song(i) for i in range(630, 633)]}, True),
    "tst-zero": ({"slug": "tst-zero", "complete": 0, "songs": [ru_song(i) for i in range(640, 643)]}, True),
    "tst-strfalse": ({"slug": "tst-strfalse", "complete": "false", "songs": [ru_song(i) for i in range(650, 653)]},
                     True),
}
BROKEN_GENIUS = {
    "genius_tst-broken.json": b"{not json",
    "genius_tst-truncated.json": b'{"slug": "tst-truncated", "complete": true, "songs": [{"text": "\xd0\xb3',
    "genius_tst-emptyfile.json": b"",
    "genius_tst-emptyobj.json": b"{}",
    "genius_tst-list.json": b"[]",
    "genius_tst-jsonnull.json": b"null",
    "genius_tst-bin.json": b"\xff\xfe\x00\x01garbage",
}


@pytest.fixture
def cache_world(world):
    with (world["pools"] / "00_existing.txt").open("a", encoding="utf-8") as fh:
        fh.write("".join(f"{slug}\n" for slug in GENIUS_FILES))  # все - рэп, чтобы попасть в микс
    for slug, (payload, _) in GENIUS_FILES.items():
        payload = dict(payload, songs=[{"title": f"t{i}", "text": t} for i, t in enumerate(payload["songs"])])
        (world["corpus"] / f"genius_{slug}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    for name, data in BROKEN_GENIUS.items():
        (world["corpus"] / name).write_bytes(data)
    (world["corpus"] / "huggingartists_tst_bad.json").write_text("{oops", encoding="utf-8")
    return world


def _prepared_hash(text: str) -> str:
    return text_hash(bm.prepare_corpus_text(text, genius=True).strip())


def test_iter_genius_skips_incomplete_and_broken(cache_world, capsys):
    stats = Counter()
    rows = list(bm.iter_genius(cache_world["corpus"], stats))
    err = capsys.readouterr().err
    artists = Counter(r["artist"] for r in rows)
    for slug, (payload, read) in GENIUS_FILES.items():
        assert (artists[slug] == len(payload["songs"])) if read else (slug not in artists), slug
    assert not any(a.startswith(("tst-broken", "tst-truncated", "tst-empty", "tst-list", "tst-jsonnull", "tst-bin"))
                   for a in artists)
    assert stats["skipped_incomplete"] == 1
    assert stats["skipped_broken"] == len(BROKEN_GENIUS)
    assert "genius_tst-incomplete.json" in err and "complete=false" in err
    for name in BROKEN_GENIUS:
        assert name in err, name
    # файлы без проблем (старые кэши world без поля complete) читаются как раньше
    assert artists["tst-rap-one"] == 42


def test_iter_genius_without_stats_argument(cache_world):
    rows = list(bm.iter_genius(cache_world["corpus"]))  # stats по умолчанию - свой Counter, не падает
    assert not any(r["artist"] == "tst-incomplete" for r in rows)


def test_collect_songs_counts_skipped_files(cache_world, rap_exclude, capsys):
    songs, stats = collect(cache_world, rx=rap_exclude)
    assert stats["skipped_incomplete"] == 1
    assert stats["skipped_broken"] == len(BROKEN_GENIUS) + 1      # + битый huggingartists
    hashes = {s["hash"] for s in songs}
    assert not hashes & {_prepared_hash(t) for t in INCOMPLETE_SONGS}
    assert not any(s["artist"] == "tst-incomplete" for s in songs)
    for slug in ("tst-complete", "tst-nokey", "tst-null", "tst-zero", "tst-strfalse"):
        assert any(s["artist"] == slug and s["pool"] == "ru_rap" for s in songs), slug


def test_collect_songs_clean_corpus_has_zero_skip_counts(world, rap_exclude):
    _, stats = collect(world, rx=rap_exclude)
    assert stats["skipped_incomplete"] == 0 and stats["skipped_broken"] == 0
    assert "skipped_incomplete" in stats and "skipped_broken" in stats  # ключи есть и при нуле


def test_main_report_skip_counts(cache_world, capsys):
    recipe = write_recipe(cache_world["root"] / "mix.yaml", {"ru_rap": 80, "ru_pop": 20})
    out = cache_world["root"] / "out"
    rc = bm.main(["--mix", str(recipe), "--out", str(out), "--corpus-dir", str(cache_world["corpus"]),
                  "--poetry-dir", str(cache_world["poetry"])])
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    assert "skipped files: incomplete=1" in cap.out
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["skipped_incomplete"] == 1
    assert report["skipped_broken"] == len(BROKEN_GENIUS) + 1
    assert report["filters"]["skipped_incomplete"] == 1
    assert report["filters"]["skipped_broken"] == len(BROKEN_GENIUS) + 1
    rows = [json.loads(ln) for sp in bm.SPLITS for ln in (out / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not any(r["artist"] == "tst-incomplete" for r in rows)
    assert not {r["hash"] for r in rows} & {_prepared_hash(t) for t in INCOMPLETE_SONGS}
    pool_artists = report["pools"]["ru_rap"]["artists"]
    assert pool_artists == 3 + 5  # tst-rap-one, tst-rap-two, tst-hf + 5 читаемых кэшей, без tst-incomplete


def test_main_report_zero_skip_counts(world, capsys):
    recipe = write_recipe(world["root"] / "mix.yaml", {"ru_rap": 80, "ru_pop": 20})
    out = world["root"] / "out"
    assert bm.main(["--mix", str(recipe), "--out", str(out), "--corpus-dir", str(world["corpus"]),
                    "--poetry-dir", str(world["poetry"])]) == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["skipped_incomplete"] == 0 and report["skipped_broken"] == 0


def test_main_dry_run_prints_skip_counts_and_writes_nothing(cache_world, capsys):
    recipe = write_recipe(cache_world["root"] / "mix.yaml", {"ru_rap": 80, "ru_pop": 20})
    before = tree_snapshot(cache_world["root"])
    rc = bm.main(["--mix", str(recipe), "--dry-run", "--out", str(cache_world["root"] / "out"),
                  "--corpus-dir", str(cache_world["corpus"]), "--poetry-dir", str(cache_world["poetry"])])
    assert rc == 0
    assert f"incomplete=1" in capsys.readouterr().out
    assert tree_snapshot(cache_world["root"]) == before
