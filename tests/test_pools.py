"""training.pools: ключ slug и исключения из пула ru_rap (training/artist_pools/ru_rap_exclude.txt)."""
import subprocess
import sys

import pytest

import training.pools as tp

from conftest import ROOT, subprocess_env


def load_exclude_again():
    return tp.load_ru_rap_exclude(tp.RU_RAP_EXCLUDE_FILE)


@pytest.mark.parametrize("slug,key", [
    ("25-17", "2517"), ("25-17-2", "2517"), ("ALBLAK-52", "alblak52"), ("ALBLAK-52-2", "alblak52"),
    ("47-ak-47", "47ak47"), ("47-ak-47-2", "47ak47"), ("I61-2", "i61"), ("104", "104"),
    ("Kizaru-2", "kizaru"), ("Big-Baby-Tape-3", "bigbabytape"), ("Pyrokinesis-", "pyrokinesis"),
])
def test_pool_key_keeps_numbers_in_names(slug, key):
    assert tp.pool_key(slug) == key


def test_load_ru_rap_exclude_parsing(tmp_path):
    f = tmp_path / "ru_rap_exclude.txt"
    f.write_text("# pop, not rap  -> ru_pop\n\nAnna-Asti\tru_pop\n  Zivert\tru_pop  # inline comment\n"
                 "#Commented-Out\tru_pop\nNoPool-Artist\n25-17\tru_rock\n   \n# rock\nIC3PEAK\tru_rock\n"
                 "Kalush\tuk_other\nYenlik\tkz_uz_other\nEmptyPool\t  \n", encoding="utf-8")
    mapping = tp.load_ru_rap_exclude_pools(f)
    assert mapping == {"annaasti": "ru_pop", "zivert": "ru_pop", "nopoolartist": "ru_other", "2517": "ru_rock",
                     "ic3peak": "ru_rock", "kalush": "uk_other", "yenlik": "kz_uz_other", "emptypool": "ru_other"}
    assert tp.load_ru_rap_exclude(f) == frozenset(mapping)          # колонка pool игнорируется
    assert tp.load_ru_rap_exclude_pools(tmp_path / "missing.txt") == {}
    assert tp.load_ru_rap_exclude(tmp_path / "missing.txt") == frozenset()


def test_real_ru_rap_exclude_file():
    assert tp.RU_RAP_EXCLUDE_FILE.name == "ru_rap_exclude.txt" and tp.RU_RAP_EXCLUDE_FILE.is_file()
    mapping = tp.load_ru_rap_exclude_pools()
    keys = tp.load_ru_rap_exclude()
    assert keys == frozenset(mapping) and keys == load_exclude_again()
    assert set(mapping.values()) == {"ru_pop", "ru_rock", "uk_other", "kz_uz_other"}
    assert mapping["annaasti"] == mapping["zivert"] == mapping["anilorak"] == mapping["monetochka"] == "ru_pop"
    assert mapping["ic3peak"] == mapping["pornofilmy"] == "ru_rock"
    assert mapping["kalush"] == mapping["alyonaalyona"] == "uk_other"
    assert mapping["yenlik"] == "kz_uz_other"
    assert not {"canis", "soh", "kazhe", "arash", "asap"} & keys   # удалены из кэша
    assert not {"kizaru", "bigbabytape", "oxxxymiron", "endspiel", "2517"} & keys
    for raw in tp.RU_RAP_EXCLUDE_FILE.read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.startswith("#"):
            assert raw.count("\t") == 1, raw                      # строго slug<TAB>pool
    header = tp.RU_RAP_EXCLUDE_FILE.read_text(encoding="utf-8").split("\n# pop")[0]
    assert "slug<TAB>pool" in header and "ru_other" in header


def test_pools_has_no_heavy_dependencies():
    code = ("import sys, training.pools; "
            "bad = [m for m in ('torch', 'training.jef_challenge', 'training.build_mix') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=subprocess_env(),
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_default_exclude_pool_constant():
    assert tp.DEFAULT_EXCLUDE_POOL == "ru_other"
    assert tp.RU_RAP_EXCLUDE_FILE == ROOT / "training" / "artist_pools" / "ru_rap_exclude.txt"


def _pool_slugs(path):
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def test_ru_pop_rock_chanson_no_kino_tsoi_or_korol_case_dupes():
    # дубли Kino/Viktor-Tsoi и Korol-i-Shut/Korol-I-Shut давали два кэша одного текста
    path = ROOT / "training" / "artist_pools" / "ru_pop_rock_chanson.txt"
    assert path.is_file()
    slugs = _pool_slugs(path)
    keys = [tp.pool_key(s) for s in slugs]
    assert keys.count("kino") == 1
    assert "viktortsoi" not in keys and "tsoi" not in keys
    assert keys.count("korolishut") == 1
    assert "Korol-I-Shut" not in slugs and "Koroli-i-Shut" not in slugs


def test_ru_rap_extra_has_mid_underground_volume():
    path = ROOT / "training" / "artist_pools" / "ru_rap_extra.txt"
    slugs = _pool_slugs(path)
    assert 150 <= len(slugs) <= 300
    keys = {tp.pool_key(s) for s in slugs}
    assert {"pasosh", "triagrutrika", "heronwater", "kunteynir", "blagowhite"} <= keys
    assert not {"zivert", "nyusha", "loboda", "anilorak"} & keys
