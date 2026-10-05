import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from core.corpus.clean import join_songs
from core.corpus.lyrics import clean_lyrics, prepare_corpus_text
from core.corpus.genius import cache_path, fetch_artist, load_cached_songs

DEFAULT_OUT = Path("data/corpus/kizaru_lyrics.txt")
HF_DATASET_URL = (
    "https://huggingface.co/datasets/huggingartists/{artist}/resolve/main/datasets.json"
)
FALLBACK_CSV = Path("data/corpus/messages_fallback.csv")

GENIUS_ARTISTS = (
    'Oxxxymiron',
    'Noize-MC',
    'Basta',
    'Guf',
    'Skryptonite',
    'Miyagi',
    'Andy-Panda',
    'ATL',
    'Husky',
    'Pharaoh',
    'Kizaru',
    'Morgenshtern',
    'Big-Baby-Tape',
    'Boulevard-Depo',
    'Face',
    'Feduk',
    'Markul',
    'LSP',
    'OG-Buda',
    'Mayot',
    'Soda-Luv',
    'Bushido-Zho',
    'Scally-Milano',
    'Saluki',
    'Toxi$',
    'Yanix',
    'T-Fest',
    'Eldzhey',
    'Gone-Fludd',
    'Lizer',
    'Macan',
    'Xcho',
    'Jakone',
    'Icegergert',
    '104',
    'Lone',
    'ST',
    'Rem-Digga',
    'Loc-Dog',
    'Slim',
    'Ptaha',
    'Timati',
    'Geegun',
    'Mot',
    'Egor-Kreed',
    'Jah-Khalib',
    'Matrang',
    'Thomas-Mraz',
    'I61',
    'Jeembo',
    '25-17',
    'Kasta',
    '47-ak-47',
    'Pasha-technique',
    'Vitya-AK',
    'Noggano',
    'Nevsky-Beat',
    'Anacondaz',
    'No-Name-MC',
    'Sted-D',
    'Horus',
    'Ka-Tet',
    'PLC',
    'Jubilee',
    'Luperkal',
    'Rickey-F',
    'Lida',
    'CMH',
    'Thrill-Pill',
    'Rocket',
    'May-Wave',
    'Flesh',
    'Gspd',
    'DK',
    'Big-Russian-Boss',
    'Young-P-H',
    'Lil-Krystalll',
    'White-Punk',
    'Loqiemean',
    'Mnogoznaal',
    'Gryaz',
    'Lesha-Svik',
    'Niletto',
    'Instasamka',
    'Rakhim',
    '10age',
    'Lovv66',
    'D-Masta',
    'Ooes',
    'Seemee',
    'Friendly-Thug-52-Ngg',
    'Big-Blunt',
    'Kai-Angel',
    '9mice',
    'Aarne',
    'Baby-Melo',
    'New-Sylveon',
    'Slava-KPSS',
    'Obladaet',
    'Platina',
    'Booker',
    '5opka',
    'Pyrokinesis',
    'Baklan',
    'Braii',
    'Sqwoz-Bab',
    'Kishlak',
    'AVG',
    'The-Limba',
    'Dose',
    'ALBLAK-52',
    '163ONMYNECK',
    'Lil-Morty',
    'TVETH',
    'Yung-Trappa',
    'Voskresenskii',
    'Miravi',
    'Slava-Marlow',
    'Hoodini',
    'Moneyken',
    'Schokk',
    'Bumble-Beezy',
    'Murovei',
    'Bahh-Tee',
    'Dead-Dynasty',
    'Summer-Of-Haze',
    'Russian-Village-Boys',
    'Gafur',
    'Ramil',
    'Habib',
    'Jaman-T',
    'WhyBaby',
    'Jellyfish',
    'Porchy',
    'Restorator',
    'Zaraza',
    'Mult',
    'Alphavite',
    'Jabbar',
    'Gansello',
    'Foggy',
    'Maddox',
    'Centr',
    'Smoky-Mo',
    'Ligalize',
    'Bad-Balance',
    'Kaspiyskiy-Gruz',
    'Krovostok',
    'Karandash',
    'GROT',
    'Makulatura',
    'Mnogotochie',
    'Misha-Mavashi',
    'Krec',
    'Kirpichi',
    'Bydlotsykl',
    'Rigos',
    'Belomorkanal',
    'ST1M',
    'Fike',
    'Asik',
    'Ant',
    'Shym',
    'Hamil',
    'Kravz',
    'Gunwest',
    'SD',
    'Obe-1-Kanobe',
    'Syava',
    'Cat-Dog',
    '4atty',
    'Slovetskii',
    'Sidodgi-Duboshit',
    '74-ou74',
    'Kaje-Oboyma',
    'Bledniy',
    'Nurminski',
    'Triada',
    'Yug',
    'Lion-MC',
    'MC-Molodoy',
    'Landy-Star',
    'Tsar',
    'Vlad',
    'Hleb',
    'Czar',
    'Lion',
    'Walkie',
    'Gokilla',
    'Drago',
    'Galat',
    'Zamay',
    'Seimur',
    'Irina-Kairatovna',
    'Ninety-One',
    'Yenlik',
    'RaiM',
    'Konsta',
    'Truwer',
    'Mona',
    'Basic-Boy',
    'Moldanazar',
    'Metox',
    'ZAQ',
    'ALEM',
    'Deuce',
    'Baller',
    'Hiro',
    'Jeltoksan',
    'Shoxrux',
    'Endspiel',
    'Alyona-Alyona',
    'Yarmak',
    'Kalush',
    'Skofka',
    'TNMK',
    'Vova-zi-Lvova',
    'Boombox',
    'Kalush-Orchestra',
    'Tof',
    'Max-Korzh',
    'Tima-Belorusskih',
    'Jony',
    'Andro',
    'HammAli-and-Navai',
    'Rauf-and-Faik',
    'Gayazov-Brother',
    'Poshlaya-Molly',
    'Dora',
    'Maybe-Baby',
    'IC3PEAK',
    'Doni',
    'Elman',
    'Nuki',
    'Miyagi-and-Andy-Panda',
    'Tdd',
    'Griby',
    'Gazirovka',
    'Estradarada',
    'Cream-Soda',
    'Monetochka',
    'Pornofilmy',
    'Shortparis',
    'Sirotkin',
    'No-Name',
    'Invisible',
    'Zion',
    'Zion-Music',
    'Canis',
    'Bitochki',
    'Mozgi',
    'Vladi',
    'Dolphin',
    'SOH',
    'Scally',
    'Hoodiee',
    'Bumble-Bee',
    'Garri-Topor',
    'Ff-sheff',
    'Albert-Nurminskiy',
    'Maloy',
    'Mc-Maloy',
    'Kazhe',
    'Mezza',
    'Feldman',
    'Roma-Zhigan',
    'Esdi',
    'Black-Star-Inc',
    'Wellboy',
    'Jerry-Heil',
    'Mary-Gu',
    'Anna-Asti',
    'Monatik',
    'Artik-and-Asti',
    'Zivert',
    'Hanna',
    'Elvira-T',
    'Nyusha',
    'Loboda',
    'Detsl-aka-Le-Truk',
    'Daite-tank',
    # исправленные slug'и вместо fail artist-not-found из прогона 300
    'Alj',
    'Malbek',
    'Crip-a-crip',
    'Invisible-inc',
    'King-sd',
    'Toxi',
    'Klava-coca',
    'Atlantida-project',
)

CIS_ARTISTS = (
    "kizaru",
    "pharaoh",
    "big-baby-tape",
    "bushido-zho",
    "5opka",
    "booker",
    "og-buda",
    "mayot",
    "face",
    "t-fest",
    "morgenshtern",
    "slava-kpss",
    "big-russian-boss",
    "25-17",
    "baklan",
    "braii",
    "scriptonite",
    "pyrokinesis",
    "oxxxymiron",
)


def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "KizaruAi/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, dest.open("wb") as handle:
        handle.write(response.read())
    return dest


def texts_from_hf_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    songs = payload.get("train") or payload.get("text") or []
    if isinstance(songs, dict):
        songs = list(songs.values())
    return [str(item).strip() for item in songs if str(item).strip()]


def fetch_huggingartists(artist, cache_dir):
    url = HF_DATASET_URL.format(artist=artist)
    raw_path = cache_dir / f"huggingartists_{artist.replace('-', '_')}.json"
    download(url, raw_path)
    texts = texts_from_hf_json(raw_path)
    if not texts:
        raise ValueError(f"huggingartists/{artist} has no texts")
    return texts


def merge_huggingartists(artists, cache_dir):
    merged = []
    stats = []
    errors = []
    for artist in artists:
        try:
            texts = fetch_huggingartists(artist, cache_dir)
            merged.extend(texts)
            stats.append((artist, len(texts), sum(len(t) for t in texts)))
            print(f"  + {artist}: docs={len(texts)} chars={stats[-1][2]}")
        except urllib.error.HTTPError as exc:
            errors.append(f"{artist}: HTTP {exc.code}")
            print(f"  skip {artist}: HTTP {exc.code}", file=sys.stderr)
        except Exception as exc:
            errors.append(f"{artist}: {exc}")
            print(f"  skip {artist}: {exc}", file=sys.stderr)
    if not merged:
        raise ValueError("No huggingartists texts downloaded:\n" + "\n".join(errors))
    return merged, stats, errors


def write_corpus(texts, out_path, genius=False):
    parts = [t for t in texts if (t or "").strip()]
    raw = "\n\n".join(parts)
    before = len(raw)
    if genius:
        cleaned_parts = [prepare_corpus_text(t, genius=True).strip() for t in parts]
        cleaned = join_songs(cleaned_parts, separator="\n")
    else:
        cleaned_parts = [clean_lyrics(t).strip() for t in parts]
        cleaned = join_songs(cleaned_parts, separator="\n")
    after = len(cleaned)
    out_path.write_text(cleaned, encoding="utf-8")
    return len(parts), before, after


def merge_genius(artists, cache_dir, refresh=False):
    stats = []
    errors = []
    for slug in artists:
        path = cache_path(cache_dir, slug)
        if path.exists() and not refresh:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("complete") is True:
                songs = payload.get("songs") or []
                n = len(songs)
                chars = sum(len(item.get("text") or "") for item in songs)
                stats.append((slug, n, chars))
                print(f"  = {slug}: cached docs={n} chars={chars}", flush=True)
                continue
        try:
            payload = fetch_artist(slug, cache_dir)
            songs = payload.get("songs") or []
            n = len(songs)
            chars = sum(len(item.get("text") or "") for item in songs)
            stats.append((slug, n, chars))
            print(f"  + {slug}: docs={n} chars={chars}", flush=True)
            for warn in payload.get("errors") or []:
                print(f"    warn {slug}: {warn}", file=sys.stderr)
        except urllib.error.HTTPError as exc:
            errors.append(f"{slug}: HTTP {exc.code}")
            print(f"  skip {slug}: HTTP {exc.code}", file=sys.stderr)
        except Exception as exc:
            errors.append(f"{slug}: {exc}")
            print(f"  skip {slug}: {exc}", file=sys.stderr)
    texts, cache_stats = load_cached_songs(cache_dir)
    if not texts:
        raise ValueError("No genius lyrics in cache:\n" + "\n".join(errors))
    return texts, stats or cache_stats, errors


def write_corpus_from_genius_cache(cache_dir, out_path):
    texts, stats = load_cached_songs(cache_dir)
    if not texts:
        raise ValueError("No genius_*.json caches found")
    n_docs, before, after = write_corpus(texts, out_path, genius=True)
    return n_docs, before, after, stats


def from_fallback_csv(out_path):
    if not FALLBACK_CSV.exists():
        raise FileNotFoundError(FALLBACK_CSV)
    parts = []
    with FALLBACK_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = (row.get("text") or "").strip()
            if text:
                parts.append(text)
    if not parts:
        raise ValueError("messages_fallback.csv is empty")
    n_docs, before, after = write_corpus(parts, out_path)
    return n_docs, before, after, [("messages_fallback.csv", n_docs, before)]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a local CIS rap lyrics corpus")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--source",
        choices=("genius", "genius-cache", "cis", "kizaru", "fallback", "auto"),
        default="auto",
        help="genius=Genius pool; genius-cache=re-clean caches; cis=HF; auto=genius-cache then fallback",
    )
    parser.add_argument(
        "--artists",
        nargs="*",
        default=None,
        help="Override artist slugs (Genius or huggingartists)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download Genius artists even if cache exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = out_path.parent
    errors = []

    if args.source == "genius":
        artists = tuple(args.artists) if args.artists else GENIUS_ARTISTS
        try:
            print(f"Fetching Genius ({len(artists)} slugs)...")
            texts, stats, genius_errors = merge_genius(
                artists, cache_dir, refresh=args.refresh
            )
            errors.extend(genius_errors)
            n_docs, before, after = write_corpus(texts, out_path, genius=True)
            ok = ", ".join(name for name, n, _ in stats if n > 0)
            print(
                f"wrote {out_path} source=genius artists=[{ok}] docs={n_docs} "
                f"chars_raw={before} chars_clean={after}"
            )
            if errors:
                print("warnings:\n" + "\n".join(errors), file=sys.stderr)
            return
        except Exception as exc:
            errors.append(f"genius: {exc}")
            print(f"genius failed: {exc}", file=sys.stderr)
            raise SystemExit("Could not prepare corpus:\n" + "\n".join(errors)) from exc

    if args.source in ("genius-cache", "auto"):
        try:
            print("Rebuilding corpus from genius_*.json cache...")
            n_docs, before, after, stats = write_corpus_from_genius_cache(cache_dir, out_path)
            ok = ", ".join(name for name, n, _ in stats if n > 0)
            print(
                f"wrote {out_path} source=genius-cache artists=[{ok}] docs={n_docs} "
                f"chars_raw={before} chars_clean={after}"
            )
            return
        except Exception as exc:
            errors.append(f"genius-cache: {exc}")
            print(f"genius-cache failed: {exc}", file=sys.stderr)
            if args.source == "genius-cache":
                raise SystemExit("Could not prepare corpus:\n" + "\n".join(errors)) from exc

    if args.source in ("cis",):
        artists = tuple(args.artists) if args.artists else CIS_ARTISTS
        try:
            print(f"Fetching huggingartists ({len(artists)} slugs)...")
            texts, stats, hf_errors = merge_huggingartists(artists, cache_dir)
            errors.extend(hf_errors)
            n_docs, before, after = write_corpus(texts, out_path)
            ok = ", ".join(name for name, _, _ in stats)
            print(
                f"wrote {out_path} artists=[{ok}] docs={n_docs} "
                f"chars_raw={before} chars_clean={after}"
            )
            if errors:
                print("warnings:\n" + "\n".join(errors), file=sys.stderr)
            return
        except Exception as exc:
            errors.append(f"cis: {exc}")
            print(f"cis failed: {exc}", file=sys.stderr)
            raise SystemExit("Could not prepare corpus:\n" + "\n".join(errors)) from exc

    if args.source in ("kizaru",):
        try:
            texts = fetch_huggingartists("kizaru", cache_dir)
            n_docs, before, after = write_corpus(texts, out_path)
            print(
                f"wrote {out_path} source=kizaru docs={n_docs} "
                f"chars_raw={before} chars_clean={after}"
            )
            return
        except Exception as exc:
            errors.append(f"kizaru: {exc}")
            print(f"kizaru failed: {exc}", file=sys.stderr)

    if args.source in ("fallback", "auto"):
        try:
            n_docs, before, after, _ = from_fallback_csv(out_path)
            print(
                f"wrote {out_path} source=fallback docs={n_docs} "
                f"chars_raw={before} chars_clean={after}"
            )
            return
        except Exception as exc:
            errors.append(f"fallback: {exc}")
            print(f"fallback failed: {exc}", file=sys.stderr)

    raise SystemExit("Could not prepare corpus:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
