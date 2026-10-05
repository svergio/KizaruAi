import urllib.error
import urllib.request
import time

candidates = [
    "kasta", "basta", "guf", "noize-mc", "krovostok", "detsl", "triagrutrika", "mnogotochie",
    "ligalize", "kaspiyskiy-gruz", "smokey-mo", "loc-dog", "st", "ak-47", "cent",
    "miyagi", "lizer", "dzhizus", "denderty", "aikko", "egor-kreed", "dj-artem-artemov", "asper-x",
    "timati", "atl", "feduk", "markul", "loqiemean", "mot", "l-one", "anacondaz", "krec", "yanix",
    "holy-poly", "saluki", "t-killah", "eljey", "gansello", "ivan-dorn", "mult", "vadim",
    "5nizza", "agata-christie", "ddt", "egor-letov", "arash", "smoky-mo", "locdog", "legalize",
    "dolph", "gazgolder", "mdma", "st1", "triagrutrik", "kaspiyski-gruz", "krov", "noize",
    "marlow", "slava-marlow", "kizaru", "pharaoh", "big-baby-tape", "bushido-zho", "5opka",
    "booker", "og-buda", "mayot", "face", "t-fest", "morgenshtern", "slava-kpss",
    "big-russian-boss", "25-17", "baklan", "braii", "scriptonite", "pyrokinesis", "oxxxymiron",
    "basta-rap", "guf-rap", "face-rap", "hleb", "hlebnikoff", "kasta-1", "noize-mc-rap",
    "oxymiron", "oxxxy", "centurion", "st-petersburg", "ivan-dorn-rap", "mot-rap",
    "feduk-rap", "markul-rap", "yanix-rap", "atl-rap", "timati-rap", "l-one-rap",
    "saluki-rap", "eljey-rap", "miyagi-rap", "lizer-rap", "krec-rap", "anacondaz-rap",
]

ok = []
bad = []
seen = set()
for slug in candidates:
    if slug in seen:
        continue
    seen.add(slug)
    url = f"https://huggingface.co/datasets/huggingartists/{slug}/resolve/main/datasets.json"
    req = urllib.request.Request(url, headers={"User-Agent": "KizaruAi/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            data = response.read()
            if len(data) < 100:
                bad.append((slug, "tiny"))
            else:
                ok.append(slug)
    except urllib.error.HTTPError as exc:
        bad.append((slug, exc.code))
    except Exception as exc:
        bad.append((slug, str(exc)[:50]))
    time.sleep(0.35)

print("OK", len(ok))
print(",".join(ok))
print("---FAIL---")
for slug, code in bad:
    print(slug, code)
