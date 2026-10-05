import json
import time
import urllib.error
import urllib.request

API = "https://huggingface.co/api/datasets?author=huggingartists&limit=1000"

req = urllib.request.Request(API, headers={"User-Agent": "KizaruAi/1.0"})
with urllib.request.urlopen(req, timeout=60) as response:
    repos = json.loads(response.read())

ids = sorted(item["id"].split("/", 1)[1] for item in repos)
print("total repos", len(ids))

already_ok = {
    "kasta", "noize-mc", "miyagi", "lizer", "dzhizus", "denderty", "aikko", "egor-kreed",
    "dj-artem-artemov", "asper-x", "5nizza", "agata-christie", "ddt", "egor-letov", "arash",
    "slava-marlow", "kizaru", "pharaoh", "big-baby-tape", "bushido-zho", "5opka", "booker",
    "og-buda", "mayot", "face", "t-fest", "morgenshtern", "slava-kpss", "big-russian-boss",
    "25-17", "baklan", "braii", "scriptonite", "pyrokinesis", "oxxxymiron",
}

skip_western = {
    "21-savage", "50-cent", "6ix9ine", "abba", "adele", "eminem", "drake", "kanye-west",
    "billie-eilish", "ariana-grande", "coldplay", "david-bowie", "bob-dylan", "elton-john",
}

ok = list(already_ok)
for slug in ids:
    if slug in already_ok or slug in skip_western:
        continue
    url = f"https://huggingface.co/datasets/huggingartists/{slug}/resolve/main/datasets.json"
    req = urllib.request.Request(url, headers={"User-Agent": "KizaruAi/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            if len(data) >= 100:
                ok.append(slug)
                print("+", slug, len(data))
    except urllib.error.HTTPError as exc:
        if exc.code not in (401, 404):
            print("?", slug, exc.code)
    except Exception:
        pass
    time.sleep(0.2)
    if len(ok) >= 55:
        break

print("FINAL OK", len(ok))
print(",".join(ok))
