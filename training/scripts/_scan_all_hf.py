import json
import time
import urllib.error
import urllib.request

API = "https://huggingface.co/api/datasets?author=huggingartists&limit=1000"
req = urllib.request.Request(API, headers={"User-Agent": "KizaruAi/1.0"})
with urllib.request.urlopen(req, timeout=60) as response:
    repos = json.loads(response.read())

ids = sorted(item["id"].split("/", 1)[1] for item in repos)
ok = []
for slug in ids:
    url = f"https://huggingface.co/datasets/huggingartists/{slug}/resolve/main/datasets.json"
    req = urllib.request.Request(url, headers={"User-Agent": "KizaruAi/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            if len(data) >= 100:
                ok.append((slug, len(data)))
    except urllib.error.HTTPError:
        pass
    except Exception:
        pass
    time.sleep(0.15)

print("downloadable", len(ok))
for slug, size in ok:
    print(slug, size)
