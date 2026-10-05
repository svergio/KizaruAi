import html as html_lib
import json
import re
import urllib.request

url = "https://genius.com/Kizaru-money-long-lyrics"
req = urllib.request.Request(
    url,
    headers={"User-Agent": "Mozilla/5.0 (compatible; KizaruAi/1.0)"},
)
page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")

m = re.search(
    r'lyricsData\\":\{\\"referents\\":\[[^\]]*\],\\"body\\":\{\\"html\\":\\"(.*?)\\"\}\}',
    page,
    re.S,
)
if not m:
    raise SystemExit("no match")

html_body = json.loads('"' + m.group(1) + '"')
text = re.sub(r"<br\s*/?>", "\n", html_body, flags=re.I)
text = re.sub(r"<a[^>]*>", "", text)
text = re.sub(r"</a>", "", text)
text = re.sub(r"<[^>]+>", "", text)
text = html_lib.unescape(text).strip()
print("len", len(text))
print(text[:400])
