import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "export" / "shade_probe"
CANDIDATES = OUT_DIR / "candidates.txt"
CELL = 64
BG = (0, 0, 0)
FG = (255, 255, 255)
DEDUP_DELTA = 0.02
ANCHORS = {" ", "\u2588"}
BLOCK_LINE_CHARS = set(
    "\u2598\u259d\u2596\u2597\u2580\u2584\u258c\u2590\u259a\u259e"
    "\u259b\u259c\u2599\u259f"
)


def load_font(size=48):
    names = [
        "Consolas",
        "Cascadia Mono",
        "CascadiaMono",
        "Segoe UI Symbol",
        "Lucida Sans Unicode",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def white_ratio(char, font):
    img = Image.new("RGB", (CELL, CELL), BG)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (CELL - tw) // 2 - bbox[0]
    y = (CELL - th) // 2 - bbox[1]
    draw.text((x, y), char, fill=FG, font=font)
    gray = img.convert("L")
    data = gray.tobytes()
    return sum(data) / (255.0 * len(data))


def dedupe(rows):
    rows = sorted(rows, key=lambda t: -t[2])
    by_char = {char: (char, cp, ratio) for char, cp, ratio in rows}
    kept = []
    last_ratio = None
    for char, cp, ratio in rows:
        if char in ANCHORS:
            continue
        if last_ratio is None or abs(ratio - last_ratio) >= DEDUP_DELTA:
            kept.append((char, cp, ratio))
            last_ratio = ratio
    for anchor in ANCHORS:
        if anchor in by_char:
            entry = by_char[anchor]
            if not any(c == anchor for c, _, _ in kept):
                kept.append(entry)
    kept.sort(key=lambda t: -t[2])
    return kept


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CANDIDATES.is_file():
        print(f"missing {CANDIDATES}", file=sys.stderr)
        sys.exit(1)
    chars = []
    for line in CANDIDATES.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        chars.append(line[0])
    font = load_font(size=48)
    rows = []
    for char in chars:
        if len(char) != 1:
            continue
        if char in BLOCK_LINE_CHARS:
            continue
        cp = f"U+{ord(char):04X}"
        ratio = white_ratio(char, font)
        rows.append((char, cp, ratio))
    kept_list = dedupe(rows)
    if any(c == " " for c, _, _ in kept_list):
        kept_list = [(c, cp, r) for c, cp, r in kept_list if c != "\u2800"]
    kept_chars = {c for c, _, _ in kept_list}

    ranked_path = OUT_DIR / "ranked.csv"
    with ranked_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["char", "codepoint", "white_ratio", "kept"])
        for char, cp, ratio in rows:
            w.writerow([char, cp, f"{ratio:.6f}", char in kept_chars])

    ratios = [r for _, _, r in kept_list]
    lo, hi = min(ratios), max(ratios)
    span = hi - lo if hi > lo else 1.0

    def norm(r):
        return round((r - lo) / span, 6)

    ramp_sorted = sorted(kept_list, key=lambda t: t[2])
    ramp = [[char, norm(ratio)] for char, cp, ratio in ramp_sorted]
    ramp_path = OUT_DIR / "ramp.json"
    ramp_path.write_text(json.dumps(ramp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    strip_h = CELL
    strip_w = CELL * len(ramp_sorted)
    strip = Image.new("RGB", (strip_w, strip_h), BG)
    draw = ImageDraw.Draw(strip)
    for i, (char, _, _) in enumerate(ramp_sorted):
        bbox = draw.textbbox((0, 0), char, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = i * CELL + (CELL - tw) // 2 - bbox[0]
        y = (CELL - th) // 2 - bbox[1]
        draw.text((x, y), char, fill=FG, font=font)
    strip_path = OUT_DIR / "strip.png"
    strip.save(strip_path)

    print(f"ranked {len(rows)} candidates, retained {len(kept_list)}")
    print(f"  {ranked_path}")
    print(f"  {ramp_path}")
    print(f"  {strip_path}")


if __name__ == "__main__":
    main()
