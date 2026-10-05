import csv
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "export" / "line_probe"
CANDIDATES = OUT_DIR / "candidates.txt"
CELL = 64
BG = (0, 0, 0)
FG = (255, 255, 255)
DEDUP_L2 = 0.08
ANCHOR_CHAR = "\u259b"
ANCHOR_PATTERN = (1.0, 1.0, 1.0, 0.0)


def load_font(size=48):
    names = [
        "Segoe UI Symbol",
        "Consolas",
        "Cascadia Mono",
        "CascadiaMono",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_glyph(char, font):
    img = Image.new("L", (CELL, CELL), 0)
    probe = ImageDraw.Draw(img)
    bbox = probe.textbbox((0, 0), char, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (CELL - tw) // 2 - bbox[0]
    y = (CELL - th) // 2 - bbox[1]
    draw = ImageDraw.Draw(img)
    draw.text((x, y), char, fill=255, font=font)
    return img, (x, y, x + tw, y + th)


def quad_levels(char, font):
    img, tbbox = render_glyph(char, font)
    data = img.load()
    white = [(px, py) for py in range(CELL) for px in range(CELL) if data[px, py] > 0]
    if white:
        xs = [p[0] for p in white]
        ys = [p[1] for p in white]
        x0, x1 = min(xs), max(xs) + 1
        y0, y1 = min(ys), max(ys) + 1
    else:
        x0, y0, x1, y1 = tbbox
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, CELL, CELL
    mx = (x0 + x1) // 2
    my = (y0 + y1) // 2

    def mean_region(xa, xb, ya, yb):
        total = 0.0
        count = 0
        for py in range(max(0, ya), min(CELL, yb)):
            for px in range(max(0, xa), min(CELL, xb)):
                total += data[px, py]
                count += 1
        if count == 0:
            return 0.0
        return total / (255.0 * count)

    tl = mean_region(x0, mx, y0, my)
    tr = mean_region(mx, x1, y0, my)
    bl = mean_region(x0, mx, my, y1)
    br = mean_region(mx, x1, my, y1)
    return (tl, tr, bl, br)


def l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def dedupe(rows):
    by_char = {row[0]: row for row in rows}
    kept_patterns = []
    kept = []
    if ANCHOR_CHAR in by_char:
        anchor = by_char[ANCHOR_CHAR]
        kept.append(anchor)
        kept_patterns.append(anchor[2])
    for char, cp, pat in rows:
        if char == ANCHOR_CHAR:
            continue
        if any(l2(pat, prev) < DEDUP_L2 for prev in kept_patterns):
            continue
        kept.append((char, cp, pat))
        kept_patterns.append(pat)
    return kept


def mini_quad_preview(pat, size=16):
    img = Image.new("L", (size, size), 0)
    half = size // 2
    levels = [
        int(round(pat[0] * 255)),
        int(round(pat[1] * 255)),
        int(round(pat[2] * 255)),
        int(round(pat[3] * 255)),
    ]
    for dy, row in enumerate(((0, 1), (2, 3))):
        for dx, idx in enumerate(row):
            val = levels[idx]
            for py in range(dy * half, (dy + 1) * half):
                for px in range(dx * half, (dx + 1) * half):
                    img.putpixel((px, py), val)
    return img


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
        cp = f"U+{ord(char):04X}"
        pat = quad_levels(char, font)
        rows.append((char, cp, pat))
    kept_list = dedupe(rows)
    kept_chars = {c for c, _, _ in kept_list}

    ranked_path = OUT_DIR / "ranked.csv"
    with ranked_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["char", "codepoint", "tl", "tr", "bl", "br", "kept"])
        for char, cp, pat in rows:
            w.writerow(
                [char, cp, f"{pat[0]:.6f}", f"{pat[1]:.6f}", f"{pat[2]:.6f}", f"{pat[3]:.6f}", char in kept_chars]
            )

    patterns = [[char, list(pat)] for char, cp, pat in kept_list]
    patterns_path = OUT_DIR / "patterns.json"
    patterns_path.write_text(json.dumps(patterns, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    preview_w = CELL + 20
    cols = min(16, max(1, len(kept_list)))
    rows_n = (len(kept_list) + cols - 1) // cols
    grid_w = cols * preview_w
    grid_h = rows_n * CELL
    grid = Image.new("RGB", (grid_w, grid_h), BG)
    for i, (char, cp, pat) in enumerate(kept_list):
        col = i % cols
        row = i // cols
        ox = col * preview_w
        oy = row * CELL
        glyph, _ = render_glyph(char, font)
        grid.paste(glyph.convert("RGB"), (ox, oy))
        mini = mini_quad_preview(pat, size=16)
        grid.paste(mini.convert("RGB"), (ox + CELL + 2, oy + (CELL - 16) // 2))
    quads_path = OUT_DIR / "quads.png"
    grid.save(quads_path)

    print(f"ranked {len(rows)} candidates, retained {len(kept_list)}")
    print(f"  {ranked_path}")
    print(f"  {patterns_path}")
    print(f"  {quads_path}")


if __name__ == "__main__":
    main()
