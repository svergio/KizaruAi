import csv
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "export" / "unicode_quad_probe"
LINE_PATTERNS_PATH = ROOT / "core" / "image" / "assets" / "line_patterns.json"
FONTS_DIR = ROOT / "data" / "fonts"

CELL = 64
BG = 0
FG = 255
DEDUP_L2 = 0.08
FLAT_SPREAD = 0.08
SOFT_SPREAD = 0.45
LINE_DUP_L2 = 0.08

# Assigned symbol / geometry blocks — not Han / full emoji.
UNICODE_RANGES = (
    (0x2500, 0x257F),  # Box Drawing
    (0x2580, 0x259F),  # Block Elements
    (0x25A0, 0x25FF),  # Geometric Shapes
    (0x2600, 0x26FF),  # Miscellaneous Symbols
    (0x2700, 0x27BF),  # Dingbats
    (0x2800, 0x28FF),  # Braille
    (0x2B00, 0x2BFF),  # Miscellaneous Symbols and Arrows
    (0x1F780, 0x1F7FF),  # Geometric Shapes Extended
    (0x1F800, 0x1F8FF),  # Supplemental Arrows-C
    (0x1FB00, 0x1FBFF),  # Symbols for Legacy Computing
)

FONT_CANDIDATES = (
    "seguisym.ttf",
    "seguiemj.ttf",
    "seguisli.ttf",
    "consola.ttf",
    "CascadiaMono.ttf",
    "CascadiaMonoNF.ttf",
    "arial.ttf",
    "arialuni.ttf",
    "NotoSansSymbols2-Regular.ttf",
    "unifont.ttf",
    "Unifont.ttf",
)


def _font_paths():
    roots = (
        Path(r"C:\Windows\Fonts"),
        FONTS_DIR,
    )
    seen = set()
    out = []
    for root in roots:
        if not root.is_dir():
            continue
        for name in FONT_CANDIDATES:
            path = root / name
            key = str(path.resolve()) if path.is_file() else None
            if key and key not in seen:
                seen.add(key)
                out.append(path)
        for path in sorted(root.glob("*.ttf")) + sorted(root.glob("*.otf")):
            key = str(path.resolve())
            if key not in seen and any(
                n.lower() in path.name.lower()
                for n in ("symbol", "emoji", "unifont", "noto", "cascadia", "consol")
            ):
                seen.add(key)
                out.append(path)
    return out


def load_fonts(size=48):
    fonts = []
    for path in _font_paths():
        try:
            fonts.append((path.name, ImageFont.truetype(str(path), size=size)))
        except OSError:
            continue
    if not fonts:
        fonts.append(("default", ImageFont.load_default()))
    return fonts


def iter_codepoints():
    for start, end in UNICODE_RANGES:
        for cp in range(start, end + 1):
            yield cp


def _mask_bytes(char, font):
    try:
        mask = font.getmask(char)
    except Exception:
        return None
    if mask.size[0] <= 0 or mask.size[1] <= 0:
        return None
    return mask.size, bytes(mask)


# Codepoints that should never have a real glyph; used to fingerprint .notdef / tofu.
_MISSING_SENTINELS = ("\uFFFE", "\uFFFF", "\U0001FFFE")


def _notdef_fingerprints(font):
    prints = []
    for sentinel in _MISSING_SENTINELS:
        packed = _mask_bytes(sentinel, font)
        if packed is not None and any(packed[1]):
            prints.append(packed)
    return prints


def glyph_has_ink(char, font, notdef_prints=None):
    packed = _mask_bytes(char, font)
    if packed is None or not any(packed[1]):
        return False
    if notdef_prints:
        for nd in notdef_prints:
            if packed[0] == nd[0] and packed[1] == nd[1]:
                return False
    return True


def render_glyph(char, font):
    img = Image.new("L", (CELL, CELL), BG)
    probe = ImageDraw.Draw(img)
    bbox = probe.textbbox((0, 0), char, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    if tw <= 0 or th <= 0:
        return img, (0, 0, CELL, CELL)
    x = (CELL - tw) // 2 - bbox[0]
    y = (CELL - th) // 2 - bbox[1]
    draw = ImageDraw.Draw(img)
    draw.text((x, y), char, fill=FG, font=font)
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


def classify(pat):
    mean = sum(pat) / 4.0
    spread = max(pat) - min(pat)
    sorted_q = sorted(pat)
    outlier = sorted_q[-1] - sorted_q[-2] if len(sorted_q) >= 2 else spread

    if spread < FLAT_SPREAD:
        klass = "flat"
    elif spread >= SOFT_SPREAD or outlier >= 0.40:
        klass = "line"
    else:
        klass = "soft"

    tl, tr, bl, br = pat
    pairs = ((tl, br), (tr, bl), (tl, tr), (bl, br), (tl, bl), (tr, br))
    checker = abs(tl - br) < 0.12 and abs(tr - bl) < 0.12 and abs(tl - tr) >= 0.08
    half_h = abs(tl - tr) < 0.10 and abs(bl - br) < 0.10 and abs((tl + tr) / 2 - (bl + br) / 2) >= 0.08
    half_v = abs(tl - bl) < 0.10 and abs(tr - br) < 0.10 and abs((tl + bl) / 2 - (tr + br) / 2) >= 0.08
    singles = sum(1 for v in pat if abs(v - mean) >= 0.12)
    if checker:
        shape = "checker"
    elif half_h:
        shape = "half_h"
    elif half_v:
        shape = "half_v"
    elif singles == 1:
        shape = "single_quad"
    elif singles == 3 or (singles == 2 and not half_h and not half_v):
        shape = "L"
    else:
        shape = "other"
    _ = pairs
    return klass, shape, mean, spread


def load_existing_line_patterns():
    if not LINE_PATTERNS_PATH.is_file():
        return []
    raw = json.loads(LINE_PATTERNS_PATH.read_text(encoding="utf-8"))
    return [(item[0], tuple(float(v) for v in item[1])) for item in raw]


def near_existing_line(pat, existing):
    return any(l2(pat, prev) < LINE_DUP_L2 for _, prev in existing)


def dedupe(rows):
    kept = []
    patterns = []
    for row in rows:
        pat = row["pat"]
        if any(l2(pat, prev) < DEDUP_L2 for prev in patterns):
            continue
        kept.append(row)
        patterns.append(pat)
    return kept


def mini_quad_preview(pat, size=16):
    img = Image.new("L", (size, size), 0)
    half = size // 2
    levels = [int(round(v * 255)) for v in pat]
    for dy, row in enumerate(((0, 1), (2, 3))):
        for dx, idx in enumerate(row):
            val = levels[idx]
            for py in range(dy * half, (dy + 1) * half):
                for px in range(dx * half, (dx + 1) * half):
                    img.putpixel((px, py), val)
    return img


def save_atlas(rows, path, fonts_by_name):
    if not rows:
        Image.new("RGB", (CELL, CELL), (0, 0, 0)).save(path)
        return
    preview_w = CELL + 20
    cols = min(16, max(1, len(rows)))
    rows_n = (len(rows) + cols - 1) // cols
    grid = Image.new("RGB", (cols * preview_w, rows_n * CELL), (0, 0, 0))
    for i, row in enumerate(rows):
        col = i % cols
        r = i // cols
        ox = col * preview_w
        oy = r * CELL
        font = fonts_by_name.get(row["font"])
        if font is None:
            font = next(iter(fonts_by_name.values()))
        glyph, _ = render_glyph(row["char"], font)
        grid.paste(glyph.convert("RGB"), (ox, oy))
        mini = mini_quad_preview(row["pat"], size=16)
        grid.paste(mini.convert("RGB"), (ox + CELL + 2, oy + (CELL - 16) // 2))
    grid.save(path)


def patterns_json(rows):
    return [[row["char"], list(row["pat"])] for row in rows]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fonts = load_fonts(size=48)
    fonts_by_name = {name: font for name, font in fonts}
    notdef_by_font = {name: _notdef_fingerprints(font) for name, font in fonts}
    existing_line = load_existing_line_patterns()
    font_hits = {name: 0 for name, _ in fonts}

    measured = []
    skipped_empty = 0
    skipped_tofu = 0
    for cp in iter_codepoints():
        char = chr(cp)
        chosen = None
        tofu_hit = False
        for name, font in fonts:
            packed = _mask_bytes(char, font)
            if packed is None or not any(packed[1]):
                continue
            nd = notdef_by_font.get(name) or []
            if any(packed[0] == n[0] and packed[1] == n[1] for n in nd):
                tofu_hit = True
                continue
            if glyph_has_ink(char, font, nd):
                chosen = (name, font)
                break
        if chosen is None:
            if tofu_hit:
                skipped_tofu += 1
            else:
                skipped_empty += 1
            continue
        name, font = chosen
        font_hits[name] += 1
        pat = quad_levels(char, font)
        klass, shape, mean, spread = classify(pat)
        measured.append(
            {
                "char": char,
                "cp": f"U+{cp:04X}",
                "pat": pat,
                "class": klass,
                "shape": shape,
                "mean": mean,
                "spread": spread,
                "font": name,
                "line_dup": near_existing_line(pat, existing_line),
            }
        )

    soft_raw = [r for r in measured if r["class"] == "soft" and not r["line_dup"]]
    line_raw = [r for r in measured if r["class"] == "line"]
    soft_kept = dedupe(sorted(soft_raw, key=lambda r: -r["spread"]))
    line_kept = dedupe(sorted(line_raw, key=lambda r: -r["spread"]))

    ranked_path = OUT_DIR / "ranked.csv"
    with ranked_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "char",
                "codepoint",
                "tl",
                "tr",
                "bl",
                "br",
                "mean",
                "spread",
                "class",
                "shape",
                "font",
                "line_dup",
                "soft_kept",
                "line_kept",
            ]
        )
        soft_chars = {r["char"] for r in soft_kept}
        line_chars = {r["char"] for r in line_kept}
        for row in measured:
            tl, tr, bl, br = row["pat"]
            w.writerow(
                [
                    row["char"],
                    row["cp"],
                    f"{tl:.6f}",
                    f"{tr:.6f}",
                    f"{bl:.6f}",
                    f"{br:.6f}",
                    f"{row['mean']:.6f}",
                    f"{row['spread']:.6f}",
                    row["class"],
                    row["shape"],
                    row["font"],
                    int(row["line_dup"]),
                    int(row["char"] in soft_chars),
                    int(row["char"] in line_chars),
                ]
            )

    soft_path = OUT_DIR / "soft_patterns.json"
    line_path = OUT_DIR / "line_like.json"
    soft_path.write_text(
        json.dumps(patterns_json(soft_kept), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    line_path.write_text(
        json.dumps(patterns_json(line_kept), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    save_atlas(soft_kept, OUT_DIR / "atlas_soft.png", fonts_by_name)
    save_atlas(line_kept, OUT_DIR / "atlas_line.png", fonts_by_name)

    summary = {
        "fonts_loaded": [name for name, _ in fonts],
        "font_hits": font_hits,
        "codepoints_scanned": sum(end - start + 1 for start, end in UNICODE_RANGES),
        "skipped_empty": skipped_empty,
        "skipped_tofu": skipped_tofu,
        "measured": len(measured),
        "class_counts": {
            "flat": sum(1 for r in measured if r["class"] == "flat"),
            "soft": sum(1 for r in measured if r["class"] == "soft"),
            "line": sum(1 for r in measured if r["class"] == "line"),
        },
        "soft_raw": len(soft_raw),
        "soft_kept": len(soft_kept),
        "line_kept": len(line_kept),
        "existing_line_patterns": len(existing_line),
        "thresholds": {
            "flat_spread": FLAT_SPREAD,
            "soft_spread": SOFT_SPREAD,
            "dedup_l2": DEDUP_L2,
        },
        "shape_soft": {
            key: sum(1 for r in soft_kept if r["shape"] == key)
            for key in ("checker", "half_h", "half_v", "single_quad", "L", "other")
        },
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"scanned={summary['codepoints_scanned']} measured={len(measured)} "
        f"empty={skipped_empty} tofu={skipped_tofu}"
    )
    print(
        f"classes flat={summary['class_counts']['flat']} "
        f"soft={summary['class_counts']['soft']} line={summary['class_counts']['line']}"
    )
    print(f"soft_kept={len(soft_kept)} line_kept={len(line_kept)}")
    print(f"  {ranked_path}")
    print(f"  {soft_path}")
    print(f"  {line_path}")
    print(f"  {summary_path}")
    print(f"  {OUT_DIR / 'atlas_soft.png'}")
    print(f"  {OUT_DIR / 'atlas_line.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
