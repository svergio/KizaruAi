import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "data" / "export" / "unicode_quad_probe"
ASSETS = ROOT / "core" / "image" / "assets"
LINE_PATH = ASSETS / "line_patterns.json"
DETAILS_PATH = ASSETS / "details_patterns.json"
SHADE_RAMP_PATH = ASSETS / "shade_ramp.json"

LEVEL_EPS = 0.055
LEVEL_EPS_SWEEP = (0.04, 0.055, 0.07)
FLAT_SPREAD = 0.08
DETAILS_MIN_L2 = 0.08
DETAILS_MIN_MEAN_DELTA = 0.04

LINE_ANCHOR_SHADE_EXCLUDE = {"\u259b", "\u2580"}  # keep measured ▛ ▀ out of flat shade ramp
LINE_ANCHORS_KEEP = frozenset(LINE_ANCHOR_SHADE_EXCLUDE)

# Circle / dot fillers that made the old honeycomb shade (do not prefer as gap fillers).
MUDDY_SHADE_SKIP = {
    "\u2026",  # …
    "\u25fd",  # ◽
    "\u25ef",  # ◯
    "\u25cb",  # ○
    "\u25cc",  # ◌
    "\u25d0",  # ◐
    "\u25d1",  # ◑
    "\u25d2",  # ◒
    "\u25d3",  # ◓
    "\u25d4",  # ◔
    "\u25d5",  # ◕
    "\u25ce",  # ◎
}

ARROWISH_GEO = {
    0x25B2,
    0x25B4,
    0x25B6,
    0x25B8,
    0x25BA,
    0x25BC,
    0x25BE,
    0x25C0,
    0x25C2,
    0x25C4,
}


def l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def block_pref(c):
    o = ord(c)
    if 0x2580 <= o <= 0x259F:
        return 0
    if 0x2500 <= o <= 0x257F:
        return 1
    if 0x2800 <= o <= 0x28FF:
        return 2
    if 0x25A0 <= o <= 0x25FF:
        return 3
    if 0x1F780 <= o <= 0x1F7FF:
        return 4
    if 0x1FB00 <= o <= 0x1FBFF:
        return 5
    if 0x2600 <= o <= 0x26FF:
        return 6
    return 99


def reject_line_like(c):
    o = ord(c)
    if 0x2700 <= o <= 0x27BF:
        return "dingbats"
    if 0x2600 <= o <= 0x26FF:
        return "misc_symbols"
    if 0x2190 <= o <= 0x21FF:
        return "arrows"
    if 0x27F0 <= o <= 0x27FF or 0x2900 <= o <= 0x297F:
        return "supp_arrows"
    if 0x2B00 <= o <= 0x2BFF:
        return "misc_arrows"
    if 0x1F800 <= o <= 0x1F8FF:
        return "supp_arrows"
    if o in ARROWISH_GEO:
        return "arrowish_geo"
    if o > 0xFFFF:
        return "emoji"
    return None


def reject_details(c):
    return reject_line_like(c)


def load_json_pairs(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [(item[0], tuple(float(v) for v in item[1])) for item in raw]


def write_json_pairs(path, rows):
    payload = [[char, list(pat)] for char, pat in rows]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_flat_candidates():
    flats = []
    with (PROBE / "ranked.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["class"] != "flat":
                continue
            char = row["char"]
            if char in LINE_ANCHOR_SHADE_EXCLUDE:
                continue
            if char == "\u2588":
                continue
            pref = block_pref(char)
            if pref > 3:
                continue
            if char in MUDDY_SHADE_SKIP:
                continue
            pat = tuple(float(row[k]) for k in ("tl", "tr", "bl", "br"))
            if max(pat) - min(pat) > FLAT_SPREAD:
                continue
            mean = sum(pat) / 4.0
            flats.append((char, mean, pat, pref))
    return flats


def _paint_mean(char):
    # Lookup levels must match 8x8 tile brightness, not probe ink on thin strokes.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from core.image.ascii import _tile_for_char

    tile = _tile_for_char(char)
    return float(tile.mean()) / 255.0


def _try_add(entries, levels, seen, char, level, level_eps):
    if char in seen or char in LINE_ANCHOR_SHADE_EXCLUDE:
        return False
    if any(abs(level - existing) < level_eps for existing in levels):
        return False
    entries.append((char, level))
    levels.append(level)
    seen.add(char)
    return True


def build_shade_entries(flats, level_eps):
    """Shade by painted 8x8 tile luma (ink=light for Discord/PNG). No thin box-drawing."""
    entries = [(" ", 0.0)]
    levels = [0.0]
    seen = {" "}
    rejected_thin = 0

    scored = []
    for char, probe_mean, _pat, pref in flats:
        if char in seen or char == "\u2588":
            continue
        # Box drawing (pref=1) is thin strokes — looks black in 8x8 tiles.
        if pref == 1:
            rejected_thin += 1
            continue
        paint = _paint_mean(char)
        if paint < 0.02:
            rejected_thin += 1
            continue
        # Probe said mid, tile paints near-black → skip (┼┋╎ class).
        if probe_mean >= 0.12 and paint < probe_mean * 0.45:
            rejected_thin += 1
            continue
        if abs(paint - probe_mean) > 0.28 and paint < probe_mean:
            rejected_thin += 1
            continue
        scored.append((char, paint, pref))

    # Block elements first, then braille/geo gap-fill — ordered by paint brightness.
    for pref_max, pref_min in ((0, 0), (3, 2)):
        batch = sorted(
            [t for t in scored if pref_min <= t[2] <= pref_max and t[0] not in seen],
            key=lambda t: (t[1], t[2], t[0]),
        )
        for char, paint, _pref in batch:
            _try_add(entries, levels, seen, char, paint, level_eps)

    entries = [(c, lv) for c, lv in entries if c != "\u2588"]
    entries.append(("\u2588", max(1.0, _paint_mean("\u2588"))))
    # Force █ sentinel at 1.0 for closed range even if tile is ~0.78.
    entries[-1] = ("\u2588", 1.0)
    entries.sort(key=lambda t: t[1])
    build_shade_entries.last_rejected_thin = rejected_thin
    return entries


def shade_entries_to_ramp(entries):
    """Symbol i owns (level[i-1], level[i]]; first owns [0, level0]; █ closes at 1.0."""
    ramp = []
    for i, (char, level) in enumerate(entries):
        if i == 0:
            lo = 0.0
            hi = float(level)
        else:
            lo = float(entries[i - 1][1])
            hi = float(level)
        ramp.append(
            {
                "char": char,
                "level": float(level),
                "lo": lo,
                "hi": hi,
            }
        )
    if ramp:
        ramp[0]["lo"] = 0.0
        ramp[-1]["hi"] = 1.0
        if ramp[-1]["char"] == "\u2588":
            ramp[-1]["level"] = 1.0
    return ramp


def drop_line_shade_collisions(line_rows, shade_chars):
    kept = []
    dropped = []
    for char, pat in line_rows:
        if char in LINE_ANCHORS_KEEP:
            kept.append((char, pat))
            continue
        if char in shade_chars:
            dropped.append(char)
            continue
        kept.append((char, pat))
    return kept, dropped


def details_too_close(pat, mean, patterns, means, spreads):
    spread = max(pat) - min(pat)
    for ep, em, es in zip(patterns, means, spreads):
        dist = l2(pat, ep)
        mean_delta = abs(mean - em)
        if dist < DETAILS_MIN_L2:
            return True
        # Near-flat pairs with tiny gray delta — drop (same idea as shade LEVEL_EPS).
        if mean_delta < DETAILS_MIN_MEAN_DELTA and max(spread, es) <= FLAT_SPREAD:
            return True
    return False


def build_details(soft_path):
    rejected = {}
    kept = []
    for char, pat in load_json_pairs(soft_path):
        reason = reject_details(char)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        if block_pref(char) > 5:
            rejected["decorative"] = rejected.get("decorative", 0) + 1
            continue
        mean = sum(pat) / 4.0
        spread = max(pat) - min(pat)
        kept.append((char, pat, mean, spread, block_pref(char)))

    final = []
    patterns = []
    means = []
    spreads = []
    for char, pat, mean, spread, pref in sorted(
        kept, key=lambda t: (t[4], abs(t[2] - 0.45), -t[3])
    ):
        if details_too_close(pat, mean, patterns, means, spreads):
            rejected["dedup"] = rejected.get("dedup", 0) + 1
            continue
        final.append((char, pat))
        patterns.append(pat)
        means.append(mean)
        spreads.append(spread)
    return final, rejected


def write_shade_ramp(path, ramp):
    path.write_text(json.dumps(ramp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pack_variant(flats, level_eps, soft_path, line_rows):
    shade_entries = build_shade_entries(flats, level_eps)
    ramp = shade_entries_to_ramp(shade_entries)
    shade_chars = {item["char"] for item in ramp}
    line_kept, line_dropped = drop_line_shade_collisions(line_rows, shade_chars)
    details, details_rejected = build_details(soft_path)
    return {
        "level_eps": level_eps,
        "shade_count": len(ramp),
        "shade_chars": "".join(item["char"] for item in ramp),
        "shade_levels": [item["level"] for item in ramp],
        "shade_ramp": ramp,
        "line_measured_count": len(line_kept),
        "line_dropped_shade_collisions": line_dropped,
        "details_count": len(details),
        "details_rejected": details_rejected,
        "details": details,
        "line_rows": line_kept,
    }


def main():
    flats = load_flat_candidates()
    line_rows = load_json_pairs(LINE_PATH)
    soft_path = PROBE / "soft_patterns.json"

    variants = {}
    for eps in LEVEL_EPS_SWEEP:
        variants[f"{eps:.3f}"] = pack_variant(flats, eps, soft_path, line_rows)

    primary = variants[f"{LEVEL_EPS:.3f}"]
    write_shade_ramp(SHADE_RAMP_PATH, primary["shade_ramp"])
    write_json_pairs(LINE_PATH, primary["line_rows"])
    write_json_pairs(DETAILS_PATH, primary["details"])

    report = {
        "level_eps": LEVEL_EPS,
        "level_eps_sweep": list(LEVEL_EPS_SWEEP),
        "shade_count": primary["shade_count"],
        "shade_chars": primary["shade_chars"],
        "shade_levels": primary["shade_levels"],
        "shade_sample_ranges": [
            {
                "char": item["char"],
                "level": item["level"],
                "lo": item["lo"],
                "hi": item["hi"],
            }
            for item in primary["shade_ramp"][:6] + primary["shade_ramp"][-3:]
        ],
        "line_measured_count": primary["line_measured_count"],
        "line_dropped_shade_collisions": primary["line_dropped_shade_collisions"],
        "details_count": primary["details_count"],
        "details_rejected": primary["details_rejected"],
        "details_min_l2": DETAILS_MIN_L2,
        "details_min_mean_delta": DETAILS_MIN_MEAN_DELTA,
        "assets": {
            "shade_ramp": str(SHADE_RAMP_PATH.relative_to(ROOT)),
            "line_patterns": str(LINE_PATH.relative_to(ROOT)),
            "details_patterns": str(DETAILS_PATH.relative_to(ROOT)),
        },
        "variants": {
            key: {
                "level_eps": pack["level_eps"],
                "shade_count": pack["shade_count"],
                "shade_chars": pack["shade_chars"],
                "shade_levels": pack["shade_levels"],
                "shade_ramp": pack["shade_ramp"],
                "line_measured_count": pack["line_measured_count"],
                "line_dropped_shade_collisions": pack["line_dropped_shade_collisions"],
                "details_count": pack["details_count"],
                "details_rejected": pack["details_rejected"],
            }
            for key, pack in variants.items()
        },
    }
    report_path = PROBE / "pack_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for key, pack in variants.items():
        variant_path = PROBE / f"pack_report_eps_{key.replace('.', '_')}.json"
        variant_payload = {
            "level_eps": pack["level_eps"],
            "shade_count": pack["shade_count"],
            "shade_chars": pack["shade_chars"],
            "shade_levels": pack["shade_levels"],
            "shade_ramp": pack["shade_ramp"],
            "line_measured_count": pack["line_measured_count"],
            "line_dropped_shade_collisions": pack["line_dropped_shade_collisions"],
            "details_count": pack["details_count"],
            "details_rejected": pack["details_rejected"],
            "primary": key == f"{LEVEL_EPS:.3f}",
        }
        variant_path.write_text(
            json.dumps(variant_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        f"LEVEL_EPS={LEVEL_EPS} shade={primary['shade_count']} "
        f"line_measured={primary['line_measured_count']} "
        f"(dropped {len(primary['line_dropped_shade_collisions'])}) "
        f"details={primary['details_count']}"
    )
    print(f"  {SHADE_RAMP_PATH}")
    print(f"  {LINE_PATH}")
    print(f"  {DETAILS_PATH}")
    print(f"  {report_path}")
    for item in primary["shade_ramp"]:
        print(
            f"  shade {item['char']!r} level={item['level']:.4f} "
            f"lo={item['lo']:.4f} hi={item['hi']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
