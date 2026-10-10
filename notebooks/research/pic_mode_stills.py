"""Export clear stills for line / render / details (no diff).

Sampling (unchanged core):
  - color original → optional nearest upscale so grid fits
  - each smile = 2x2 quads; each quad = mean of scale x scale source px
  - so one glyph covers (2*scale)^2 source pixels

Stages:
  1_line     — Rec.601 + line/shade
  2_render   — OKLab L* + color edges
  3_details  — OKLab L* + soft 2x2 details dict

From repo root:
  python notebooks/research/pic_mode_stills.py
  python notebooks/research/pic_mode_stills.py --scales 1 2 3 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.image.ascii import (
    CLI_DEFAULT_CELLS,
    CLI_DEFAULT_SCALE,
    LINE_PAINT_CHARS,
    OUTPUT_SMILE_PX,
    indexes_to_image,
    load_rgb,
    pic_indexes,
)

OUT_DIR = ROOT / "data" / "export" / "pic_modes"

STAGES = (
    ("1_line", "line"),
    ("2_render", "render"),
    ("3_details", "details"),
)

# CLI default cells=320 → PNG width 320*8=2560 (not the old gold c160=1280).
JOBS = (
    {"stem": "couple2", "path": ROOT / "data" / "import" / "couple2.jpg"},
    {"stem": "gwyndolin", "path": ROOT / "data" / "import" / "gwyndolin.jpg"},
    {"stem": "cyber_rooftop", "path": ROOT / "data" / "import" / "cyber_rooftop.jpg"},
    {"stem": "geo_shapes", "path": ROOT / "data" / "import" / "geo_shapes.png"},
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--cells",
        type=int,
        default=CLI_DEFAULT_CELLS,
        help=f"smile columns (CLI default {CLI_DEFAULT_CELLS} → width {CLI_DEFAULT_CELLS * OUTPUT_SMILE_PX})",
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[CLI_DEFAULT_SCALE],
        help="source px per quadrant (smile src = 2*scale; scale=4 → 8x8 src → 8x8 glyph)",
    )
    parser.add_argument(
        "--stems",
        nargs="+",
        default=None,
        help="subset of jobs, e.g. geo_shapes",
    )
    args = parser.parse_args()
    out_root = args.out if args.out.is_absolute() else ROOT / args.out
    cells = max(8, int(args.cells))
    scales = [max(1, int(s)) for s in args.scales]
    expect_w = cells * OUTPUT_SMILE_PX
    want = set(args.stems) if args.stems else None

    for job in JOBS:
        path = job["path"]
        stem = job["stem"]
        if want is not None and stem not in want:
            continue
        if not path.is_file():
            print("skip missing", path)
            continue
        rgb = load_rgb(path)
        src = np.clip(rgb, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb

        for scale in scales:
            out = out_root / stem / f"s{scale}" if len(scales) > 1 else out_root / stem
            out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(src, mode="RGB").save(out / "0_source.png")
            print(
                f"[{stem}] cells={cells} scale={scale} "
                f"(quad={scale}px, smile_src={2 * scale}px, png_w={expect_w})"
            )
            for file_stem, mode in STAGES:
                indexes, resolved = pic_indexes(
                    rgb,
                    mode=mode,
                    cells=cells,
                    discord=False,
                    form=True,
                    scale=scale,
                    upscale=True,
                )
                png = indexes_to_image(indexes, mode=resolved, chars=LINE_PAINT_CHARS)
                if png.width != expect_w:
                    raise RuntimeError(
                        f"{mode}: got {png.width}x{png.height}, expected width {expect_w}"
                    )
                dest = out / f"{file_stem}.png"
                png.save(dest)
                print(f"  {dest.relative_to(out_root)}  {png.width}x{png.height}")
    print("wrote", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
