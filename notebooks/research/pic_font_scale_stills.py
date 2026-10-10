"""Font / scale stills without upscale (image shrinks with scale).

Source: data/import/dorako_malfoy.jpg (PluviumG / X).

Test A: scale=8 fixed, fonts 4 6 8 12 16 24 — 3 modes
Test B: PAIRS font=2*scale (smile_src matches glyph tile):
  (scale, font) = (4,8) (6,12) (8,16) (12,24) (14,28) (16,32)

No upscale: cols = src_w // (2*scale). PNG = cols * font.

  python notebooks/research/pic_font_scale_stills.py
  python notebooks/research/pic_font_scale_stills.py --only b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.image.ascii import load_rgb, render_png, resolve_frame

SRC = ROOT / "data" / "import" / "dorako_malfoy.jpg"
OUT = ROOT / "data" / "export" / "pic_font_scale"
MODES = ("line", "render", "details")

TEST_A_FONTS = (4, 6, 8, 12, 16, 24)
TEST_A_SCALE = 8
# Paired: font == smile_src == 2*scale (1:1 source block → glyph tile).
TEST_B_PAIRS = (
    (4, 8),
    (6, 12),
    (8, 16),
    (12, 24),
    (14, 28),
    (16, 32),
)


def _run(stem: str, scale: int, font: int, mode: str, rgb, out_root: Path) -> dict:
    cols, rows = resolve_frame(rgb.shape[1], rgb.shape[0], None, scale, False, False)
    png = render_png(
        rgb,
        mode=mode,
        cells=None,
        discord=False,
        scale=scale,
        upscale=False,
        smile_px=font,
    )
    folder = out_root / stem / f"s{scale}_f{font}"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{mode}.png"
    png.save(dest)
    meta = {
        "mode": mode,
        "scale": scale,
        "font": font,
        "cols": cols,
        "rows": rows,
        "smile_src": 2 * scale,
        "png": f"{png.width}x{png.height}",
        "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
    }
    print(
        f"  [{mode}] scale={scale} font={font} "
        f"grid={cols}x{rows} smile_src={2*scale} → {png.width}x{png.height}"
    )
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("a", "b", "all"), default="all")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if not SRC.is_file():
        raise SystemExit(f"missing {SRC} — download dorako_malfoy first")
    out_root = args.out if args.out.is_absolute() else ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(SRC)
    print(f"source {SRC.name} {rgb.shape[1]}x{rgb.shape[0]} upscale=False")
    rows = []

    if args.only in ("a", "all"):
        print("=== A: scale=8, fonts", TEST_A_FONTS)
        for font in TEST_A_FONTS:
            for mode in MODES:
                rows.append(_run("A_scale8", TEST_A_SCALE, font, mode, rgb, out_root))

    if args.only in ("b", "all"):
        print("=== B: pairs font=2*scale", TEST_B_PAIRS)
        for scale, font in TEST_B_PAIRS:
            for mode in MODES:
                rows.append(_run("B_pairs", scale, font, mode, rgb, out_root))

    (out_root / "manifest.json").write_text(
        json.dumps({"source": str(SRC.relative_to(ROOT)).replace("\\", "/"), "rows": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print("wrote", out_root, "n=", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
