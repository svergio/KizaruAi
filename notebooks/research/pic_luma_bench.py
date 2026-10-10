"""Color-in / gray-ref luma bench for pic line|render|details.

RGB goes into the encoder; Rec.601 gray of the original is the metric reference.
Glyph PNG → gray → bicubic to source size → MSE/MAE vs ref.

Sweeps shade-ramp LEVEL_EPS variants from unicode_quad_probe pack reports
(0.04 / 0.055 / 0.07) by hot-swapping ascii shade globals (CPU path only).

From repo root:
  python notebooks/research/pic_luma_bench.py
  python notebooks/research/pic_luma_bench.py --skip-geo --eps 0.055
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.image.ascii as ascii_mod  # noqa: E402
from core.image.ascii import (  # noqa: E402
    DETAILS_THR,
    LINE_PAINT_CHARS,
    LINE_THR,
    OUTPUT_SMILE_PX,
    _DETAILS_COUNT,
    _MEASURED_COUNT,
    _SHADE_COUNT,
    _luminance,
    indexes_to_image,
    load_rgb,
    pic_indexes,
)

OUT_DIR = ROOT / "data" / "export" / "pic_luma_bench"
PROBE = ROOT / "data" / "export" / "unicode_quad_probe"
GOLD_COUPLE2_LINE = ROOT / "data" / "export" / "couple2" / "line_s2_c160.png"

DEFAULT_JOBS = (
    {
        "stem": "couple2",
        "path": ROOT / "data" / "import" / "couple2.jpg",
        "cells": 160,
        "scale": 2,
    },
    {
        "stem": "gwyndolin",
        "path": ROOT / "data" / "import" / "gwyndolin.jpg",
        "cells": 200,
        "scale": 2,
    },
    {
        "stem": "cyber_rooftop",
        "path": ROOT / "data" / "import" / "cyber_rooftop.jpg",
        "cells": 160,
        "scale": 2,
    },
    {
        "stem": "geo_shapes",
        "path": ROOT / "data" / "import" / "geo_shapes.png",
        "cells": 160,
        "scale": 2,
        "optional": True,
    },
)

MODES = ("line", "render", "details")
LEVEL_EPS_DEFAULTS = (0.04, 0.055, 0.07)
DETAILS_THR_SWEEP = (0.06, 0.08, 0.10, 0.16)
LINE_ANCHORS_KEEP = frozenset({"\u259b", "\u2580"})  # ▛ ▀


def _pack_report_path(eps: float) -> Path:
    key = f"{eps:.3f}".replace(".", "_")
    return PROBE / f"pack_report_eps_{key}.json"


def _load_variant_ramp(eps: float) -> list[dict]:
    path = _pack_report_path(eps)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    ramp = payload.get("shade_ramp")
    if not ramp:
        raise ValueError(f"no shade_ramp in {path}")
    return ramp


def _rebuild_line_globals(shade_spec, shade_hi, measured_spec, details_spec):
    ascii_mod._SHADE_SPEC = tuple(shade_spec)
    ascii_mod._SHADE_HI = np.asarray(shade_hi, dtype=np.float32)
    ascii_mod._MEASURED_LINE_SPEC = tuple(measured_spec)
    ascii_mod._DETAILS_SPEC = tuple(details_spec)
    ascii_mod.DETAILS_CHARS = np.array([char for char, _ in details_spec])
    ascii_mod.DETAILS_PATTERNS = np.array(
        [pattern for _, pattern in details_spec], dtype=np.float32
    )
    ascii_mod._DETAILS_COUNT = len(details_spec)
    line_spec = tuple(
        (char, (level, level, level, level)) for char, level in shade_spec
    ) + tuple(measured_spec)
    ascii_mod._LINE_SPEC = line_spec
    ascii_mod.LINE_CHARS = np.array([char for char, _ in line_spec])
    ascii_mod.LINE_PATTERNS = np.array(
        [pattern for _, pattern in line_spec], dtype=np.float32
    )
    ascii_mod.LINE_COLORS = np.repeat(
        ascii_mod.LINE_PATTERNS.mean(axis=1)[:, None] * 255.0, 3, axis=1
    )
    ascii_mod._SHADE_COUNT = len(shade_spec)
    ascii_mod._MEASURED_COUNT = len(measured_spec)
    ascii_mod._DETAILS_BASE = ascii_mod._SHADE_COUNT + ascii_mod._MEASURED_COUNT
    ascii_mod.LINE_PAINT_CHARS = np.concatenate(
        [ascii_mod.LINE_CHARS, ascii_mod.DETAILS_CHARS]
    )
    ascii_mod.LINE_PAINT_PATTERNS = np.concatenate(
        [ascii_mod.LINE_PATTERNS, ascii_mod.DETAILS_PATTERNS], axis=0
    )
    ascii_mod.MEASURED_PATTERNS = ascii_mod.LINE_PATTERNS[
        ascii_mod._SHADE_COUNT : ascii_mod._DETAILS_BASE
    ]
    ascii_mod._GLYPH_ATLAS.clear()


def apply_shade_ramp(ramp: list[dict]) -> None:
    """Hot-swap shade ramp + rebuild shade/measured/details paint tables."""
    if not ramp:
        raise ValueError("empty shade ramp")
    shade_spec = []
    his = []
    for item in ramp:
        shade_spec.append((item["char"], float(item["level"])))
        his.append(float(item["hi"]))
    his[-1] = 1.0
    shade_spec[-1] = (shade_spec[-1][0], 1.0)
    his = np.asarray(his, dtype=np.float32)
    if np.any(np.diff(his) < 0):
        raise ValueError("shade ramp hi must be non-decreasing")

    shade_chars = {char for char, _ in shade_spec}
    measured = []
    for char, pattern in ascii_mod._load_measured_line_patterns():
        if char in LINE_ANCHORS_KEEP:
            measured.append((char, pattern))
            continue
        if char in shade_chars:
            continue
        measured.append((char, pattern))
    details = ascii_mod._load_details_patterns()
    _rebuild_line_globals(shade_spec, his, measured, details)


@contextmanager
def shade_ramp_context(ramp: list[dict] | None):
    """Apply ramp for the duration; restore module dict tables afterward."""
    snapshot = {
        "shade_spec": ascii_mod._SHADE_SPEC,
        "shade_hi": ascii_mod._SHADE_HI.copy(),
        "measured": ascii_mod._MEASURED_LINE_SPEC,
        "details": ascii_mod._DETAILS_SPEC,
    }
    try:
        if ramp is not None:
            apply_shade_ramp(ramp)
        yield
    finally:
        _rebuild_line_globals(
            snapshot["shade_spec"],
            snapshot["shade_hi"],
            snapshot["measured"],
            snapshot["details"],
        )


def image_to_gray01(image: Image.Image) -> np.ndarray:
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def upscale_bicubic(gray: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    frame = Image.fromarray(np.clip(gray * 255.0, 0, 255).astype(np.uint8), mode="L")
    frame = frame.resize(size_wh, Image.BICUBIC)
    return np.asarray(frame, dtype=np.float32) / 255.0


def mse_mae(ref: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    diff = pred.astype(np.float64) - ref.astype(np.float64)
    return float(np.mean(diff * diff)), float(np.mean(np.abs(diff)))


def ssim_luma(ref: np.ndarray, pred: np.ndarray) -> float | None:
    a = ref.astype(np.float64)
    b = pred.astype(np.float64)
    if a.shape != b.shape or a.size < 16:
        return None
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = a.mean()
    mu_b = b.mean()
    var_a = a.var()
    var_b = b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    if den <= 0.0:
        return None
    return float(num / den)


def diff_overlay(ref: np.ndarray, pred: np.ndarray) -> Image.Image:
    err = np.abs(pred.astype(np.float32) - ref.astype(np.float32))
    scale = float(np.percentile(err, 99)) if err.size else 1.0
    if scale < 1e-6:
        scale = 1.0
    heat = np.clip(err / scale, 0.0, 1.0)
    base = np.stack([ref, ref, ref], axis=-1)
    tint = np.stack([heat, np.zeros_like(heat), np.zeros_like(heat)], axis=-1)
    mix = np.clip(base * (1.0 - 0.55 * heat[..., None]) + tint * 0.9, 0.0, 1.0)
    return Image.fromarray((mix * 255.0).astype(np.uint8), mode="RGB")


def render_glyph_png(
    source,
    *,
    mode: str,
    cells: int,
    scale: int,
    line_thr: float,
    details_thr: float | None,
) -> Image.Image:
    indexes, resolved = pic_indexes(
        source,
        mode=mode,
        cells=cells,
        discord=False,
        form=True,
        scale=scale,
        upscale=True,
        line_thr=line_thr,
        details_thr=details_thr,
    )
    return indexes_to_image(indexes, mode=resolved, chars=LINE_PAINT_CHARS)


def side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    left = left.convert("L")
    right = right.convert("L")
    h = max(left.height, right.height)
    canvas = Image.new("L", (left.width + right.width + 8, h), 32)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + 8, 0))
    return canvas


def run_case(
    rgb: np.ndarray,
    ref: np.ndarray,
    *,
    stem: str,
    mode: str,
    cells: int,
    scale: int,
    level_eps: float,
    line_thr: float,
    details_thr: float | None,
    out_dir: Path,
    save_artifacts: bool,
) -> dict:
    thr = DETAILS_THR if details_thr is None else float(details_thr)
    label = f"{mode}_eps{level_eps:.3f}"
    if mode == "details":
        label = f"{label}_dthr{thr:.2f}"
    t0 = time.perf_counter()
    glyph = render_glyph_png(
        rgb,
        mode=mode,
        cells=cells,
        scale=scale,
        line_thr=line_thr,
        details_thr=thr if mode == "details" else None,
    )
    recon_native = image_to_gray01(glyph)
    h, w = ref.shape[:2]
    recon = upscale_bicubic(recon_native, (w, h))
    mse, mae = mse_mae(ref, recon)
    ssim = ssim_luma(ref, recon)
    elapsed = time.perf_counter() - t0

    glyph_rel = recon_rel = diff_rel = None
    if save_artifacts:
        if mode == "details" and abs(thr - DETAILS_THR) > 1e-9:
            base = f"{mode}_dthr{thr:.2f}".replace(".", "p")
        else:
            base = mode
        glyph_path = out_dir / f"{base}.png"
        recon_path = out_dir / f"{base}_recon.png"
        diff_path = out_dir / f"{base}_diff.png"
        glyph.save(glyph_path)
        Image.fromarray(np.clip(recon * 255.0, 0, 255).astype(np.uint8), mode="L").save(
            recon_path
        )
        diff_overlay(ref, recon).save(diff_path)
        glyph_rel = str(glyph_path.relative_to(ROOT)).replace("\\", "/")
        recon_rel = str(recon_path.relative_to(ROOT)).replace("\\", "/")
        diff_rel = str(diff_path.relative_to(ROOT)).replace("\\", "/")

    return {
        "stem": stem,
        "mode": mode,
        "level_eps": level_eps,
        "details_thr": thr if mode == "details" else None,
        "line_thr": line_thr,
        "cells": cells,
        "scale": scale,
        "label": label,
        "mse": mse,
        "mae": mae,
        "ssim": ssim,
        "seconds": elapsed,
        "glyph_wh": [glyph.width, glyph.height],
        "shade_count": int(ascii_mod._SHADE_COUNT),
        "measured_count": int(ascii_mod._MEASURED_COUNT),
        "details_count": int(ascii_mod._DETAILS_COUNT),
        "smile_px": OUTPUT_SMILE_PX,
        "glyph_png": glyph_rel,
        "recon_png": recon_rel,
        "diff_png": diff_rel,
    }


def pick_jobs(skip_geo: bool) -> list[dict]:
    jobs = []
    for job in DEFAULT_JOBS:
        if job.get("optional") and skip_geo:
            continue
        if not job["path"].is_file():
            if job.get("optional"):
                continue
            raise FileNotFoundError(job["path"])
        jobs.append(job)
    return jobs


def rank_modes(rows: list[dict], stem: str, level_eps: float) -> list[dict]:
    subset = [
        r
        for r in rows
        if r["stem"] == stem
        and abs(float(r["level_eps"]) - level_eps) < 1e-9
        and (
            r["mode"] != "details"
            or r["details_thr"] is None
            or abs(float(r["details_thr"]) - DETAILS_THR) < 1e-9
        )
    ]
    return sorted(subset, key=lambda r: float(r["mse"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-geo", action="store_true")
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=list(LEVEL_EPS_DEFAULTS),
        help="LEVEL_EPS shade variants to load from pack_report_eps_*.json",
    )
    parser.add_argument("--line-thr", type=float, default=LINE_THR)
    parser.add_argument(
        "--details-thr-sweep",
        type=float,
        nargs="+",
        default=list(DETAILS_THR_SWEEP),
    )
    parser.add_argument(
        "--primary-eps",
        type=float,
        default=0.055,
        help="eps folder that gets full PNG artifacts (others: metrics only)",
    )
    parser.add_argument("--no-diff", action="store_true")
    args = parser.parse_args(argv)

    out_root = args.out if args.out.is_absolute() else ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = pick_jobs(skip_geo=args.skip_geo)
    eps_list = tuple(float(x) for x in args.eps)
    thr_sweep = tuple(float(x) for x in args.details_thr_sweep)
    primary_eps = float(args.primary_eps)
    line_thr = float(args.line_thr)

    all_rows: list[dict] = []
    for job in jobs:
        stem = job["stem"]
        src_path = job["path"]
        cells = int(job["cells"])
        scale = int(job["scale"])
        stem_dir = out_root / stem
        stem_dir.mkdir(parents=True, exist_ok=True)

        rgb = load_rgb(src_path)
        ref = np.clip(_luminance(rgb), 0.0, 1.0).astype(np.float32)
        h, w = ref.shape[:2]
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").save(
            stem_dir / "source.png"
        )
        Image.fromarray(np.clip(ref * 255.0, 0, 255).astype(np.uint8), mode="L").save(
            stem_dir / "ref_gray.png"
        )

        for eps in eps_list:
            ramp = _load_variant_ramp(eps)
            save_art = abs(eps - primary_eps) < 1e-9

            print(f"[{stem}] LEVEL_EPS={eps:.3f} shade={len(ramp)} ...", flush=True)
            with shade_ramp_context(ramp):
                for mode in MODES:
                    thr_values = thr_sweep if mode == "details" else (None,)
                    for thr in thr_values:
                        # Primary eps: full PNG/diff set. Other eps: metrics only.
                        row = run_case(
                            rgb,
                            ref,
                            stem=stem,
                            mode=mode,
                            cells=cells,
                            scale=scale,
                            level_eps=eps,
                            line_thr=line_thr,
                            details_thr=thr,
                            out_dir=stem_dir,
                            save_artifacts=save_art,
                        )
                        if args.no_diff and row.get("diff_png"):
                            diff_path = ROOT / row["diff_png"]
                            if diff_path.is_file():
                                diff_path.unlink()
                            row["diff_png"] = None
                        all_rows.append(row)
                        print(
                            f"  {row['label']}: MSE={row['mse']:.6f} "
                            f"MAE={row['mae']:.6f} SSIM={row['ssim']} "
                            f"{row['seconds']:.1f}s",
                            flush=True,
                        )

        if stem == "couple2" and GOLD_COUPLE2_LINE.is_file():
            gold = Image.open(GOLD_COUPLE2_LINE)
            cur = stem_dir / "line.png"
            if cur.is_file():
                side_by_side(gold, Image.open(cur)).save(
                    stem_dir / "eye_gold_vs_line.png"
                )

        ranked = rank_modes(all_rows, stem, primary_eps)
        (stem_dir / "ranking_primary_eps.json").write_text(
            json.dumps(ranked, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    csv_path = out_root / "metrics.csv"
    fields = [
        "stem",
        "mode",
        "level_eps",
        "details_thr",
        "line_thr",
        "cells",
        "scale",
        "mse",
        "mae",
        "ssim",
        "seconds",
        "shade_count",
        "measured_count",
        "details_count",
        "glyph_png",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k) for k in fields})

    # Aggregate: mean MSE by (level_eps, mode) on photo stems only.
    photo_stems = {"couple2", "gwyndolin"}
    agg: dict[tuple, list[float]] = {}
    for row in all_rows:
        if row["stem"] not in photo_stems:
            continue
        if row["mode"] == "details" and (
            row["details_thr"] is None
            or abs(float(row["details_thr"]) - DETAILS_THR) >= 1e-9
        ):
            continue
        key = (float(row["level_eps"]), row["mode"])
        agg.setdefault(key, []).append(float(row["mse"]))
    mean_by = {
        f"eps{eps:.3f}/{mode}": float(np.mean(mses))
        for (eps, mode), mses in sorted(agg.items())
    }

    details_agg: dict[float, list[float]] = {}
    for row in all_rows:
        if row["stem"] not in photo_stems or row["mode"] != "details":
            continue
        if abs(float(row["level_eps"]) - primary_eps) >= 1e-9:
            continue
        details_agg.setdefault(float(row["details_thr"]), []).append(float(row["mse"]))
    details_mean = {
        f"{thr:.2f}": float(np.mean(mses)) for thr, mses in sorted(details_agg.items())
    }
    best_details_thr = (
        min(details_agg.items(), key=lambda item: float(np.mean(item[1])))[0]
        if details_agg
        else DETAILS_THR
    )

    # Best eps by mean line MSE on photos (eye still required).
    eps_line_mean = {}
    for (eps, mode), mses in agg.items():
        if mode == "line":
            eps_line_mean[eps] = float(np.mean(mses))
    best_eps = (
        min(eps_line_mean.items(), key=lambda item: item[1])[0]
        if eps_line_mean
        else primary_eps
    )

    summary = {
        "protocol": "color RGB in; Rec.601 gray ref; glyph→gray→bicubic; MSE/MAE",
        "modes": list(MODES),
        "level_eps": list(eps_list),
        "primary_eps": primary_eps,
        "line_thr": line_thr,
        "details_thr_default": float(DETAILS_THR),
        "details_thr_sweep": list(thr_sweep),
        "dict_counts_runtime": {
            "shade": _SHADE_COUNT,
            "line_measured": _MEASURED_COUNT,
            "details": _DETAILS_COUNT,
        },
        "jobs": [
            {
                "stem": j["stem"],
                "source": str(j["path"].relative_to(ROOT)).replace("\\", "/"),
                "cells": j["cells"],
                "scale": j["scale"],
            }
            for j in jobs
        ],
        "mean_mse_photos_by_eps_mode": mean_by,
        "mean_mse_details_thr_primary_eps": details_mean,
        "mse_rank_hint": {
            "best_level_eps_by_line_mse": best_eps,
            "best_details_thr_by_mse": best_details_thr,
            "note": "MSE ranks candidates; eye vs Oct-7 couple2/line_s2_c160.png is required",
        },
        "gold_eye": str(GOLD_COUPLE2_LINE.relative_to(ROOT)).replace("\\", "/")
        if GOLD_COUPLE2_LINE.is_file()
        else None,
        "rows": all_rows,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["mse_rank_hint"], indent=2), flush=True)
    print(f"wrote {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
