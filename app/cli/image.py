import argparse
import sys
from pathlib import Path

from core.bus.broker import DEFAULT_DB, emit_job_result, enqueue_and_wait
from core.bus.queue import JobStore
from core.image.ascii import CLI_DEFAULT_CELLS, CLI_DEFAULT_SCALE, resolve_mode
from core.bus.routing import PIC_MODE_KEYS, build_pic_payload

DATA_DIR = Path("data")
IMPORT_DIR = DATA_DIR / "import"
EXPORT_DIR = DATA_DIR / "export"


def _add_common_args(parser):
    parser.add_argument("image", help="Source path or filename under data/import/")
    parser.add_argument(
        "--mode",
        default="line",
        help="simple|line|render|details (aliases 1|2|3|4)",
    )
    parser.add_argument(
        "--cells",
        type=int,
        default=None,
        help=f"Grid width in symbols (default {CLI_DEFAULT_CELLS}, file modes)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for PNG/GIF (line|render|details; simple --out writes PNG)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=None,
        help=f"Source px per smile quadrant (default {CLI_DEFAULT_SCALE})",
    )
    parser.add_argument(
        "--font",
        "--smile-px",
        dest="smile_px",
        type=int,
        default=None,
        help="Output glyph tile size in px (default 8). PNG width = cols * font",
    )
    parser.add_argument(
        "--no-upscale",
        action="store_true",
        help="Do not upscale source to fill cells; cols shrink to width//(2*scale)",
    )
    parser.add_argument(
        "--dark",
        default=None,
        help="simple: shadow glyph (default black square emoji)",
    )
    parser.add_argument(
        "--light",
        default=None,
        help="simple: highlight glyph (default white square emoji)",
    )
    parser.add_argument(
        "--line-thr",
        type=float,
        default=None,
        help="line|render|details: min 2x2 spread for measured line (default 0.45)",
    )
    parser.add_argument(
        "--details-thr",
        type=float,
        default=None,
        help="details mode: min 2x2 spread for soft DETAILS_PATTERNS (default 0.08)",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Enqueue pic jobs (same queue as Discord /ai pic)"
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help="SQLite job broker (default: data/bot.db)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    still = sub.add_parser(
        "still",
        help="Still: simple=text (or PNG with --out); line|render|details=PNG",
    )
    _add_common_args(still)

    anim = sub.add_parser(
        "anim",
        help="Animation -> GIF (mode line|render|details)",
    )
    _add_common_args(anim)

    return parser.parse_args(argv)


def _resolve_image(path):
    path = Path(path)
    if path.exists():
        return path
    candidate = IMPORT_DIR / path.name
    if candidate.exists():
        return candidate
    return path


def _stem(image_path):
    return Path(image_path).stem.lstrip("_")


def _default_file_out(image_path, mode, scale, gif=False):
    root = EXPORT_DIR / _stem(image_path)
    name = f"{mode}_s{int(scale)}"
    suffix = ".gif" if gif else ".png"
    return root / "pik" / f"{name}{suffix}"


def _mode_kwargs(args, mode_name):
    kw = {}
    if mode_name == "simple":
        if args.dark is not None:
            kw["dark"] = args.dark
        if args.light is not None:
            kw["light"] = args.light
        if args.line_thr is not None or args.details_thr is not None:
            print(
                "warning: --line-thr/--details-thr apply only to line|render|details",
                file=sys.stderr,
            )
    else:
        if args.dark is not None or args.light is not None:
            print("warning: --dark/--light apply only to --mode simple", file=sys.stderr)
        if args.line_thr is not None:
            kw["line_thr"] = args.line_thr
        if args.details_thr is not None:
            if mode_name != "details":
                print(
                    "warning: --details-thr is for --mode details (ignored by line|render compose)",
                    file=sys.stderr,
                )
            kw["details_thr"] = args.details_thr
    return kw


def _run_still(args, store: JobStore):
    image = _resolve_image(args.image)
    if not image.exists():
        raise SystemExit(f"source not found: {image}")
    mode_name = resolve_mode(args.mode)
    mode_key = PIC_MODE_KEYS[mode_name]
    kw = _mode_kwargs(args, mode_name)
    if mode_name == "simple" and args.out is None:
        try:
            payload = build_pic_payload(
                mode_key,
                image.read_bytes(),
                image.name,
                as_file=False,
                **kw,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        out = None
    else:
        scale = CLI_DEFAULT_SCALE if args.scale is None else args.scale
        cells = CLI_DEFAULT_CELLS if args.cells is None else args.cells
        out = Path(args.out) if args.out else _default_file_out(image, mode_name, scale, gif=False)
        payload = build_pic_payload(
            mode_key,
            None,
            image.name,
            scale=scale,
            cells=cells,
            animated=False,
            source_path=str(image.resolve()),
            out_path=str(out.resolve()),
            as_file=True,
            smile_px=args.smile_px,
            upscale=False if args.no_upscale else None,
            **kw,
        )
    result = enqueue_and_wait(store, "pic", payload)
    emit_job_result(result, out_path=out)


def _run_anim(args, store: JobStore):
    image = _resolve_image(args.image)
    if not image.exists():
        raise SystemExit(f"source not found: {image}")
    mode_name = resolve_mode(args.mode)
    if mode_name == "simple":
        raise SystemExit("anim supports --mode line|render|details only")
    mode_key = PIC_MODE_KEYS[mode_name]
    kw = _mode_kwargs(args, mode_name)
    scale = CLI_DEFAULT_SCALE if args.scale is None else args.scale
    cells = CLI_DEFAULT_CELLS if args.cells is None else args.cells
    if args.out:
        out = Path(args.out)
        if out.suffix.lower() != ".gif":
            out = out.with_suffix(".gif")
    else:
        out = _default_file_out(image, mode_name, scale, gif=True)
    payload = build_pic_payload(
        mode_key,
        None,
        image.name,
        scale=scale,
        cells=cells,
        animated=True,
        source_path=str(image.resolve()),
        out_path=str(out.resolve()),
        as_file=True,
        **kw,
    )
    result = enqueue_and_wait(store, "pic", payload)
    emit_job_result(result, out_path=out)


def main():
    args = parse_args()
    store = JobStore(Path(args.db), recover_running=False)
    if args.command == "still":
        _run_still(args, store)
    elif args.command == "anim":
        _run_anim(args, store)
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
