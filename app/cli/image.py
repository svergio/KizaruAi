import argparse
from pathlib import Path

from core.bus.broker import DEFAULT_DB, emit_job_result, enqueue_and_wait
from core.bus.queue import JobStore
from core.image.ascii import CLI_DEFAULT_CELLS, CLI_DEFAULT_SCALE, resolve_mode
from core.bus.routing import build_pic_payload, pic_mode_name_to_key

DATA_DIR = Path("data")
IMPORT_DIR = DATA_DIR / "import"
EXPORT_DIR = DATA_DIR / "export"


def _add_common_render_args(parser):
    parser.add_argument("image", help="Source path or filename under data/import/")
    parser.add_argument(
        "--mode",
        default="line",
        help="line|simple (aliases 1|2)",
    )
    parser.add_argument(
        "--cells",
        type=int,
        default=None,
        help=f"Grid width in symbols (default {CLI_DEFAULT_CELLS}, mode 3 / anim)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for PNG/GIF from worker (mode 3 / anim)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=None,
        help=f"CLI line scale for mode 3 (default {CLI_DEFAULT_SCALE})",
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

    still = sub.add_parser("still", help="Still image (pic_still; mode 3 line -> PNG)")
    _add_common_render_args(still)

    anim = sub.add_parser("anim", help="Animation -> GIF via pic_video worker")
    _add_common_render_args(anim)

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


def _run_still(args, store: JobStore):
    image = _resolve_image(args.image)
    if not image.exists():
        raise SystemExit(f"source not found: {image}")
    data = image.read_bytes()
    mode_name = resolve_mode(args.mode)
    if mode_name == "line":
        scale = CLI_DEFAULT_SCALE if args.scale is None else args.scale
        cells = CLI_DEFAULT_CELLS if args.cells is None else args.cells
        payload = build_pic_payload(
            "3",
            data,
            image.name,
            scale=scale,
            cells=cells,
            animated=False,
        )
        out = Path(args.out) if args.out else _default_file_out(image, mode_name, scale, gif=False)
    else:
        payload = build_pic_payload(
            pic_mode_name_to_key(args.mode),
            data,
            image.name,
        )
        out = None
    result = enqueue_and_wait(store, "pic", payload)
    emit_job_result(result, out_path=out)


def _run_anim(args, store: JobStore):
    image = _resolve_image(args.image)
    if not image.exists():
        raise SystemExit(f"source not found: {image}")
    data = image.read_bytes()
    scale = CLI_DEFAULT_SCALE if args.scale is None else args.scale
    cells = CLI_DEFAULT_CELLS if args.cells is None else args.cells
    mode_name = resolve_mode(args.mode)
    if mode_name != "line":
        raise SystemExit("anim supports --mode line only (Discord mode 3)")
    payload = build_pic_payload(
        "3",
        data,
        image.name,
        scale=scale,
        cells=cells,
        animated=True,
    )
    if args.out:
        out = Path(args.out)
        if out.suffix.lower() != ".gif":
            out = out.with_suffix(".gif")
    else:
        out = _default_file_out(image, mode_name, scale, gif=True)
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
