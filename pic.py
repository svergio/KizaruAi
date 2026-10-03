import argparse

from kizaru.pic import render_pic, resolve_mode


def parse_args():
    parser = argparse.ArgumentParser(description="Turn a picture into a sticker mosaic")
    parser.add_argument("image")
    parser.add_argument("--mode", default="simple", help="1/line or 2/simple")
    parser.add_argument("--cells", type=int, default=36)
    return parser.parse_args()


def main():
    args = parse_args()
    print(render_pic(args.image, mode=resolve_mode(args.mode), cells=args.cells))


if __name__ == "__main__":
    main()
