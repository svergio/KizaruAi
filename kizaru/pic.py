from io import BytesIO

import numpy as np
from PIL import Image

HEADER = "Пикча:"


def _flat(level):
    return (level, level, level, level)


def _sticker(char, rgb, pattern, kind="flat"):
    return (char, tuple(rgb), tuple(pattern), kind)


def _build_stickers():
    items = []

    gray = [
        ("⬛", 8, _flat(0.04)),
        ("⚫", 16, _flat(0.08)),
        ("◼️", 28, _flat(0.12)),
        ("◼", 40, _flat(0.16)),
        ("◾", 52, _flat(0.20)),
        ("▪️", 64, _flat(0.26)),
        ("▪", 78, _flat(0.32)),
        ("●", 92, _flat(0.38)),
        ("🖤", 30, (0.12, 0.12, 0.28, 0.08)),
        ("▓", 108, _flat(0.44)),
        ("▒", 142, _flat(0.56)),
        ("░", 176, _flat(0.70)),
        ("▪", 96, (0.30, 0.42, 0.42, 0.30)),
        ("◉", 118, (0.25, 0.25, 0.25, 0.55)),
        ("◎", 150, (0.40, 0.40, 0.40, 0.75)),
        ("○", 198, (0.62, 0.62, 0.62, 0.90)),
        ("◯", 214, (0.72, 0.72, 0.72, 0.95)),
        ("⚪", 228, _flat(0.90)),
        ("▫️", 236, _flat(0.93)),
        ("◽", 226, _flat(0.88)),
        ("⬜", 248, _flat(0.98)),
        ("♡", 232, (0.78, 0.78, 0.92, 0.62)),
        ("♥", 70, (0.22, 0.22, 0.40, 0.10)),
        ("❤", 168, (0.55, 0.55, 0.75, 0.35)),
        ("🤍", 242, (0.90, 0.90, 0.98, 0.72)),
    ]
    for char, level, pattern in gray:
        items.append(_sticker(char, (level, level, level), pattern))

    lines = [
        ("▌", (70, 70, 74), (0.08, 0.82, 0.08, 0.82)),
        ("▐", (74, 70, 70), (0.82, 0.08, 0.82, 0.08)),
        ("▀", (68, 74, 70), (0.08, 0.08, 0.82, 0.82)),
        ("▄", (74, 68, 70), (0.82, 0.82, 0.08, 0.08)),
        ("▚", (80, 80, 70), (0.12, 0.78, 0.78, 0.12)),
        ("▞", (70, 80, 80), (0.78, 0.12, 0.12, 0.78)),
        ("┌", (90, 90, 96), (0.15, 0.15, 0.15, 0.85)),
        ("┐", (96, 90, 90), (0.15, 0.15, 0.85, 0.15)),
        ("└", (90, 96, 90), (0.15, 0.85, 0.15, 0.15)),
        ("┘", (100, 90, 90), (0.85, 0.15, 0.15, 0.15)),
        ("├", (86, 86, 92), (0.20, 0.55, 0.20, 0.85)),
        ("┤", (92, 86, 86), (0.55, 0.20, 0.85, 0.20)),
        ("┬", (86, 92, 86), (0.18, 0.18, 0.62, 0.62)),
        ("┴", (92, 86, 92), (0.62, 0.62, 0.18, 0.18)),
        ("┼", (88, 88, 88), (0.30, 0.55, 0.55, 0.30)),
        ("╱", (110, 110, 100), (0.85, 0.25, 0.25, 0.85)),
        ("╲", (100, 110, 110), (0.25, 0.85, 0.85, 0.25)),
        ("╳", (104, 104, 104), (0.20, 0.70, 0.70, 0.20)),
        ("┃", (48, 48, 52), (0.05, 0.70, 0.05, 0.70)),
        ("━", (52, 48, 48), (0.05, 0.05, 0.70, 0.70)),
        ("┏", (60, 60, 66), (0.10, 0.10, 0.10, 0.80)),
        ("┓", (66, 60, 60), (0.10, 0.10, 0.80, 0.10)),
        ("┗", (60, 66, 60), (0.10, 0.80, 0.10, 0.10)),
        ("┛", (66, 60, 66), (0.80, 0.10, 0.10, 0.10)),
        ("╭", (140, 140, 148), (0.35, 0.55, 0.35, 0.90)),
        ("╮", (148, 140, 140), (0.55, 0.35, 0.90, 0.35)),
        ("╰", (140, 148, 140), (0.35, 0.90, 0.35, 0.55)),
        ("╯", (148, 140, 148), (0.90, 0.35, 0.55, 0.35)),
        ("█", (12, 12, 14), _flat(0.02)),
        ("▛", (36, 36, 40), (0.05, 0.05, 0.05, 0.70)),
        ("▜", (40, 36, 36), (0.05, 0.05, 0.70, 0.05)),
        ("▙", (36, 40, 36), (0.05, 0.70, 0.05, 0.05)),
        ("▟", (40, 36, 40), (0.70, 0.05, 0.05, 0.05)),
    ]
    items.extend(_sticker(char, rgb, pattern, kind="line") for char, rgb, pattern in lines)

    colors = [
        ("🔴", (214, 36, 36), _flat(0.34)),
        ("🟥", (198, 42, 48), _flat(0.32)),
        ("❤️", (204, 28, 46), (0.28, 0.28, 0.48, 0.16)),
        ("🧡", (232, 122, 36), (0.55, 0.55, 0.72, 0.40)),
        ("🟠", (236, 140, 32), _flat(0.58)),
        ("🔶", (220, 132, 40), (0.50, 0.62, 0.62, 0.40)),
        ("🟡", (236, 206, 40), _flat(0.78)),
        ("🟨", (232, 196, 48), _flat(0.76)),
        ("💛", (236, 196, 48), (0.72, 0.72, 0.86, 0.58)),
        ("🟢", (36, 168, 64), _flat(0.48)),
        ("🟩", (40, 150, 70), _flat(0.46)),
        ("💚", (36, 156, 78), (0.40, 0.40, 0.62, 0.28)),
        ("🔵", (36, 92, 210), _flat(0.40)),
        ("🟦", (40, 96, 196), _flat(0.42)),
        ("💙", (36, 104, 204), (0.34, 0.34, 0.55, 0.22)),
        ("🟣", (146, 48, 196), _flat(0.42)),
        ("🟪", (150, 60, 186), _flat(0.44)),
        ("💜", (154, 64, 196), (0.40, 0.40, 0.60, 0.26)),
        ("🟤", (122, 72, 40), _flat(0.32)),
        ("🟫", (118, 74, 46), _flat(0.34)),
        ("🤎", (128, 78, 48), (0.32, 0.32, 0.50, 0.20)),
        ("💖", (232, 86, 146), (0.70, 0.70, 0.86, 0.50)),
        ("💗", (236, 110, 156), (0.74, 0.74, 0.88, 0.55)),
        ("💓", (220, 70, 110), (0.60, 0.60, 0.78, 0.40)),
        ("💕", (236, 130, 168), (0.78, 0.78, 0.90, 0.62)),
        ("💞", (230, 100, 150), (0.68, 0.72, 0.84, 0.50)),
        ("💘", (214, 60, 96), (0.58, 0.62, 0.74, 0.36)),
        ("💝", (196, 48, 86), (0.48, 0.48, 0.66, 0.30)),
        ("💟", (176, 90, 186), (0.62, 0.62, 0.80, 0.48)),
        ("💠", (70, 176, 196), (0.55, 0.70, 0.70, 0.45)),
        ("🔷", (46, 126, 204), (0.40, 0.55, 0.55, 0.30)),
        ("🔹", (80, 160, 220), _flat(0.62)),
        ("🔸", (220, 150, 60), _flat(0.64)),
        ("🔺", (196, 40, 40), (0.20, 0.45, 0.55, 0.55)),
        ("🔻", (176, 32, 40), (0.55, 0.55, 0.45, 0.20)),
        ("🔶", (210, 120, 36), (0.45, 0.60, 0.60, 0.35)),
        ("🔷", (36, 110, 190), (0.35, 0.55, 0.55, 0.28)),
        ("⬛", (6, 6, 8), _flat(0.03)),
        ("⬜", (250, 250, 252), _flat(0.99)),
        ("🔴", (180, 24, 32), (0.22, 0.22, 0.22, 0.50)),
        ("🟠", (200, 110, 24), (0.48, 0.48, 0.48, 0.72)),
        ("🟡", (210, 180, 30), (0.70, 0.70, 0.70, 0.88)),
        ("🟢", (24, 140, 52), (0.36, 0.36, 0.36, 0.64)),
        ("🔵", (24, 70, 180), (0.28, 0.28, 0.28, 0.58)),
        ("🟣", (120, 36, 170), (0.32, 0.32, 0.32, 0.60)),
        ("🟤", (100, 58, 32), (0.26, 0.26, 0.26, 0.48)),
        ("⭕", (190, 40, 40), (0.55, 0.55, 0.55, 0.90)),
        ("❌", (180, 40, 40), (0.20, 0.70, 0.70, 0.20)),
        ("❗", (210, 40, 40), (0.70, 0.30, 0.70, 0.20)),
        ("❓", (40, 90, 200), (0.65, 0.35, 0.55, 0.25)),
        ("⭐", (236, 190, 40), (0.70, 0.40, 0.55, 0.25)),
        ("🌟", (240, 200, 60), (0.75, 0.50, 0.60, 0.30)),
        ("✨", (230, 210, 120), (0.85, 0.60, 0.70, 0.45)),
        ("⚡", (240, 200, 40), (0.80, 0.35, 0.45, 0.20)),
        ("🔥", (230, 90, 30), (0.75, 0.45, 0.35, 0.20)),
        ("💧", (40, 140, 220), (0.70, 0.40, 0.35, 0.20)),
        ("🌊", (30, 110, 190), (0.45, 0.60, 0.35, 0.55)),
        ("🌿", (40, 140, 60), (0.55, 0.30, 0.40, 0.25)),
        ("🌸", (236, 160, 180), (0.80, 0.70, 0.75, 0.55)),
        ("🌼", (236, 200, 60), (0.78, 0.62, 0.70, 0.40)),
        ("☀️", (240, 190, 40), (0.90, 0.55, 0.55, 0.30)),
        ("🌙", (200, 200, 170), (0.75, 0.40, 0.55, 0.30)),
        ("☁️", (210, 214, 220), _flat(0.84)),
        ("❄️", (170, 210, 230), (0.70, 0.45, 0.45, 0.70)),
        ("💜", (130, 50, 180), (0.35, 0.50, 0.55, 0.22)),
        ("💙", (30, 80, 190), (0.30, 0.48, 0.52, 0.18)),
        ("💚", (30, 140, 70), (0.32, 0.50, 0.55, 0.20)),
        ("💛", (220, 180, 40), (0.68, 0.80, 0.82, 0.50)),
        ("🧡", (220, 110, 30), (0.55, 0.68, 0.70, 0.36)),
        ("🖤", (24, 24, 28), (0.10, 0.10, 0.22, 0.06)),
        ("🤍", (236, 236, 240), (0.88, 0.88, 0.96, 0.70)),
        ("🤎", (112, 70, 42), (0.30, 0.30, 0.46, 0.16)),
        ("💖", (220, 70, 130), (0.62, 0.74, 0.80, 0.42)),
        ("💗", (228, 100, 150), (0.70, 0.78, 0.84, 0.50)),
        ("⭕", (210, 210, 210), (0.60, 0.60, 0.60, 0.92)),
        ("⚫", (18, 18, 22), (0.08, 0.08, 0.08, 0.28)),
        ("⚪", (230, 230, 234), (0.72, 0.72, 0.72, 0.94)),
        ("🔲", (180, 180, 186), (0.35, 0.35, 0.35, 0.80)),
        ("🔳", (40, 40, 44), (0.15, 0.70, 0.70, 0.15)),
        ("▪️", (58, 58, 62), _flat(0.24)),
        ("▫️", (220, 220, 224), _flat(0.86)),
        ("◼️", (32, 32, 36), _flat(0.14)),
        ("◻️", (200, 200, 204), _flat(0.80)),
        ("🟥", (170, 30, 36), (0.18, 0.40, 0.40, 0.18)),
        ("🟧", (190, 100, 28), (0.40, 0.58, 0.58, 0.32)),
        ("🟨", (200, 170, 36), (0.62, 0.74, 0.74, 0.48)),
        ("🟩", (28, 130, 58), (0.28, 0.46, 0.46, 0.22)),
        ("🟦", (28, 80, 170), (0.26, 0.44, 0.44, 0.20)),
        ("🟪", (120, 40, 160), (0.30, 0.48, 0.48, 0.22)),
        ("🟫", (96, 60, 36), (0.24, 0.40, 0.40, 0.16)),
    ]
    seen = set()
    stickers = []
    for char, rgb, pattern, kind in items + [_sticker(c, rgb, pattern) for c, rgb, pattern in colors]:
        if char in seen:
            continue
        seen.add(char)
        stickers.append((char, rgb, pattern, kind))
    return stickers


STICKERS = _build_stickers()
FLAT = [item for item in STICKERS if item[3] == "flat"]
SIMPLE_CHARS = np.array([item[0] for item in FLAT])
SIMPLE_COLORS = np.array([item[1] for item in FLAT], dtype=np.float32)
LINE_CHARS = np.array([item[0] for item in STICKERS])
LINE_PATTERNS = np.array([item[2] for item in STICKERS], dtype=np.float32)
COLORS = SIMPLE_COLORS


def load_rgb(source):
    if isinstance(source, np.ndarray):
        array = source.astype(np.float32)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        return array
    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, (bytes, bytearray)):
        image = Image.open(BytesIO(source))
    else:
        image = Image.open(source)
    image = image.convert("RGB")
    return np.asarray(image, dtype=np.float32)


def _resize(image, size, resample):
    if image.ndim == 2:
        frame = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="L")
        frame = frame.resize(size, resample)
        return np.asarray(frame, dtype=np.float32) / 255.0
    frame = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")
    frame = frame.resize(size, resample)
    return np.asarray(frame, dtype=np.float32)


def _luminance(rgb):
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0


def _conv2(image, kernel):
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    out = np.zeros(image.shape, dtype=np.float32)
    for y in range(kh):
        for x in range(kw):
            out += kernel[y, x] * padded[y:y + image.shape[0], x:x + image.shape[1]]
    return out


def _edges(gray):
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    magnitude = np.sqrt(_conv2(gray, kernel_x) ** 2 + _conv2(gray, kernel_y) ** 2)
    scale = float(np.percentile(magnitude, 95)) + 1e-6
    return np.clip(magnitude / scale, 0.0, 1.0)


def _overlay(base, top):
    return np.where(base < 0.5, 2.0 * base * top, 1.0 - 2.0 * (1.0 - base) * (1.0 - top))


def _limit_side(rgb, side=480):
    height, width = rgb.shape[:2]
    scale = side / max(height, width)
    if scale >= 1:
        return rgb
    return _resize(rgb, (max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BOX)


def _grid(width, height, cells, square):
    cells = int(max(8, min(cells, 48)))
    if square:
        return cells, cells
    if width >= height:
        cols = cells
        rows = max(8, int(round(cells * height / max(width, 1))))
    else:
        rows = cells
        cols = max(8, int(round(cells * width / max(height, 1))))
    return cols, rows


def _nearest_colors(blocks):
    diff = blocks[:, :, None, :] - SIMPLE_COLORS[None, None, :, :]
    return np.argmin(np.sum(diff * diff, axis=-1), axis=-1)


def _nearest_patterns(patterns):
    diff = patterns[:, :, None, :] - LINE_PATTERNS[None, None, :, :]
    return np.argmin(np.sum(diff * diff, axis=-1), axis=-1)


def _paint(indexes, chars):
    rows = []
    for y in range(indexes.shape[0]):
        rows.append("".join(chars[int(index)] for index in indexes[y]))
    return rows


def simple_indexes(rgb, cells=32):
    work = _limit_side(rgb)
    cols, rows = _grid(work.shape[1], work.shape[0], cells, square=False)
    sampled = _resize(work, (cols * 4, rows * 4), Image.Resampling.BOX)
    blocks = sampled.reshape(rows, 4, cols, 4, 3).mean(axis=(1, 3))
    return _nearest_colors(blocks)


def line_indexes(rgb, cells=32):
    work = _limit_side(rgb)
    cols, rows = _grid(work.shape[1], work.shape[0], cells, square=True)
    gray = _luminance(work)
    line = 1.0 - _edges(gray)
    side = cols * 2
    gray_sq = _resize(gray, (side, side), Image.Resampling.BOX)
    line_sq = _resize(line, (side, side), Image.Resampling.BILINEAR)
    merged = np.clip(_overlay(gray_sq, line_sq), 0.0, 1.0)
    tiles = merged.reshape(rows, 2, cols, 2)
    patterns = np.stack(
        [
            tiles[:, 0, :, 0],
            tiles[:, 0, :, 1],
            tiles[:, 1, :, 0],
            tiles[:, 1, :, 1],
        ],
        axis=-1,
    )
    return _nearest_patterns(patterns)


def render_pic(source, mode="simple", cells=32):
    mode_name = resolve_mode(mode)
    rgb = load_rgb(source)
    if mode_name == "line":
        indexes = line_indexes(rgb, cells=cells)
        chars = LINE_CHARS
    else:
        indexes = simple_indexes(rgb, cells=cells)
        chars = SIMPLE_CHARS
    rows = _paint(indexes, chars)
    return HEADER + "\n" + "\n".join(rows)


def resolve_mode(mode):
    key = str(mode or "simple").strip().lower()
    if key in {"1", "line", "lines", "cool"}:
        return "line"
    if key in {"2", "simple", "color", "avg"}:
        return "simple"
    raise ValueError(f"Unknown pic mode: {mode}")


def discord_chunks(text, limit=1800):
    lines = text.split("\n")
    chunks = []
    current = ""
    for line in lines:
        piece = line if not current else "\n" + line
        if current and len(current) + len(piece) > limit:
            chunks.append(current)
            current = line
        else:
            current += piece
    if current:
        chunks.append(current)
    return chunks
