import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HEADER = "Пикча:"
MIN_CELLS = 8
# Один смайл в PNG/GIF — OUTPUT_SMILE_PX×OUTPUT_SMILE_PX. Ширина = cols×OUTPUT_SMILE_PX; --scale — исходных px на квадрант.
OUTPUT_SMILE_PX = 8
_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\seguisym.ttf"),
    Path(r"C:\Windows\Fonts\seguiemj.ttf"),
    Path(r"C:\Windows\Fonts\consola.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
)
_GLYPH_ATLAS = {}
_GLYPH_FONTS = {}
DISCORD_MAX_ROWS = 31
DISCORD_MAX_COLS = 102
DISCORD_SMALL_CELLS = 31
DISCORD_LARGE_CELLS = DISCORD_MAX_COLS
DEFAULT_CELLS = DISCORD_SMALL_CELLS
MAX_CELLS = DISCORD_MAX_COLS
CLI_MAX_COLS = 320
CLI_DEFAULT_CELLS = CLI_MAX_COLS
CLI_DEFAULT_SCALE = 2
PATCH = 16
LINE_THRESHOLD = 48.0
LINE_MIN_SIZE = 2
LINE_CONTRAST = 0.55
LINE_STEP = 32.0
LINE_MAX_DEPTH = 12
# Spread gates on 2x2 cell (max-min). line_thr ~0.45 (probe); details_thr = FLAT_SPREAD.
LINE_THR = 0.45
DETAILS_THR = 0.08

# Режим simple: одиннадцать эмодзи от чёрного к белому.
RAMP = (
    ("⬛", 0),
    ("⚫", 26),
    ("◼️", 51),
    ("◾", 77),
    ("▪️", 102),
    ("🔘", 128),
    ("◽", 153),
    ("▫️", 179),
    ("◻️", 204),
    ("⚪", 230),
    ("⬜", 255),
)
SIMPLE_DARK_DEFAULT = "⬛"
SIMPLE_LIGHT_DEFAULT = "⬜"


def _flat(level):
    tone = level / 255.0
    return (tone, tone, tone, tone)


def _build_stickers():
    stickers = []
    for char, level in RAMP:
        stickers.append((char, (level, level, level), _flat(level), "flat"))
    return stickers


STICKERS = _build_stickers()
RAMP_CHARS = np.array([item[0] for item in STICKERS])
SIMPLE_CHARS = RAMP_CHARS
SIMPLE_COLORS = np.array([item[1] for item in STICKERS], dtype=np.float32)
COLORS = SIMPLE_COLORS


def _one_simple_glyph(raw, default, label):
    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default
    if any(ch in text for ch in "\n\r\t"):
        raise ValueError(f"{label}: need one symbol, not whitespace or control characters")
    ascii_letters = sum(1 for ch in text if ord(ch) < 128 and ch.isalnum())
    if len(text) > 1 and ascii_letters >= 2:
        raise ValueError(f"{label}: need one symbol, got {len(text)} characters")
    if len(text) > 16:
        raise ValueError(f"{label}: symbol too long")
    return text


def resolve_simple_glyphs(dark=None, light=None):
    dark_char = _one_simple_glyph(dark, SIMPLE_DARK_DEFAULT, "dark")
    light_char = _one_simple_glyph(light, SIMPLE_LIGHT_DEFAULT, "light")
    return {"dark": dark_char, "light": light_char}


def simple_quantize_steps(simple=None):
    # Always binary: cell mean → 0|1 → dark|light. RAMP is legacy dictionary only.
    del simple
    return 1


def simple_paint_chars(simple=None):
    glyphs = resolve_simple_glyphs(
        None if simple is None else simple.get("dark"),
        None if simple is None else simple.get("light"),
    )
    return np.array([glyphs["dark"], glyphs["light"]])


# На ячейку — непрерывная матрица 2×2 (квадрант 0..1, не жёсткий бинарник). Порядок TL, TR, BL, BR.
LINE_SPREAD = 0.28
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_SHADE_RAMP_PATH = _ASSETS_DIR / "shade_ramp.json"
_LINE_PATTERNS_PATH = _ASSETS_DIR / "line_patterns.json"
_DETAILS_PATTERNS_PATH = _ASSETS_DIR / "details_patterns.json"


def _load_pattern_pairs(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple((char, tuple(float(v) for v in pattern)) for char, pattern in raw)


def _load_measured_line_patterns():
    return _load_pattern_pairs(_LINE_PATTERNS_PATH)


def _load_details_patterns():
    return _load_pattern_pairs(_DETAILS_PATTERNS_PATH)


def _load_shade_ramp(path=_SHADE_RAMP_PATH):
    # Glyph i owns (level[i-1], level[i]]; first [0, level0]; █ closes at 1.0.
    # Right edges = hi; searchsorted(side=left). Pack may write space lo=hi=0.
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw:
        raise ValueError(f"empty shade ramp: {path}")
    spec = []
    his = []
    for item in raw:
        char = item["char"]
        level = float(item["level"])
        # Degenerate first bin (space lo=hi=0): searchsorted(side=left) maps 0→index 0.
        hi = float(item["hi"])
        spec.append((char, level))
        his.append(hi)
    his[-1] = 1.0
    spec[-1] = (spec[-1][0], 1.0)
    his = np.asarray(his, dtype=np.float32)
    if np.any(np.diff(his) < 0):
        raise ValueError("shade ramp hi must be non-decreasing")
    return tuple(spec), his


_SHADE_SPEC, _SHADE_HI = _load_shade_ramp()
_MEASURED_LINE_SPEC = _load_measured_line_patterns()
_DETAILS_SPEC = _load_details_patterns()
DETAILS_CHARS = np.array([char for char, _ in _DETAILS_SPEC])
DETAILS_PATTERNS = np.array([pattern for _, pattern in _DETAILS_SPEC], dtype=np.float32)
_DETAILS_COUNT = len(_DETAILS_SPEC)
_LINE_SPEC = tuple((char, (level, level, level, level)) for char, level in _SHADE_SPEC) + _MEASURED_LINE_SPEC
LINE_CHARS = np.array([char for char, _ in _LINE_SPEC])
LINE_PATTERNS = np.array([pattern for _, pattern in _LINE_SPEC], dtype=np.float32)
LINE_COLORS = np.repeat(LINE_PATTERNS.mean(axis=1)[:, None] * 255.0, 3, axis=1)
_SHADE_COUNT = len(_SHADE_SPEC)
_MEASURED_COUNT = len(_MEASURED_LINE_SPEC)
_DETAILS_BASE = _SHADE_COUNT + _MEASURED_COUNT
# Paint alphabet: shade + measured line + details (indexes from compose_line_indexes).
LINE_PAINT_CHARS = np.concatenate([LINE_CHARS, DETAILS_CHARS])
LINE_PAINT_PATTERNS = np.concatenate([LINE_PATTERNS, DETAILS_PATTERNS], axis=0)
MEASURED_PATTERNS = LINE_PATTERNS[_SHADE_COUNT:_DETAILS_BASE]


def clamp_cells(cells, max_cells=MAX_CELLS):
    value = int(max(MIN_CELLS, int(round(cells))))
    if max_cells is None:
        return value
    return min(value, int(max_cells))


def resolve_size(size):
    key = str(size or "small").strip().lower()
    if key in {"small", "s", "sm", "mini", "маленький", "мал", "м"}:
        return "small", DISCORD_SMALL_CELLS
    if key in {"large", "l", "lg", "big", "max", "большой", "бол", "б"}:
        return "large", DISCORD_LARGE_CELLS
    raise ValueError(f"Unknown pic size: {size}")


def _discord_len(text):
    # Длина сообщения Discord как в JS — UTF-16 code units.
    return len(text.encode("utf-16-le")) // 2


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
    # В стикерах под alpha=0 лежит белый; сброс alpha закрасит прозрачность белым.
    if image.mode in {"RGBA", "LA"} or ("transparency" in image.info):
        rgba = image.convert("RGBA")
        rgba = _crop_alpha(rgba)
        bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        image = Image.alpha_composite(bg, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.float32)


def _crop_alpha(rgba, threshold=8, pad=1):
    alpha = np.asarray(rgba.split()[-1])
    ys, xs = np.where(alpha > threshold)
    if ys.size == 0:
        return rgba
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(rgba.height, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(rgba.width, int(xs.max()) + 1 + pad)
    return rgba.crop((x0, y0, x1, y1))


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


def oklab_L(rgb):
    # OKLab L* from sRGB 0..255; output roughly [0,1].
    rgb = np.asarray(rgb, dtype=np.float32)
    r = np.clip(rgb[..., 0], 0.0, 255.0) / 255.0
    g = np.clip(rgb[..., 1], 0.0, 255.0) / 255.0
    b = np.clip(rgb[..., 2], 0.0, 255.0) / 255.0

    def _linear(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    r, g, b = _linear(r), _linear(g), _linear(b)
    l_ = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m_ = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s_ = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l = np.cbrt(l_)
    m = np.cbrt(m_)
    s = np.cbrt(s_)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s).astype(np.float32)


def structure_field(rgb):
    # Color-structure mask source in [0,1]: chroma spread + spatial RGB gradient.
    rgb = np.asarray(rgb, dtype=np.float32)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    chroma = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(b - r)) / 255.0
    dx = np.abs(rgb[:, 1:, :] - rgb[:, :-1, :]).mean(axis=-1)
    dy = np.abs(rgb[1:, :, :] - rgb[:-1, :, :]).mean(axis=-1)
    grad = np.zeros(rgb.shape[:2], dtype=np.float32)
    grad[:, 1:] += dx
    grad[:, :-1] += dx
    grad[1:, :] += dy
    grad[:-1, :] += dy
    grad = np.clip(grad / (4.0 * 255.0), 0.0, 1.0)
    return np.clip(0.5 * chroma + 0.5 * grad, 0.0, 1.0).astype(np.float32)


def _tone_field(rgb, opts):
    if opts.get("tone") == "oklab":
        return oklab_L(rgb)
    return _luminance(np.asarray(rgb, dtype=np.float32))


def _structure_source(rgb, opts):
    if opts.get("structure") == "color":
        return structure_field(rgb)
    return _luminance(np.asarray(rgb, dtype=np.float32))


def _stretch(gray, gamma=0.72):
    flat = np.asarray(gray, dtype=np.float32).ravel()
    # Чистый чёрный не в квантиль; midtones чуть поднимаем — тёмное аниме читается.
    lit = flat[flat > 0.04]
    sample = lit if lit.size >= max(32, flat.size // 20) else flat
    lo = float(np.percentile(sample, 5))
    hi = float(np.percentile(sample, 98))
    if hi - lo < 1e-4:
        stretched = np.clip(gray, 0.0, 1.0)
    else:
        stretched = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return np.power(stretched, gamma)


def _grid(width, height, cells, patch=PATCH, discord=True):
    # Жёсткий лимит — ширина в символах; высота по aspect.
    width = max(int(width), 1)
    height = max(int(height), 1)
    max_cols = DISCORD_MAX_COLS if discord else CLI_MAX_COLS
    default = DEFAULT_CELLS if discord else CLI_DEFAULT_CELLS
    cols = clamp_cells(default if cells is None else cells, max_cells=max_cols)
    cols = min(cols, max_cols)
    rows = max(1, int(round(cols * height / width)))
    return cols, rows


def resolve_frame(width, height, cells, scale, upscale, discord):
    # Discord: 31/102. В CLI cols уменьшаются только с --no-upscale.
    scale = max(1, int(scale))
    width = max(int(width), 1)
    height = max(int(height), 1)
    if discord:
        return _grid(width, height, cells, discord=True)
    requested = clamp_cells(
        CLI_DEFAULT_CELLS if cells is None else cells,
        max_cells=CLI_MAX_COLS,
    )
    need = requested * 2 * scale
    if width >= need or upscale:
        cols = requested
    else:
        cols = max(1, width // (2 * scale))
    rows = max(1, int(round(cols * height / float(width))))
    return cols, rows


def _window_origins(length, count, win):
    count = max(1, int(count))
    win = max(1, int(win))
    length = max(1, int(length))
    last = max(0, length - win)
    if count == 1:
        return np.zeros(1, dtype=np.int32)
    origins = np.rint(np.linspace(0, last, count)).astype(np.int32)
    return np.clip(origins, 0, last)


def _raster_for_smiles(gray, rows, cols, scale, upscale):
    # Растр ровно под сетку смайлов: nearest-up при --upscale и коротком источнике; иначе nearest-fit под блок scale×scale.
    win = 2 * max(1, int(scale))
    target_h = max(1, int(rows)) * win
    target_w = max(1, int(cols)) * win
    height, width = gray.shape[:2]
    if height == target_h and width == target_w:
        return gray
    if upscale or (height >= target_h and width >= target_w):
        return _resize(gray, (target_w, target_h), Image.NEAREST)
    cropped = gray[: min(height, target_h), : min(width, target_w)]
    pad_h = target_h - cropped.shape[0]
    pad_w = target_w - cropped.shape[1]
    if pad_h or pad_w:
        return np.pad(cropped, ((0, pad_h), (0, pad_w)), mode="edge")
    return cropped


def _fit_smile_windows(gray, win):
    height, width = gray.shape[:2]
    pad_h = max(0, int(win) - height)
    pad_w = max(0, int(win) - width)
    if pad_h or pad_w:
        return np.pad(gray, ((0, pad_h), (0, pad_w)), mode="edge")
    return gray


def _reduce_block(block, reduce):
    if reduce == "max":
        return block.max(axis=(2, 3))
    return block.mean(axis=(2, 3))


def smile_blocks(field, rows, cols, scale, reduce="mean"):
    # Квадрант — сплошной блок scale×scale (A B / C D).
    field = np.asarray(field, dtype=np.float32)
    scale = max(1, int(scale))
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    win = 2 * scale
    if field.shape[0] == rows * win and field.shape[1] == cols * win:
        view = field.reshape(rows, win, cols, win)
        windows = np.transpose(view, (0, 2, 1, 3))
    else:
        field = _fit_smile_windows(field, win)
        ys = _window_origins(field.shape[0], rows, win)
        xs = _window_origins(field.shape[1], cols, win)
        y_idx = ys[:, None, None, None] + np.arange(win, dtype=np.int32)[None, None, :, None]
        x_idx = xs[None, :, None, None] + np.arange(win, dtype=np.int32)[None, None, None, :]
        windows = field[y_idx, x_idx]
    half = scale
    quads = np.stack(
        [
            _reduce_block(windows[:, :, 0:half, 0:half], reduce),
            _reduce_block(windows[:, :, 0:half, half:win], reduce),
            _reduce_block(windows[:, :, half:win, 0:half], reduce),
            _reduce_block(windows[:, :, half:win, half:win], reduce),
        ],
        axis=-1,
    )
    return quads, windows.mean(axis=(2, 3))


def _sample_gray(rgb, cols, rows, scale, upscale, gray=None):
    if gray is None:
        gray = _luminance(np.asarray(rgb, dtype=np.float32))
    gray = _raster_for_smiles(gray, rows, cols, scale, upscale)
    return _stretch(gray)


def _prepare_for_grid(rgb, cols, rows):
    # Ячейка стикера — 4 субпикселя; если источник меньше сетки — nearest-up, чтобы subblock был ≥1 px.
    rgb = np.asarray(rgb, dtype=np.float32)
    height, width = rgb.shape[:2]
    sub_w = max(1, int(cols) * 2)
    sub_h = max(1, int(rows) * 2)
    if width / sub_w < 1.0 or height / sub_h < 1.0:
        return _resize(rgb, (sub_w, sub_h), Image.NEAREST)
    return rgb


def _quantize(gray, *, steps=None):
    if steps is None:
        steps = len(RAMP_CHARS) - 1
    return np.rint(np.clip(gray, 0.0, 1.0) * steps).astype(np.int32)


def quads_from_patch(block):
    if block.ndim == 2:
        block = np.repeat(block[:, :, None], 3, axis=2)
    patch = int(block.shape[0])
    half = patch // 2
    view = block[:patch, :patch].reshape(2, half, 2, half, block.shape[2])
    means = view.mean(axis=(1, 3))
    return np.stack([means[0, 0], means[0, 1], means[1, 0], means[1, 1]], axis=0)


def _paint(indexes, chars):
    rows = []
    for y in range(indexes.shape[0]):
        rows.append("".join(chars[int(index)] for index in indexes[y]))
    return rows


def _bin_edges(length, count):
    return np.linspace(0, length, count + 1).astype(np.int32)


def _adaptive_pool2d(field, out_h, out_w, reduce="mean"):
    import torch
    import torch.nn.functional as F

    tensor = torch.as_tensor(np.asarray(field, dtype=np.float32)).reshape(1, 1, field.shape[0], field.shape[1])
    if reduce == "max":
        pooled = F.adaptive_max_pool2d(tensor, (out_h, out_w))
    else:
        pooled = F.adaptive_avg_pool2d(tensor, (out_h, out_w))
    return pooled[0, 0].cpu().numpy()


def downsample_field(field, cols, rows, reduce="mean"):
    # Сжатие с исходной сетки через adaptive pool.
    field = np.asarray(field, dtype=np.float32)
    return _adaptive_pool2d(field, rows, cols, reduce=reduce)


def downsample_quads(field, cols, rows, reduce="mean"):
    # На ячейку — 2×2 регионов источника: TL TR BL BR.
    field = np.asarray(field, dtype=np.float32)
    grid = _adaptive_pool2d(field, rows * 2, cols * 2, reduce=reduce)
    return np.stack(
        [
            grid[0::2, 0::2],
            grid[0::2, 1::2],
            grid[1::2, 0::2],
            grid[1::2, 1::2],
        ],
        axis=-1,
    )


def _nearest_pattern_indexes(quads, patterns):
    quads = np.clip(np.asarray(quads, dtype=np.float32), 0.0, 1.0)
    patterns = np.asarray(patterns, dtype=np.float32)
    diff = quads[..., None, :] - patterns.reshape((1,) * (quads.ndim - 1) + patterns.shape)
    return np.argmin(np.sum(diff * diff, axis=-1), axis=-1)


def _pattern_indexes(quads):
    # Ближайший непрерывный паттерн 2×2 из полного line-словаря (shade + measured).
    return _nearest_pattern_indexes(quads, LINE_PATTERNS)


def shade_indexes(means):
    # Range lookup on right edges hi: glyph i owns (hi[i-1], hi[i]] (first: [0, hi0]).
    means = np.clip(np.asarray(means, dtype=np.float32), 0.0, 1.0)
    idx = np.searchsorted(_SHADE_HI, means, side="left")
    return np.minimum(idx, _SHADE_COUNT - 1).astype(np.int32)


def compose_line_indexes(
    quads,
    window_means,
    details=False,
    line_thr=None,
    details_thr=None,
):
    # line/render (details=False): classic NN — nearest among shade+measured;
    # keep measured glyph, else flat shade from window mean. No hard LINE_THR gate.
    # details mode only: high spread → measured; mid → DETAILS_PATTERNS; flat → shade.
    quads = np.clip(np.asarray(quads, dtype=np.float32), 0.0, 1.0)
    bg = shade_indexes(window_means)
    if not details:
        nearest = _pattern_indexes(quads)
        return np.where(nearest >= _SHADE_COUNT, nearest, bg).astype(np.int32)
    line_thr = LINE_THR if line_thr is None else float(line_thr)
    details_thr = DETAILS_THR if details_thr is None else float(details_thr)
    spread = quads.max(axis=-1) - quads.min(axis=-1)
    line_local = _nearest_pattern_indexes(quads, MEASURED_PATTERNS)
    line_idx = line_local + _SHADE_COUNT
    use_line = spread >= line_thr
    detail_local = _nearest_pattern_indexes(quads, DETAILS_PATTERNS)
    detail_idx = detail_local + _DETAILS_BASE
    use_details = (~use_line) & (spread >= details_thr)
    return np.where(use_line, line_idx, np.where(use_details, detail_idx, bg)).astype(np.int32)


def compose_masked_line(
    ink_quads,
    window_means,
    details=False,
    line_thr=None,
    details_thr=None,
):
    # ink 0 = пиксель линии; смайл под маской — плоский чёрный (shade index), оставляем ink shade, не фон фото.
    indexes = compose_line_indexes(
        ink_quads,
        window_means,
        details=details,
        line_thr=line_thr,
        details_thr=details_thr,
    )
    covered = np.asarray(ink_quads, dtype=np.float32).min(axis=-1) < 1.0 - 1e-6
    ink_idx = _nearest_pattern_indexes(ink_quads, MEASURED_PATTERNS) + _SHADE_COUNT
    flat = indexes < _SHADE_COUNT
    return np.where(covered & flat, ink_idx, indexes).astype(np.int32)


def line_options(
    threshold=None,
    min_size=None,
    contrast=None,
    step=None,
    depth=None,
    tone=None,
    structure=None,
    details=None,
    line_thr=None,
    details_thr=None,
):
    # tone: "rec601" (default) | "oklab"; structure: "luma" (default) | "color".
    # Public mode "render" → oklab + color. Mode "details" → oklab + soft mid-band.
    # Mode "line" keeps Rec.601. details=True only for mode "details".
    tone_key = "rec601" if tone is None else str(tone).strip().lower()
    if tone_key not in ("rec601", "oklab"):
        raise ValueError(f"unknown line tone: {tone}")
    structure_key = "luma" if structure is None else str(structure).strip().lower()
    if structure_key not in ("luma", "color"):
        raise ValueError(f"unknown line structure: {structure}")
    return {
        "threshold": LINE_THRESHOLD if threshold is None else float(threshold),
        "min_size": LINE_MIN_SIZE if min_size is None else max(1, int(min_size)),
        "contrast": LINE_CONTRAST if contrast is None else float(np.clip(contrast, 0.0, 1.0)),
        "step": LINE_STEP if step is None else float(step),
        "depth": LINE_MAX_DEPTH if depth is None else max(1, int(depth)),
        "tone": tone_key,
        "structure": structure_key,
        "details": False if details is None else bool(details),
        "line_thr": LINE_THR if line_thr is None else float(np.clip(line_thr, 0.0, 1.0)),
        "details_thr": DETAILS_THR if details_thr is None else float(np.clip(details_thr, 0.0, 1.0)),
    }


def render_line_opts():
    # Mode "render": OKLab L* tone + color-structure mask.
    return {"tone": "oklab", "structure": "color"}


def _adapt_mask_opts(opts, scale):
    # Situational: huge smile windows + color edges → dense ink / scanline noise.
    # scale 5..23: coarsen color mask. scale>=24: OKLab tone stays, mask falls back to luma.
    scale = max(1, int(scale))
    out = dict(opts)
    if out.get("structure") != "color" or scale <= 4:
        return out
    if scale >= 24:
        out["structure"] = "luma"
        out["min_size"] = LINE_MIN_SIZE
        out["threshold"] = LINE_THRESHOLD
        out["step"] = LINE_STEP
        out["contrast"] = min(float(out["contrast"]), 0.45)
        return out
    grow = scale / 4.0
    out["min_size"] = max(int(out["min_size"]), max(2, scale // 8))
    out["threshold"] = float(out["threshold"]) * min(1.0 + 0.2 * (grow ** 0.5), 1.8)
    out["step"] = float(out["step"]) * min(grow, 4.0)
    if scale >= 12:
        out["contrast"] = float(out["contrast"]) * max(0.4, 1.0 - 0.015 * (scale - 12))
    return out


def lab_line_payload(lab=False, *, simple=False):
    # Deprecated alias; prefer mode "render" / render_line_opts().
    if not lab:
        return None
    if simple:
        return {"tone": "oklab"}
    return render_line_opts()


def tint_line_quads(photo_quads, mask_quads, contrast=None):
    # Цвет линии — от соседей без маски в том же смайле, затем затемнение.
    photo = np.clip(np.asarray(photo_quads, dtype=np.float32), 0.0, 1.0)
    mask = np.clip(np.asarray(mask_quads, dtype=np.float32), 0.0, 1.0)
    contrast = LINE_CONTRAST if contrast is None else float(np.clip(contrast, 0.0, 1.0))
    on = mask > 0.05
    off = ~on
    count = np.maximum(off.sum(axis=-1, keepdims=True), 1).astype(np.float32)
    neighbor = (photo * off).sum(axis=-1, keepdims=True) / count
    fallback = photo.mean(axis=-1, keepdims=True)
    neighbor = np.where(on.all(axis=-1, keepdims=True), fallback, neighbor)
    ink = neighbor * (1.0 - contrast)
    return np.where(on, ink, photo)


def compose_tinted_line(
    photo_quads,
    mask_quads,
    window_means,
    contrast=None,
    details=False,
    line_thr=None,
    details_thr=None,
):
    tinted = tint_line_quads(photo_quads, mask_quads, contrast=contrast)
    bg = shade_indexes(window_means)
    composed = compose_line_indexes(
        tinted,
        window_means,
        details=details,
        line_thr=line_thr,
        details_thr=details_thr,
    )
    covered = np.asarray(mask_quads, dtype=np.float32).max(axis=-1) > 0.05
    return np.where(covered, composed, bg).astype(np.int32)


def pick_blocks(tone, form=None, details=False, line_thr=None, details_thr=None):
    # Подбор под непрерывную tone-матрицу; form может чуть заострить контраст.
    tone = np.asarray(tone, dtype=np.float32)
    if form is None:
        target = tone
    else:
        form = np.asarray(form, dtype=np.float32)
        spread = (form.max(axis=-1) - form.min(axis=-1))[..., None]
        mix = np.clip(spread / max(LINE_SPREAD, 1e-6), 0.0, 1.0) * 0.35
        target = tone * (1.0 - mix) + form * mix
    means = target.mean(axis=-1)
    return compose_line_indexes(
        target,
        means,
        details=details,
        line_thr=line_thr,
        details_thr=details_thr,
    )


def simple_indexes(rgb, cells=None, discord=True, scale=None, upscale=True, line=None, simple=None):
    scale = CLI_DEFAULT_SCALE if scale is None else max(1, int(scale))
    opts = line_options(**(line or {}))
    steps = simple_quantize_steps(simple)
    if discord:
        cols, rows = _grid(rgb.shape[1], rgb.shape[0], cells, discord=True)
        rgb = _prepare_for_grid(rgb, cols, rows)
        gray = _stretch(_tone_field(rgb, opts))
        tone = downsample_quads(gray, cols, rows, reduce="mean")
        return _quantize(tone.mean(axis=-1), steps=steps)
    cols, rows = resolve_frame(rgb.shape[1], rgb.shape[0], cells, scale, upscale, False)
    gray = _sample_gray(rgb, cols, rows, scale, upscale, gray=_tone_field(rgb, opts))
    _quads, means = smile_blocks(gray, rows, cols, scale)
    return _quantize(means, steps=steps)


def _block_mse(gray, y, x, height, width):
    # MSE как в quadtree Power BI, но относительно средней заливки блока.
    block = gray[y:y + height, x:x + width]
    if block.size == 0:
        return 0.0
    values = block * 255.0
    mean = float(values.mean())
    return float(np.mean((values - mean) ** 2))


def _quadtree_leaves(gray, y, x, height, width, threshold, min_size, depth, max_depth, leaves):
    mse = _block_mse(gray, y, x, height, width)
    if mse <= threshold:
        return
    if depth >= max_depth or height <= min_size or width <= min_size:
        leaves.append((y, x, height, width))
        return
    half_h = height // 2
    half_w = width // 2
    if half_h < 1 or half_w < 1:
        leaves.append((y, x, height, width))
        return
    quads = (
        (y, x, half_h, half_w),
        (y, x + half_w, half_h, width - half_w),
        (y + half_h, x, height - half_h, half_w),
        (y + half_h, x + half_w, height - half_h, width - half_w),
    )
    for qy, qx, qh, qw in quads:
        if qh > 0 and qw > 0:
            _quadtree_leaves(gray, qy, qx, qh, qw, threshold, min_size, depth + 1, max_depth, leaves)


def _square_covers(height, width):
    side = min(height, width)
    if side < 1:
        return []
    covers = []
    if height >= width:
        y = 0
        while y + side <= height:
            covers.append((y, 0, side))
            y += side
        if not covers or covers[-1][0] + side < height:
            covers.append((max(0, height - side), 0, side))
    else:
        x = 0
        while x + side <= width:
            covers.append((0, x, side))
            x += side
        if not covers or covers[-1][1] + side < width:
            covers.append((0, max(0, width - side), side))
    return covers


def _paint_quadtree(gray, min_size, threshold=48.0):
    gray = np.asarray(gray, dtype=np.float32)
    height, width = gray.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    leaves = []
    for y, x, side in _square_covers(height, width):
        _quadtree_leaves(gray, y, x, side, side, threshold, min_size, 0, 12, leaves)
    for y, x, leaf_h, leaf_w in leaves:
        mask[y:y + leaf_h, x:x + leaf_w] = 1.0
    return mask


def _paint_seam(mask, y0, y1, x0, x1):
    if y1 > y0 and x1 > x0:
        mask[y0:y1, x0:x1] = 1.0


def _shrink_line(gray, y, x, height, width, threshold, min_size, depth, max_depth, mask, step):
    # Делим квадрат, пока в нём больше одного цвета; линия — 1px шов между листьями, не заливка.
    if height < 1 or width < 1:
        return
    mse = _block_mse(gray, y, x, height, width)
    if mse <= threshold:
        return
    if depth >= max_depth or height <= min_size or width <= min_size:
        return
    half_h = height // 2
    half_w = width // 2
    if half_h < 1 or half_w < 1:
        return
    quads = (
        (y, x, half_h, half_w),
        (y, x + half_w, half_h, width - half_w),
        (y + half_h, x, height - half_h, half_w),
        (y + half_h, x + half_w, height - half_h, width - half_w),
    )
    means = []
    leaf = []
    for qy, qx, qh, qw in quads:
        if qh < 1 or qw < 1:
            means.append(None)
            leaf.append(True)
            continue
        means.append(float(gray[qy:qy + qh, qx:qx + qw].mean()))
        child_mse = _block_mse(gray, qy, qx, qh, qw)
        leaf.append(
            child_mse <= threshold or depth + 1 >= max_depth or qh <= min_size or qw <= min_size
        )
    def far(left, right):
        return left is not None and right is not None and abs(left - right) >= step

    if leaf[0] and leaf[1] and far(means[0], means[1]):
        _paint_seam(mask, y, y + half_h, x + half_w, x + half_w + 1)
    if leaf[2] and leaf[3] and far(means[2], means[3]):
        _paint_seam(mask, y + half_h, y + height, x + half_w, x + half_w + 1)
    if leaf[0] and leaf[2] and far(means[0], means[2]):
        _paint_seam(mask, y + half_h, y + half_h + 1, x, x + half_w)
    if leaf[1] and leaf[3] and far(means[1], means[3]):
        _paint_seam(mask, y + half_h, y + half_h + 1, x + half_w, x + width)
    for (qy, qx, qh, qw), done in zip(quads, leaf):
        if qh > 0 and qw > 0 and not done:
            _shrink_line(
                gray, qy, qx, qh, qw, threshold, min_size, depth + 1, max_depth, mask, step
            )


def line_mask(gray, threshold=None, min_size=None, step=None, depth=None):
    # Контур в полном разрешении: квадраты сжимаются до одного цвета; ink — 1px шов между листьями. Scale позже при сэмпле сетки смайлов.
    opts = line_options(threshold=threshold, min_size=min_size, step=step, depth=depth)
    gray = np.asarray(gray, dtype=np.float32)
    height, width = gray.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    seam = opts["step"] / 255.0
    for y, x, side in _square_covers(height, width):
        _shrink_line(
            gray,
            y,
            x,
            side,
            side,
            opts["threshold"],
            opts["min_size"],
            0,
            opts["depth"],
            mask,
            seam,
        )
    return mask


def quadtree_form(gray, cols, rows, threshold=48.0):
    gray = np.asarray(gray, dtype=np.float32)
    height, width = gray.shape[:2]
    cell = max(2, min(height // max(rows, 1), width // max(cols, 1)))
    min_size = max(2, cell // 2)
    mask = _paint_quadtree(gray, min_size, threshold=threshold)
    return downsample_quads(mask, cols, rows, reduce="max")


def _raster_mask(mask, rows, cols, scale, upscale):
    # Тот же растр смайлов, что у фото; при сжатии max-pool, чтобы тонкая линия не пропала.
    win = 2 * max(1, int(scale))
    target_h = max(1, int(rows)) * win
    target_w = max(1, int(cols)) * win
    height, width = mask.shape[:2]
    if height == target_h and width == target_w:
        return mask
    if not (upscale or (height >= target_h and width >= target_w)):
        cropped = mask[: min(height, target_h), : min(width, target_w)]
        pad_h = target_h - cropped.shape[0]
        pad_w = target_w - cropped.shape[1]
        if pad_h or pad_w:
            return np.pad(cropped, ((0, pad_h), (0, pad_w)), mode="edge")
        return cropped
    if height >= target_h and width >= target_w:
        return _adaptive_pool2d(mask, target_h, target_w, reduce="max")
    return _resize(mask, (target_w, target_h), Image.NEAREST)


def _merge_line_opts(line=None, details=None, line_thr=None, details_thr=None):
    merged = dict(line or {})
    if details is not None:
        merged["details"] = details
    if line_thr is not None:
        merged["line_thr"] = line_thr
    if details_thr is not None:
        merged["details_thr"] = details_thr
    return line_options(**merged)


def line_indexes(
    rgb,
    cells=None,
    patch=PATCH,
    discord=True,
    form=True,
    scale=None,
    upscale=True,
    line=None,
    details=None,
    line_thr=None,
    details_thr=None,
):
    scale = CLI_DEFAULT_SCALE if scale is None else max(1, int(scale))
    opts = _adapt_mask_opts(
        _merge_line_opts(line, details=details, line_thr=line_thr, details_thr=details_thr),
        scale,
    )
    compose_kw = {
        "details": opts["details"],
        "line_thr": opts["line_thr"],
        "details_thr": opts["details_thr"],
    }
    split = opts["tone"] != "rec601" or opts["structure"] != "luma"
    # Stretch amplifies chroma noise into false seams when smile windows are huge.
    stretch_color_struct = opts["structure"] == "color" and scale < 24
    if discord:
        cols, rows = _grid(rgb.shape[1], rgb.shape[0], cells, patch=patch, discord=True)
        rgb = _prepare_for_grid(rgb, cols, rows)
        if not split:
            gray = _luminance(rgb)
            tone = downsample_quads(_stretch(gray), cols, rows, reduce="mean")
            if not form:
                return pick_blocks(tone, **compose_kw)
            from core.image.mask_pool import quadtree_forms_parallel

            form_quads = quadtree_forms_parallel(
                gray, cols, rows, opts["threshold"]
            )[0]
            return pick_blocks(tone, form_quads, **compose_kw)
        tone_raw = _tone_field(rgb, opts)
        struct_raw = _structure_source(rgb, opts)
        struct_field = _stretch(struct_raw) if stretch_color_struct else struct_raw
        tone = downsample_quads(_stretch(tone_raw), cols, rows, reduce="mean")
        if not form:
            return pick_blocks(tone, **compose_kw)
        from core.image.mask_pool import quadtree_forms_parallel

        form_quads = quadtree_forms_parallel(
            struct_field, cols, rows, opts["threshold"]
        )[0]
        return pick_blocks(tone, form_quads, **compose_kw)
    cols, rows = resolve_frame(rgb.shape[1], rgb.shape[0], cells, scale, upscale, False)
    if not split:
        lum = _luminance(np.asarray(rgb, dtype=np.float32))
        mask = _raster_mask(
            line_mask(
                lum,
                threshold=opts["threshold"],
                min_size=opts["min_size"],
                step=opts["step"],
                depth=opts["depth"],
            ),
            rows,
            cols,
            scale,
            upscale,
        )
        mask_quads, _unused = smile_blocks(mask, rows, cols, scale, reduce="max")
        gray = _sample_gray(rgb, cols, rows, scale, upscale)
        photo, means = smile_blocks(gray, rows, cols, scale)
        return compose_tinted_line(
            photo, mask_quads, means, contrast=opts["contrast"], **compose_kw
        )
    tone_raw = _tone_field(rgb, opts)
    struct_raw = _structure_source(rgb, opts)
    struct_field = _stretch(struct_raw) if stretch_color_struct else struct_raw
    mask = _raster_mask(
        line_mask(
            struct_field,
            threshold=opts["threshold"],
            min_size=opts["min_size"],
            step=opts["step"],
            depth=opts["depth"],
        ),
        rows,
        cols,
        scale,
        upscale,
    )
    mask_quads, _unused = smile_blocks(mask, rows, cols, scale, reduce="max")
    gray = _sample_gray(rgb, cols, rows, scale, upscale, gray=tone_raw)
    photo, means = smile_blocks(gray, rows, cols, scale)
    return compose_tinted_line(
        photo, mask_quads, means, contrast=opts["contrast"], **compose_kw
    )


def _paint_mode(mode_name):
    if mode_name in ("line", "render", "details"):
        return "line"
    return "simple"


def _mode_line_defaults(mode_name, line=None, details=None, line_thr=None, details_thr=None):
    # Four modes, four algorithms — not a details flag on one compose.
    merged = dict(line or {})
    if mode_name == "render":
        if "tone" not in merged:
            merged["tone"] = "oklab"
        if "structure" not in merged:
            merged["structure"] = "color"
        if details is None and "details" not in merged:
            merged["details"] = False
    elif mode_name == "details":
        # Soft 2x2 dict on OKLab L* (same luma space as render; no color-edge mask).
        if "tone" not in merged:
            merged["tone"] = "oklab"
        if details is None and "details" not in merged:
            merged["details"] = True
    elif mode_name == "line":
        if details is None and "details" not in merged:
            merged["details"] = False
    return _merge_line_opts(merged, details=details, line_thr=line_thr, details_thr=details_thr)


def pic_indexes(
    source,
    mode="simple",
    cells=None,
    discord=True,
    form=True,
    scale=None,
    upscale=True,
    line=None,
    simple=None,
    details=None,
    line_thr=None,
    details_thr=None,
):
    mode_name = resolve_mode(mode)
    rgb = load_rgb(source)
    if mode_name in ("line", "render", "details"):
        line_opts = _mode_line_defaults(
            mode_name, line=line, details=details, line_thr=line_thr, details_thr=details_thr
        )
        return line_indexes(
            rgb,
            cells=cells,
            discord=discord,
            form=form,
            scale=scale,
            upscale=upscale,
            line=line_opts,
        ), mode_name
    return (
        simple_indexes(
            rgb,
            cells=cells,
            discord=discord,
            scale=scale,
            upscale=upscale,
            line=line,
            simple=simple,
        ),
        mode_name,
    )


def render_pic(
    source,
    mode="simple",
    cells=None,
    discord=True,
    form=True,
    scale=None,
    upscale=True,
    line=None,
    simple=None,
    details=None,
    line_thr=None,
    details_thr=None,
):
    indexes, mode_name = pic_indexes(
        source,
        mode=mode,
        cells=cells,
        discord=discord,
        form=form,
        scale=scale,
        upscale=upscale,
        line=line,
        simple=simple,
        details=details,
        line_thr=line_thr,
        details_thr=details_thr,
    )
    chars = (
        LINE_PAINT_CHARS
        if _paint_mode(mode_name) == "line"
        else simple_paint_chars(simple)
    )
    rows = _paint(indexes, chars)
    return HEADER + "\n" + "\n".join(rows)


def render_png(
    source,
    mode="simple",
    cells=None,
    discord=True,
    scale=None,
    form=True,
    upscale=True,
    line=None,
    simple=None,
    details=None,
    line_thr=None,
    details_thr=None,
    smile_px=None,
):
    mode_name = resolve_mode(mode)
    line_opts = None
    if mode_name in ("line", "render", "details"):
        line_opts = _mode_line_defaults(
            mode_name, line=line, details=details, line_thr=line_thr, details_thr=details_thr
        )
    paint_mode = _paint_mode(mode_name)
    smile = OUTPUT_SMILE_PX if smile_px is None else max(1, int(smile_px))
    # CUDA atlas is 8px-only for now; non-default smile_px uses CPU paint.
    if not discord and paint_mode in ("line", "simple") and smile == OUTPUT_SMILE_PX:
        from core.image.render import pick_device, render_frame_array

        if pick_device().type == "cuda":
            array = render_frame_array(
                source,
                mode=mode_name,
                cells=cells,
                discord=False,
                scale=scale,
                upscale=upscale,
                line=line_opts,
                simple=simple,
            )
            return Image.fromarray(array, mode="L")
    indexes, resolved = pic_indexes(
        source,
        mode=mode_name,
        cells=cells,
        discord=discord,
        form=form,
        scale=scale,
        upscale=upscale,
        line=line_opts,
        simple=simple,
    )
    paint_chars = None
    if resolved == "simple":
        paint_chars = simple_paint_chars(simple)
    elif _paint_mode(resolved) == "line":
        paint_chars = LINE_PAINT_CHARS
    return indexes_to_image(indexes, mode=paint_mode, chars=paint_chars, smile_px=smile)


def _smile_font(size):
    cached = _GLYPH_FONTS.get(size)
    if cached is not None:
        return cached
    for path in _FONT_CANDIDATES:
        if not path.exists():
            continue
        try:
            font = ImageFont.truetype(str(path), size=size)
            _GLYPH_FONTS[size] = font
            return font
        except OSError:
            continue
    font = ImageFont.load_default()
    _GLYPH_FONTS[size] = font
    return font


def _font_at(size, path):
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return None


def _draw_char_cell(char, size):
    if not char or char.isspace():
        return np.zeros((size, size), dtype=np.uint8)
    big = max(48, size * 8)
    center = (big // 2, big // 2)

    def paint(font):
        canvas = Image.new("L", (big, big), 0)
        draw = ImageDraw.Draw(canvas)
        draw.text(center, char, fill=255, font=font, anchor="mm")
        return np.asarray(canvas.resize((size, size), Image.LANCZOS), dtype=np.uint8)

    tile = paint(_smile_font(big))
    if tile.max() > 0:
        return tile
    for path in _FONT_CANDIDATES:
        if not path.exists():
            continue
        font = _font_at(big, path)
        if font is None:
            continue
        tile = paint(font)
        if tile.max() > 0:
            return tile
    return tile


def _char_tile_cache():
    return _GLYPH_ATLAS.setdefault("_chars", {})


def _tile_for_char(char, size=OUTPUT_SMILE_PX):
    cache = _char_tile_cache()
    key = (char, size)
    tile = cache.get(key)
    if tile is None:
        tile = _draw_char_cell(char, size)
        cache[key] = tile
    return tile


def _row_symbols(row, lookup):
    keys = sorted(lookup.keys(), key=len, reverse=True)
    i = 0
    while i < len(row):
        matched = None
        for key in keys:
            if row.startswith(key, i):
                matched = key
                break
        if matched is None:
            matched = row[i]
        yield matched
        i += len(matched)


def text_grid_to_png(text, mode="line"):
    body = art_body(text).replace("```", "").strip("\n")
    if not body:
        return Image.new("L", (0, 0))
    return indexes_to_image(parse_art(text, mode=mode), mode=mode)


def glyph_atlas(mode="line", smile_px=None):
    # Те же тайлы, что text_grid_to_png, по индексу словаря для anim/batch.
    mode_name = _paint_mode(resolve_mode(mode))
    size = OUTPUT_SMILE_PX if smile_px is None else max(1, int(smile_px))
    cache_key = (mode_name, size)
    cached = _GLYPH_ATLAS.get(cache_key)
    if cached is not None:
        return cached
    if mode_name == "line":
        chars = LINE_PAINT_CHARS
    else:
        chars = simple_paint_chars(None)
    atlas = np.zeros((len(chars), size, size), dtype=np.uint8)
    for index, char in enumerate(chars.tolist()):
        atlas[index] = _tile_for_char(char, size)
    _GLYPH_ATLAS[cache_key] = atlas
    return atlas


def indexes_to_image(indexes, mode="line", chars=None, smile_px=None):
    # smile_px = font/tile size in output PNG (default OUTPUT_SMILE_PX=8).
    indexes = np.asarray(indexes, dtype=np.int32)
    rows, cols = indexes.shape
    smile = OUTPUT_SMILE_PX if smile_px is None else max(1, int(smile_px))
    if rows == 0 or cols == 0:
        return Image.new("L", (cols * smile, rows * smile))
    if chars is not None:
        char_list = np.asarray(chars)
        atlas = np.zeros((len(char_list), smile, smile), dtype=np.uint8)
        for index, char in enumerate(char_list.tolist()):
            atlas[index] = _tile_for_char(char, smile)
    else:
        atlas = glyph_atlas(mode, smile_px=smile)
    last = atlas.shape[0] - 1
    canvas = np.zeros((rows * smile, cols * smile), dtype=np.uint8)
    for y in range(rows):
        for x in range(cols):
            canvas[y * smile:(y + 1) * smile, x * smile:(x + 1) * smile] = atlas[
                int(np.clip(indexes[y, x], 0, last))
            ]
    return Image.fromarray(canvas, mode="L")


def _split_header(text):
    if text.startswith(HEADER):
        return HEADER + "\n", text[len(HEADER):].lstrip("\n")
    return "", text


def fence_pic(text):
    header, body = _split_header(text)
    return f"{header}```\n{body}\n```"


def art_body(text):
    _, body = _split_header(text)
    return body


def fence_md(text):
    # CLI markdown: только сырые символы; превью рисует как есть.
    return art_body(text)


def parse_art(text, mode="line"):
    body = art_body(text).replace("```", "").strip("\n")
    if not body:
        return np.zeros((0, 0), dtype=np.int32)
    rows = body.split("\n")
    mode_name = _paint_mode(resolve_mode(mode))
    chars = LINE_PAINT_CHARS if mode_name == "line" else simple_paint_chars(None)
    lookup = {char: index for index, char in enumerate(chars.tolist())}
    cells = [list(_row_symbols(row, lookup)) for row in rows]
    width = max((len(row) for row in cells), default=0)
    indexes = np.zeros((len(rows), width), dtype=np.int32)
    for y, row in enumerate(cells):
        for x, char in enumerate(row):
            indexes[y, x] = lookup.get(char, 0)
    return indexes


def md_to_png(text, mode="line"):
    return text_grid_to_png(text, mode=mode)


def resolve_mode(mode):
    key = str(mode or "simple").strip().lower()
    if key in {"1", "line", "lines", "cool"}:
        return "line"
    if key in {"2", "simple", "color", "avg"}:
        return "simple"
    if key in {"3", "render", "pic"}:
        return "render"
    if key in {"4", "details", "detail", "soft"}:
        return "details"
    raise ValueError(f"Unknown pic mode: {mode}")


def discord_chunks(text, limit=None, max_rows=DISCORD_MAX_ROWS):
    # Режем по строкам (31 строка █ на fence). limit — для узких тестов.
    header, body = _split_header(text)
    lines = body.split("\n") if body else []
    fence = 8
    max_rows = max(1, int(max_rows))
    chunks = []
    current_lines = []
    first = True

    def flush():
        nonlocal current_lines, first
        if not current_lines and chunks:
            return
        current = "\n".join(current_lines)
        prefix = header if first else ""
        chunks.append(f"{prefix}```\n{current}\n```")
        current_lines = []
        first = False

    for line in lines:
        trial = current_lines + [line]
        over_rows = len(trial) > max_rows
        over_chars = False
        if limit is not None:
            body_trial = "\n".join(trial)
            budget = limit - fence - (_discord_len(header) if first else 0)
            over_chars = _discord_len(body_trial) > budget
        if current_lines and (over_rows or over_chars):
            flush()
            current_lines = [line]
        else:
            current_lines = trial
    if current_lines or not chunks:
        flush()
    return chunks
