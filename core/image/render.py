import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.image.ascii import (
    CLI_DEFAULT_SCALE,
    DETAILS_PATTERNS,
    DETAILS_THR,
    LINE_THR,
    OUTPUT_SMILE_PX,
    _DETAILS_BASE,
    _MEASURED_COUNT,
    _SHADE_COUNT,
    _SHADE_HI,
    LINE_PATTERNS,
    SIMPLE_COLORS,
    glyph_atlas,
    line_options,
    load_rgb,
    resolve_frame,
    resolve_mode,
    simple_paint_chars,
    simple_quantize_steps,
)

_WORKER = {}
GPU_BATCH = 4
_QUANTILE_MAX = 1 << 20


def _quantile_1d(sample, q):
    n = int(sample.numel())
    if n <= _QUANTILE_MAX:
        return torch.quantile(sample, q)
    step = max(1, n // _QUANTILE_MAX)
    return torch.quantile(sample[::step][:_QUANTILE_MAX], q)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_gpu_batch(height, width, device=None):
    # Target ~80% VRAM in use, keep ~20% free. No source downscale.
    device = device or pick_device()
    if device.type != "cuda":
        return 1
    pixels = max(1, int(height) * int(width))
    per_frame = pixels * 3 * 4 * 5
    try:
        free, total = torch.cuda.mem_get_info(device)
    except Exception:
        free, total = 2 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024
    used = max(0, int(total) - int(free))
    budget = max(0, int(total * 0.80) - used)
    batch = max(1, budget // max(per_frame, 1))
    if pixels >= 3840 * 2000:
        batch = min(batch, 8)
    elif pixels >= 1920 * 1000:
        batch = min(batch, 20)
    else:
        batch = min(batch, 40)
    return max(1, int(batch))


def _patterns_tensor(device):
    return torch.as_tensor(LINE_PATTERNS, dtype=torch.float32, device=device)


def _details_tensor(device):
    return torch.as_tensor(DETAILS_PATTERNS, dtype=torch.float32, device=device)


def _rec601_torch(rgb):
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0


def oklab_L_torch(rgb):
    # rgb: (..., H, W, 3) in 0..255 → OKLab L ≈ [0,1]
    r = (rgb[..., 0] / 255.0).clamp(0.0, 1.0)
    g = (rgb[..., 1] / 255.0).clamp(0.0, 1.0)
    b = (rgb[..., 2] / 255.0).clamp(0.0, 1.0)

    def _linear(c):
        return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))

    r, g, b = _linear(r), _linear(g), _linear(b)
    l_ = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m_ = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s_ = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l = torch.pow(l_.clamp(min=0.0), 1.0 / 3.0)
    m = torch.pow(m_.clamp(min=0.0), 1.0 / 3.0)
    s = torch.pow(s_.clamp(min=0.0), 1.0 / 3.0)
    return 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s


def structure_field_torch(rgb):
    # Match CPU structure_field; rgb (B,H,W,3) or (H,W,3) in 0..255.
    single = rgb.ndim == 3
    if single:
        rgb = rgb.unsqueeze(0)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    chroma = torch.maximum(
        torch.maximum((r - g).abs(), (g - b).abs()),
        (b - r).abs(),
    ) / 255.0
    dx = (rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).abs().mean(dim=-1)
    dy = (rgb[:, 1:, :, :] - rgb[:, :-1, :, :]).abs().mean(dim=-1)
    grad = torch.zeros(rgb.shape[:3], dtype=rgb.dtype, device=rgb.device)
    grad[:, :, 1:] = grad[:, :, 1:] + dx
    grad[:, :, :-1] = grad[:, :, :-1] + dx
    grad[:, 1:, :] = grad[:, 1:, :] + dy
    grad[:, :-1, :] = grad[:, :-1, :] + dy
    grad = (grad / (4.0 * 255.0)).clamp(0.0, 1.0)
    out = (0.5 * chroma + 0.5 * grad).clamp(0.0, 1.0)
    return out[0] if single else out


def _tone_field_torch(rgb, opts, luma=None):
    if opts.get("tone") == "oklab":
        return oklab_L_torch(rgb)
    return luma if luma is not None else _rec601_torch(rgb)


def _structure_source_torch(rgb, opts, luma=None):
    if opts.get("structure") == "color":
        return structure_field_torch(rgb)
    return luma if luma is not None else _rec601_torch(rgb)


def _stretch_torch(gray, gamma=0.72):
    # gray: (H, W) или (B, H, W); квантили на том же device.
    single = gray.ndim == 2
    if single:
        gray = gray.unsqueeze(0)
    batch, height, width = gray.shape
    flat = gray.reshape(batch, -1)
    min_lit = max(32, flat.shape[1] // 20)
    frames = []
    for index in range(batch):
        row = flat[index]
        lit = row[row > 0.04]
        sample = lit if lit.numel() >= min_lit else row
        lo = _quantile_1d(sample, 0.05)
        hi = _quantile_1d(sample, 0.98)
        span = hi - lo
        stretched = torch.where(
            span < 1e-4,
            torch.clamp(row, 0.0, 1.0),
            torch.clamp((row - lo) / span.clamp(min=1e-4), 0.0, 1.0),
        )
        frames.append(stretched.reshape(height, width).pow(gamma))
    stacked = torch.stack(frames, dim=0)
    return stacked[0] if single else stacked


def _pattern_indexes_torch(quads, patterns):
    # quads (..., 4), patterns (P, 4): cdist даёт (N, P), а не (H, W, P, 4).
    shape = quads.shape[:-1]
    flat = quads.reshape(-1, 4)
    return torch.cdist(flat, patterns).argmin(dim=-1).reshape(shape)


def _window_origins_torch(length, count, win, device):
    count = max(1, int(count))
    last = max(0, int(length) - int(win))
    if count == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    origins = torch.linspace(0, last, count, device=device).round().to(torch.long)
    return origins.clamp(0, last)


def _maybe_upscale_torch(gray, rows, cols, scale, upscale):
    win = 2 * max(1, int(scale))
    target_h = max(1, int(rows)) * win
    target_w = max(1, int(cols)) * win
    height, width = int(gray.shape[-2]), int(gray.shape[-1])
    if height == target_h and width == target_w:
        return gray
    if upscale or (height >= target_h and width >= target_w):
        return F.interpolate(gray.unsqueeze(1), size=(target_h, target_w), mode="nearest")[:, 0]
    crop_h = min(height, target_h)
    crop_w = min(width, target_w)
    cropped = gray[:, :crop_h, :crop_w]
    return F.pad(cropped, (0, target_w - crop_w, 0, target_h - crop_h), mode="replicate")


def _smile_blocks_torch(gray, rows, cols, scale, reduce="mean"):
    # gray (B, H, W): квадрант — сплошной блок scale×scale.
    scale = max(1, int(scale))
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    win = 2 * scale
    height, width = int(gray.shape[-2]), int(gray.shape[-1])
    pad_h = max(0, win - height)
    pad_w = max(0, win - width)
    if pad_h or pad_w:
        gray = F.pad(gray, (0, pad_w, 0, pad_h), mode="replicate")
        height, width = int(gray.shape[-2]), int(gray.shape[-1])
    if height == rows * win and width == cols * win:
        view = gray.reshape(gray.shape[0], rows, win, cols, win)
        windows = view.permute(0, 1, 3, 2, 4)
    else:
        ys = _window_origins_torch(height, rows, win, gray.device)
        xs = _window_origins_torch(width, cols, win, gray.device)
        y_off = torch.arange(win, device=gray.device)
        x_off = torch.arange(win, device=gray.device)
        yy = ys.view(rows, 1, 1, 1) + y_off.view(1, 1, win, 1)
        xx = xs.view(1, cols, 1, 1) + x_off.view(1, 1, 1, win)
        windows = gray[:, yy, xx]
    half = scale

    def _red(block):
        if reduce == "max":
            return block.amax(dim=(-1, -2))
        return block.mean(dim=(-1, -2))

    quads = torch.stack(
        [
            _red(windows[:, :, :, 0:half, 0:half]),
            _red(windows[:, :, :, 0:half, half:win]),
            _red(windows[:, :, :, half:win, 0:half]),
            _red(windows[:, :, :, half:win, half:win]),
        ],
        dim=-1,
    )
    return quads, windows.mean(dim=(-1, -2))


def _shade_indexes_torch(means):
    # Mirror ascii.shade_indexes: searchsorted on right edges hi (side=left).
    means = means.clamp(0.0, 1.0)
    hi = torch.as_tensor(_SHADE_HI, device=means.device, dtype=means.dtype)
    flat = means.reshape(-1)
    idx = torch.searchsorted(hi, flat, right=False).reshape(means.shape)
    return idx.clamp(max=_SHADE_COUNT - 1)


def _compose_line_torch(
    quads,
    means,
    patterns,
    details_patterns=None,
    details=False,
    line_thr=None,
    details_thr=None,
):
    bg = _shade_indexes_torch(means)
    if not details:
        # Classic NN for line/render — no hard spread gate.
        nearest = _pattern_indexes_torch(quads, patterns[:_SHADE_COUNT + _MEASURED_COUNT])
        return torch.where(nearest >= _SHADE_COUNT, nearest, bg)
    line_thr = LINE_THR if line_thr is None else float(line_thr)
    details_thr = DETAILS_THR if details_thr is None else float(details_thr)
    spread = quads.amax(dim=-1) - quads.amin(dim=-1)
    measured = patterns[_SHADE_COUNT : _SHADE_COUNT + _MEASURED_COUNT]
    line_idx = _pattern_indexes_torch(quads, measured) + _SHADE_COUNT
    use_line = spread >= line_thr
    if details_patterns is None:
        details_patterns = _details_tensor(quads.device)
    detail_idx = _pattern_indexes_torch(quads, details_patterns) + _DETAILS_BASE
    use_details = (~use_line) & (spread >= details_thr)
    return torch.where(use_line, line_idx, torch.where(use_details, detail_idx, bg))


def _tint_line_quads_torch(photo, mask, contrast):
    on = mask > 0.05
    off = ~on
    count = off.sum(dim=-1, keepdim=True).clamp(min=1).to(photo.dtype)
    neighbor = (photo * off.to(photo.dtype)).sum(dim=-1, keepdim=True) / count
    fallback = photo.mean(dim=-1, keepdim=True)
    neighbor = torch.where(on.all(dim=-1, keepdim=True), fallback, neighbor)
    ink = neighbor * (1.0 - contrast)
    return torch.where(on, ink, photo)


def _compose_tinted_line_torch(
    photo,
    mask,
    means,
    patterns,
    contrast,
    details_patterns=None,
    details=False,
    line_thr=None,
    details_thr=None,
):
    tinted = _tint_line_quads_torch(photo, mask, contrast)
    bg = _shade_indexes_torch(means)
    composed = _compose_line_torch(
        tinted,
        means,
        patterns,
        details_patterns=details_patterns,
        details=details,
        line_thr=line_thr,
        details_thr=details_thr,
    )
    covered = mask.amax(dim=-1) > 0.05
    return torch.where(covered, composed, bg)


def _compose_masked_line_torch(
    ink,
    means,
    patterns,
    details_patterns=None,
    details=False,
    line_thr=None,
    details_thr=None,
):
    indexes = _compose_line_torch(
        ink,
        means,
        patterns,
        details_patterns=details_patterns,
        details=details,
        line_thr=line_thr,
        details_thr=details_thr,
    )
    covered = ink.amin(dim=-1) < 1.0 - 1e-6
    measured = patterns[_SHADE_COUNT : _SHADE_COUNT + _MEASURED_COUNT]
    ink_idx = _pattern_indexes_torch(ink, measured) + _SHADE_COUNT
    flat = indexes < _SHADE_COUNT
    return torch.where(covered & flat, ink_idx, indexes)


def _raster_mask_torch(mask, rows, cols, scale, upscale):
    win = 2 * max(1, int(scale))
    target_h = max(1, int(rows)) * win
    target_w = max(1, int(cols)) * win
    height, width = int(mask.shape[-2]), int(mask.shape[-1])
    if height == target_h and width == target_w:
        return mask
    if not (upscale or (height >= target_h and width >= target_w)):
        crop_h = min(height, target_h)
        crop_w = min(width, target_w)
        cropped = mask[:, :crop_h, :crop_w]
        return F.pad(cropped, (0, target_w - crop_w, 0, target_h - crop_h), mode="replicate")
    if height >= target_h and width >= target_w:
        return F.adaptive_max_pool2d(mask.unsqueeze(1), (target_h, target_w))[:, 0]
    return F.interpolate(mask.unsqueeze(1), size=(target_h, target_w), mode="nearest")[:, 0]


def _line_masks_torch(gray, opts, masks_cpu=None):
    if masks_cpu is not None:
        return torch.as_tensor(masks_cpu, dtype=torch.float32, device=gray.device)
    from core.image.line_mask_cuda import line_masks_torch

    return line_masks_torch(gray, opts).to(dtype=torch.float32)


_ATLAS_CACHE = {}


def _atlas_tensor(mode, device, simple=None):
    from core.image.ascii import _paint_mode, _tile_for_char

    mode_name = resolve_mode(mode)
    paint = _paint_mode(mode_name)
    if paint == "simple":
        chars = simple_paint_chars(simple)
        key = ("simple", tuple(chars.tolist()), device.type, getattr(device, "index", None))
        cached = _ATLAS_CACHE.get(key)
        if cached is None or cached.device != device:
            size = OUTPUT_SMILE_PX
            atlas = np.zeros((len(chars), size, size), dtype=np.uint8)
            for index, char in enumerate(chars.tolist()):
                atlas[index] = _tile_for_char(char, size)
            cached = torch.as_tensor(atlas, dtype=torch.uint8, device=device)
            _ATLAS_CACHE[key] = cached
        return cached
    key = ("line", device.type, getattr(device, "index", None))
    cached = _ATLAS_CACHE.get(key)
    if cached is None or cached.device != device:
        cached = torch.as_tensor(glyph_atlas("line"), dtype=torch.uint8, device=device)
        _ATLAS_CACHE[key] = cached
    return cached


def _indexes_to_u8_batch(indexes, mode, patterns, device, simple=None):
    del patterns
    smile = OUTPUT_SMILE_PX
    if indexes.ndim == 2:
        indexes = indexes.unsqueeze(0)
    batch, rows, cols = indexes.shape
    atlas = _atlas_tensor(mode, device, simple=simple)
    tiles = atlas[indexes.long().clamp(0, atlas.shape[0] - 1)]
    return tiles.permute(0, 1, 3, 2, 4).reshape(batch, rows * smile, cols * smile)


def line_indexes_torch(
    rgb,
    cells=None,
    discord=False,
    device=None,
    patterns=None,
    scale=None,
    upscale=True,
    line=None,
):
    device = device or pick_device()
    patterns = patterns if patterns is not None else _patterns_tensor(device)
    scale = CLI_DEFAULT_SCALE if scale is None else max(1, int(scale))
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    array = np.clip(rgb, 0, 255).astype(np.uint8)
    rendered = _indexes_from_frames(
        array[None, ...],
        mode="line",
        cells=cells,
        discord=discord,
        scale=scale,
        upscale=upscale,
        device=device,
        patterns=patterns,
        line=line,
    )
    return rendered[0]


def simple_indexes_torch(
    rgb, cells=None, discord=False, device=None, scale=None, upscale=True, line=None, simple=None
):
    device = device or pick_device()
    scale = CLI_DEFAULT_SCALE if scale is None else max(1, int(scale))
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    array = np.clip(rgb, 0, 255).astype(np.uint8)
    rendered = _indexes_from_frames(
        array[None, ...],
        mode="simple",
        cells=cells,
        discord=discord,
        scale=scale,
        upscale=upscale,
        device=device,
        patterns=None,
        line=line,
        simple=simple,
    )
    return rendered[0]


def indexes_to_uint8_torch(indexes, mode="line", patterns=None, device=None, simple=None):
    device = device or indexes.device
    patterns = patterns if patterns is not None else _patterns_tensor(device)
    painted = _indexes_to_u8_batch(indexes, mode, patterns, device, simple=simple)
    frame = painted[0] if painted.ndim == 3 else painted
    return frame.cpu().numpy()


def _frames_to_rgb_tensor(frames, device, non_blocking=False):
    if torch.is_tensor(frames):
        rgb = frames.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        return rgb
    array = np.asarray(frames)
    if array.ndim == 3:
        array = array[None, ...]
    if non_blocking and device.type == "cuda":
        host = torch.from_numpy(np.ascontiguousarray(array))
        try:
            host = host.pin_memory()
        except RuntimeError:
            pass
        return host.to(device=device, dtype=torch.float32, non_blocking=True)
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def prepare_mask_source(frames, mode="line", line=None, device=None, non_blocking=False):
    # GPU tone/structure for file line/render; returns CPU gray for line_mask + GPU rgb.
    device = device or pick_device()
    rgb = _frames_to_rgb_tensor(frames, device, non_blocking=non_blocking)
    opts = line_options(**(line or {}))
    mode_name = resolve_mode(mode)
    luma = _rec601_torch(rgb) if opts["tone"] == "rec601" or opts["structure"] == "luma" else None
    if opts["tone"] != "rec601" or opts["structure"] != "luma":
        struct = _structure_source_torch(rgb, opts, luma=luma)
        if opts["structure"] == "color":
            struct = _stretch_torch(struct)
    else:
        struct = luma if luma is not None else _rec601_torch(rgb)
    return {
        "rgb": rgb,
        "opts": opts,
        "mode_name": mode_name,
        "mask_cpu": struct.detach().cpu().numpy(),
    }


def _indexes_from_frames(
    frames,
    mode,
    cells,
    discord,
    scale,
    upscale,
    device,
    patterns,
    line=None,
    masks_cpu=None,
    rgb_tensor=None,
    simple=None,
):
    if rgb_tensor is not None:
        rgb = rgb_tensor
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
    else:
        rgb = _frames_to_rgb_tensor(frames, device)
    from core.image.ascii import _mode_line_defaults, _paint_mode

    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    scale = CLI_DEFAULT_SCALE if scale is None else max(1, int(scale))
    cols, rows = resolve_frame(width, height, cells, scale, upscale, discord)
    mode_name = resolve_mode(mode)
    paint = _paint_mode(mode_name)
    patterns = patterns if patterns is not None else _patterns_tensor(device)
    from core.image.ascii import _adapt_mask_opts

    if mode_name in ("line", "render", "details"):
        opts = _adapt_mask_opts(_mode_line_defaults(mode_name, line=line), scale)
    else:
        opts = _adapt_mask_opts(line_options(**(line or {})), scale)
    compose_kw = {
        "details_patterns": _details_tensor(device),
        "details": opts["details"],
        "line_thr": opts["line_thr"],
        "details_thr": opts["details_thr"],
    }
    split = opts["tone"] != "rec601" or opts["structure"] != "luma"
    stretch_color_struct = opts["structure"] == "color" and scale < 24
    if not split:
        gray = _rec601_torch(rgb)
        if paint == "line" and not discord:
            mask = _raster_mask_torch(
                _line_masks_torch(gray, opts, masks_cpu=masks_cpu),
                rows,
                cols,
                scale,
                upscale,
            )
            mask_quads, _unused = _smile_blocks_torch(mask, rows, cols, scale, reduce="max")
            tone = _stretch_torch(_maybe_upscale_torch(gray, rows, cols, scale, upscale))
            photo, means = _smile_blocks_torch(tone, rows, cols, scale)
            return _compose_tinted_line_torch(
                photo, mask_quads, means, patterns, opts["contrast"], **compose_kw
            )
        gray = _maybe_upscale_torch(gray, rows, cols, scale, upscale)
        tone = _stretch_torch(gray)
        quads, means = _smile_blocks_torch(tone, rows, cols, scale)
        if paint == "line":
            return _compose_line_torch(quads, means, patterns, **compose_kw)
        steps = simple_quantize_steps(simple)
        return torch.round(torch.clamp(means, 0.0, 1.0) * steps).to(torch.int32)
    luma = _rec601_torch(rgb) if opts["tone"] == "rec601" or opts["structure"] == "luma" else None
    tone_raw = _tone_field_torch(rgb, opts, luma=luma)
    if paint == "line" and not discord:
        struct_raw = _structure_source_torch(rgb, opts, luma=luma)
        if stretch_color_struct:
            struct_raw = _stretch_torch(struct_raw)
        mask = _raster_mask_torch(
            _line_masks_torch(struct_raw, opts, masks_cpu=masks_cpu),
            rows,
            cols,
            scale,
            upscale,
        )
        mask_quads, _unused = _smile_blocks_torch(mask, rows, cols, scale, reduce="max")
        tone = _stretch_torch(_maybe_upscale_torch(tone_raw, rows, cols, scale, upscale))
        photo, means = _smile_blocks_torch(tone, rows, cols, scale)
        return _compose_tinted_line_torch(
            photo, mask_quads, means, patterns, opts["contrast"], **compose_kw
        )
    tone_raw = _maybe_upscale_torch(tone_raw, rows, cols, scale, upscale)
    tone = _stretch_torch(tone_raw)
    quads, means = _smile_blocks_torch(tone, rows, cols, scale)
    if paint == "line":
        return _compose_line_torch(quads, means, patterns, **compose_kw)
    steps = simple_quantize_steps(simple)
    return torch.round(torch.clamp(means, 0.0, 1.0) * steps).to(torch.int32)


def render_rgb_batch(
    frames,
    mode="line",
    cells=None,
    discord=False,
    scale=2,
    device=None,
    patterns=None,
    cap=False,
    upscale=True,
    line=None,
    masks_cpu=None,
    rgb_tensor=None,
    simple=None,
):
    # frames uint8 (B,H,W,3) или (H,W,3); cap для API; сетка через resolve_frame.
    del cap
    device = device or pick_device()
    indexes = _indexes_from_frames(
        frames,
        mode=mode,
        cells=cells,
        discord=discord,
        scale=scale,
        upscale=upscale,
        device=device,
        patterns=patterns,
        line=line,
        masks_cpu=masks_cpu,
        rgb_tensor=rgb_tensor,
        simple=simple,
    )
    patterns = patterns if patterns is not None else _patterns_tensor(device)
    return _indexes_to_u8_batch(indexes, mode, patterns, device, simple=simple)


def render_frame_array(
    source,
    mode="line",
    cells=None,
    discord=False,
    scale=None,
    device=None,
    patterns=None,
    upscale=True,
    line=None,
    simple=None,
):
    device = device or pick_device()
    from core.image.ascii import _mode_line_defaults, _paint_mode

    mode_name = resolve_mode(mode)
    paint = _paint_mode(mode_name)
    line_opts = line
    if mode_name in ("line", "render", "details"):
        line_opts = _mode_line_defaults(mode_name, line=line)
    rgb = load_rgb(source)
    if paint == "line":
        indexes = line_indexes_torch(
            rgb,
            cells=cells,
            discord=discord,
            device=device,
            patterns=patterns,
            scale=scale,
            upscale=upscale,
            line=line_opts,
        )
    else:
        indexes = simple_indexes_torch(
            rgb,
            cells=cells,
            discord=discord,
            device=device,
            scale=scale,
            upscale=upscale,
            line=line_opts,
            simple=simple,
        )
    return indexes_to_uint8_torch(
        indexes, mode=paint, patterns=patterns, device=device, simple=simple
    )


def render_frame_to_path(
    source,
    out_path,
    mode="line",
    cells=None,
    discord=False,
    scale=None,
    device=None,
    patterns=None,
    upscale=True,
    line=None,
):
    array = render_frame_array(
        source,
        mode=mode,
        cells=cells,
        discord=discord,
        scale=scale,
        device=device,
        patterns=patterns,
        upscale=upscale,
        line=line,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(out_path)
    return out_path


def _init_cpu_worker(mode, cells, discord, scale, upscale, line):
    _WORKER["mode"] = mode
    _WORKER["cells"] = cells
    _WORKER["discord"] = discord
    _WORKER["scale"] = scale
    _WORKER["upscale"] = upscale
    _WORKER["line"] = line
    _WORKER["device"] = torch.device("cpu")
    _WORKER["patterns"] = _patterns_tensor(_WORKER["device"])


def _cpu_worker_job(payload):
    index, source, out_path = payload
    render_frame_to_path(
        source,
        out_path,
        mode=_WORKER["mode"],
        cells=_WORKER["cells"],
        discord=_WORKER["discord"],
        scale=_WORKER["scale"],
        upscale=_WORKER["upscale"],
        line=_WORKER["line"],
        device=_WORKER["device"],
    )
    return index


def render_frame_paths(
    sources,
    out_dir,
    mode="line",
    cells=None,
    discord=False,
    scale=None,
    batch_size=8,
    workers=None,
    progress=True,
    upscale=True,
    line=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    total = len(sources)
    outs = [out_dir / f"frame_{index + 1:06d}.png" for index in range(total)]

    if device.type == "cuda":
        patterns = _patterns_tensor(device)
        last_report = 0
        for start in range(0, total, max(1, int(batch_size))):
            end = min(total, start + max(1, int(batch_size)))
            for index in range(start, end):
                render_frame_to_path(
                    sources[index],
                    outs[index],
                    mode=mode,
                    cells=cells,
                    discord=discord,
                    scale=scale,
                    device=device,
                    patterns=patterns,
                    upscale=upscale,
                    line=line,
                )
            if progress and (end == total or end - last_report >= 25 or start == 0):
                print(f"anim {end}/{total}", flush=True)
                last_report = end
        return outs

    workers = max(1, int(workers or max(1, (os.cpu_count() or 2) - 1)))
    jobs = [(index, sources[index], outs[index]) for index in range(total)]
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_cpu_worker,
        initargs=(mode, cells, discord, scale, upscale, line),
    ) as pool:
        futures = [pool.submit(_cpu_worker_job, job) for job in jobs]
        for future in as_completed(futures):
            future.result()
            done += 1
            if progress and (done == total or done % 25 == 0 or done == 1):
                print(f"anim {done}/{total}", flush=True)
    return outs
