import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.image.ascii import (
    CLI_DEFAULT_SCALE,
    OUTPUT_SMILE_PX,
    _SHADE_COUNT,
    LINE_PATTERNS,
    SIMPLE_COLORS,
    glyph_atlas,
    line_mask,
    line_options,
    load_rgb,
    resolve_frame,
    resolve_mode,
)

_WORKER = {}
GPU_BATCH = 4
_QUANTILE_MAX = 1 << 20


def _quantile_1d(sample, q):
    n = int(sample.numel())
    if n <= _QUANTILE_MAX:
        return torch.quantile(sample, q)
    step = max(1, n // _QUANTILE_MAX)
    return torch.quantile(sample[::step][: _QUANTILE_MAX], q)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _patterns_tensor(device):
    return torch.as_tensor(LINE_PATTERNS, dtype=torch.float32, device=device)


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


def _compose_line_torch(quads, means, patterns):
    shade_levels = patterns[:_SHADE_COUNT, 0]
    bg = (means.unsqueeze(-1) - shade_levels).pow(2).argmin(dim=-1)
    line = _pattern_indexes_torch(quads, patterns)
    return torch.where(line >= _SHADE_COUNT, line, bg)


def _tint_line_quads_torch(photo, mask, contrast):
    on = mask > 0.05
    off = ~on
    count = off.sum(dim=-1, keepdim=True).clamp(min=1).to(photo.dtype)
    neighbor = (photo * off.to(photo.dtype)).sum(dim=-1, keepdim=True) / count
    fallback = photo.mean(dim=-1, keepdim=True)
    neighbor = torch.where(on.all(dim=-1, keepdim=True), fallback, neighbor)
    ink = neighbor * (1.0 - contrast)
    return torch.where(on, ink, photo)


def _compose_tinted_line_torch(photo, mask, means, patterns, contrast):
    tinted = _tint_line_quads_torch(photo, mask, contrast)
    shade_levels = patterns[:_SHADE_COUNT, 0]
    bg = (means.unsqueeze(-1) - shade_levels).pow(2).argmin(dim=-1)
    line = _pattern_indexes_torch(tinted, patterns)
    covered = mask.amax(dim=-1) > 0.05
    return torch.where(covered, line, bg)


def _compose_masked_line_torch(ink, means, patterns):
    indexes = _compose_line_torch(ink, means, patterns)
    covered = ink.amin(dim=-1) < 1.0 - 1e-6
    ink_idx = _pattern_indexes_torch(ink, patterns)
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


def _line_masks_torch(gray, opts):
    frames = gray.detach().cpu().numpy()
    masks = np.stack(
        [
            line_mask(
                frames[index],
                threshold=opts["threshold"],
                min_size=opts["min_size"],
                step=opts["step"],
                depth=opts["depth"],
            )
            for index in range(frames.shape[0])
        ]
    )
    return torch.as_tensor(masks, dtype=torch.float32, device=gray.device)


def _indexes_to_u8_batch(indexes, mode, patterns, device):
    del patterns
    mode_name = resolve_mode(mode)
    smile = OUTPUT_SMILE_PX
    if indexes.ndim == 2:
        indexes = indexes.unsqueeze(0)
    batch, rows, cols = indexes.shape
    atlas = torch.as_tensor(glyph_atlas(mode_name), dtype=torch.uint8, device=device)
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


def simple_indexes_torch(rgb, cells=None, discord=False, device=None, scale=None, upscale=True):
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
    )
    return rendered[0]


def indexes_to_uint8_torch(indexes, mode="line", patterns=None, device=None):
    device = device or indexes.device
    patterns = patterns if patterns is not None else _patterns_tensor(device)
    painted = _indexes_to_u8_batch(indexes, mode, patterns, device)
    frame = painted[0] if painted.ndim == 3 else painted
    return frame.cpu().numpy()


def _indexes_from_frames(frames, mode, cells, discord, scale, upscale, device, patterns, line=None):
    array = np.asarray(frames)
    if array.ndim == 3:
        array = array[None, ...]
    rgb = torch.as_tensor(array, dtype=torch.float32, device=device)
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    scale = CLI_DEFAULT_SCALE if scale is None else max(1, int(scale))
    cols, rows = resolve_frame(width, height, cells, scale, upscale, discord)
    mode_name = resolve_mode(mode)
    patterns = patterns if patterns is not None else _patterns_tensor(device)
    opts = line_options(**(line or {}))
    gray = (0.299 * rgb[:, :, :, 0] + 0.587 * rgb[:, :, :, 1] + 0.114 * rgb[:, :, :, 2]) / 255.0
    if mode_name == "line" and not discord:
        mask = _raster_mask_torch(_line_masks_torch(gray, opts), rows, cols, scale, upscale)
        mask_quads, _unused = _smile_blocks_torch(mask, rows, cols, scale, reduce="max")
        tone = _stretch_torch(_maybe_upscale_torch(gray, rows, cols, scale, upscale))
        photo, means = _smile_blocks_torch(tone, rows, cols, scale)
        return _compose_tinted_line_torch(photo, mask_quads, means, patterns, opts["contrast"])
    gray = _maybe_upscale_torch(gray, rows, cols, scale, upscale)
    tone = _stretch_torch(gray)
    quads, means = _smile_blocks_torch(tone, rows, cols, scale)
    if mode_name == "line":
        return _compose_line_torch(quads, means, patterns)
    steps = SIMPLE_COLORS.shape[0] - 1
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
    )
    patterns = patterns if patterns is not None else _patterns_tensor(device)
    return _indexes_to_u8_batch(indexes, mode, patterns, device)


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
):
    device = device or pick_device()
    mode_name = resolve_mode(mode)
    rgb = load_rgb(source)
    if mode_name == "line":
        indexes = line_indexes_torch(
            rgb,
            cells=cells,
            discord=discord,
            device=device,
            patterns=patterns,
            scale=scale,
            upscale=upscale,
            line=line,
        )
    else:
        indexes = simple_indexes_torch(
            rgb,
            cells=cells,
            discord=discord,
            device=device,
            scale=scale,
            upscale=upscale,
        )
    return indexes_to_uint8_torch(indexes, mode=mode_name, patterns=patterns, device=device)


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
