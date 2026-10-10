import torch
import torch.nn.functional as F

from core.image.ascii import _square_covers, line_options


def _integral(field):
    # field (B,H,W) -> sat (B,H+1,W+1); sat[:,y,x] = sum of field[0:y, 0:x]
    return F.pad(field, (1, 0, 1, 0)).cumsum(dim=-1).cumsum(dim=-2)


def _rect_sum(sat, y0, x0, y1, x1):
    return sat[:, y1, x1] - sat[:, y0, x1] - sat[:, y1, x0] + sat[:, y0, x0]


def _block_stats(sat, sat2, y, x, h, w):
    y0 = y
    x0 = x
    y1 = y + h
    x1 = x + w
    n = (h * w).to(sat.dtype).clamp(min=1)
    total = _rect_sum(sat, y0, x0, y1, x1)
    total2 = _rect_sum(sat2, y0, x0, y1, x1)
    mean255 = total / n
    mse = (total2 / n - mean255 * mean255).clamp(min=0.0)
    empty = (h < 1) | (w < 1)
    mean01 = mean255 / 255.0
    mse = torch.where(empty, torch.zeros_like(mse), mse)
    return mean01, mse, empty


def _scatter_vlines(mask, paint, y0, y1, x):
    if not bool(paint.any()):
        return
    b, i, j = paint.nonzero(as_tuple=True)
    height, width = mask.shape[-2], mask.shape[-1]
    rows0 = y0[i, j].clamp(0, height)
    rows1 = y1[i, j].clamp(0, height)
    cols = x[i, j].clamp(0, max(0, width - 1))
    leng = (rows1 - rows0).clamp(min=0)
    total = int(leng.sum().item())
    if total <= 0:
        return
    ends = leng.cumsum(0)
    pos = torch.arange(total, device=mask.device)
    seg = torch.searchsorted(ends, pos, right=True)
    starts = ends - leng
    inner = pos - starts[seg]
    mask[b[seg], rows0[seg] + inner, cols[seg]] = 1.0


def _scatter_hlines(mask, paint, x0, x1, y):
    if not bool(paint.any()):
        return
    b, i, j = paint.nonzero(as_tuple=True)
    height, width = mask.shape[-2], mask.shape[-1]
    cols0 = x0[i, j].clamp(0, width)
    cols1 = x1[i, j].clamp(0, width)
    rows = y[i, j].clamp(0, max(0, height - 1))
    leng = (cols1 - cols0).clamp(min=0)
    total = int(leng.sum().item())
    if total <= 0:
        return
    ends = leng.cumsum(0)
    pos = torch.arange(total, device=mask.device)
    seg = torch.searchsorted(ends, pos, right=True)
    starts = ends - leng
    inner = pos - starts[seg]
    mask[b[seg], rows[seg], cols0[seg] + inner] = 1.0


def _split_geom(y, x, h, w):
    half_h = torch.div(h, 2, rounding_mode="floor")
    half_w = torch.div(w, 2, rounding_mode="floor")
    quads = (
        (y, x, half_h, half_w),
        (y, x + half_w, half_h, w - half_w),
        (y + half_h, x, h - half_h, half_w),
        (y + half_h, x + half_w, h - half_h, w - half_w),
    )
    return half_h, half_w, quads


def _expand_geom(y, x, h, w):
    half_h, half_w, quads = _split_geom(y, x, h, w)
    grid = y.shape[-1] * 2
    device = y.device
    y2 = torch.empty((grid, grid), dtype=y.dtype, device=device)
    x2 = torch.empty((grid, grid), dtype=x.dtype, device=device)
    h2 = torch.empty((grid, grid), dtype=h.dtype, device=device)
    w2 = torch.empty((grid, grid), dtype=w.dtype, device=device)
    slots = ((0, 0), (0, 1), (1, 0), (1, 1))
    for (qy, qx, qh, qw), (dy, dx) in zip(quads, slots):
        y2[dy::2, dx::2] = qy
        x2[dy::2, dx::2] = qx
        h2[dy::2, dx::2] = qh
        w2[dy::2, dx::2] = qw
    return y2, x2, h2, w2


def _expand_entered(entered, child_continue):
    # entered (B,G,G), child_continue (B,G,G,4) TL TR BL BR
    batch, grid, _g = entered.shape
    out = torch.zeros(
        (batch, grid * 2, grid * 2), dtype=torch.bool, device=entered.device
    )
    out[:, 0::2, 0::2] = entered & child_continue[..., 0]
    out[:, 0::2, 1::2] = entered & child_continue[..., 1]
    out[:, 1::2, 0::2] = entered & child_continue[..., 2]
    out[:, 1::2, 1::2] = entered & child_continue[..., 3]
    return out


def _cover_mask(sat, sat2, oy, ox, side, opts, height, width):
    device = sat.device
    batch = sat.shape[0]
    threshold = float(opts["threshold"])
    min_size = int(opts["min_size"])
    max_depth = int(opts["depth"])
    seam = float(opts["step"]) / 255.0
    mask = sat.new_zeros((batch, height, width), dtype=torch.float32)
    y = torch.tensor([[oy]], dtype=torch.long, device=device)
    x = torch.tensor([[ox]], dtype=torch.long, device=device)
    h = torch.tensor([[side]], dtype=torch.long, device=device)
    w = torch.tensor([[side]], dtype=torch.long, device=device)
    entered = torch.ones((batch, 1, 1), dtype=torch.bool, device=device)
    for depth in range(max_depth):
        if not bool(entered.any()):
            break
        half_h, half_w, quads = _split_geom(y, x, h, w)
        _mean, mse, _empty = _block_stats(sat, sat2, y, x, h, w)
        split = (
            entered
            & (mse > threshold)
            & (h.unsqueeze(0) > min_size)
            & (w.unsqueeze(0) > min_size)
            & (half_h.unsqueeze(0) >= 1)
            & (half_w.unsqueeze(0) >= 1)
        )
        means = []
        leaves = []
        empties = []
        for qy, qx, qh, qw in quads:
            mean, child_mse, empty = _block_stats(sat, sat2, qy, qx, qh, qw)
            leaf = (
                empty
                | (child_mse <= threshold)
                | ((depth + 1) >= max_depth)
                | (qh.unsqueeze(0) <= min_size)
                | (qw.unsqueeze(0) <= min_size)
            )
            means.append(mean)
            leaves.append(leaf)
            empties.append(empty)

        def far(left, right, empty_left, empty_right):
            return (~empty_left) & (~empty_right) & ((left - right).abs() >= seam)

        paint = split
        _scatter_vlines(
            mask,
            paint & leaves[0] & leaves[1] & far(means[0], means[1], empties[0], empties[1]),
            y,
            y + half_h,
            x + half_w,
        )
        _scatter_vlines(
            mask,
            paint & leaves[2] & leaves[3] & far(means[2], means[3], empties[2], empties[3]),
            y + half_h,
            y + h,
            x + half_w,
        )
        _scatter_hlines(
            mask,
            paint & leaves[0] & leaves[2] & far(means[0], means[2], empties[0], empties[2]),
            x,
            x + half_w,
            y + half_h,
        )
        _scatter_hlines(
            mask,
            paint & leaves[1] & leaves[3] & far(means[1], means[3], empties[1], empties[3]),
            x + half_w,
            x + w,
            y + half_h,
        )
        child_continue = torch.stack(
            [~leaves[0], ~leaves[1], ~leaves[2], ~leaves[3]], dim=-1
        )
        entered = _expand_entered(split, child_continue)
        y, x, h, w = _expand_geom(y, x, h, w)
    return mask


def line_masks_torch(gray, opts=None):
    # gray: (B,H,W) or (H,W) in [0,1] on CUDA/CPU. Same seams as ascii.line_mask.
    if opts is None:
        opts = line_options()
    single = gray.ndim == 2
    if single:
        gray = gray.unsqueeze(0)
    height, width = int(gray.shape[-2]), int(gray.shape[-1])
    values = gray.to(dtype=torch.float64) * 255.0
    sat = _integral(values)
    sat2 = _integral(values * values)


    out = gray.new_zeros((gray.shape[0], height, width), dtype=torch.float32)
    for oy, ox, side in _square_covers(height, width):
        out = torch.maximum(
            out, _cover_mask(sat, sat2, oy, ox, side, opts, height, width)
        )
    return out[0] if single else out
