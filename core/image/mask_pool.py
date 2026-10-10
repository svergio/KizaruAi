import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_POOL = None
_POOL_WORKERS = 0


def mask_worker_count(reserve=2):
    cores = os.cpu_count() or 2
    return max(1, int(cores) - int(reserve))


def _line_mask_job(payload):
    from core.image.ascii import line_mask

    gray, threshold, min_size, step, depth = payload
    return line_mask(
        gray,
        threshold=threshold,
        min_size=min_size,
        step=step,
        depth=depth,
    )


def _quadtree_form_job(payload):
    from core.image.ascii import quadtree_form

    gray, cols, rows, threshold = payload
    return quadtree_form(gray, cols, rows, threshold=threshold)


def ensure_mask_pool(workers=None):
    global _POOL, _POOL_WORKERS
    want = mask_worker_count() if workers is None else max(1, int(workers))
    if _POOL is not None and _POOL_WORKERS == want:
        return _POOL, _POOL_WORKERS
    close_mask_pool()
    _POOL = ProcessPoolExecutor(max_workers=want)
    _POOL_WORKERS = want
    return _POOL, _POOL_WORKERS


def close_mask_pool():
    global _POOL, _POOL_WORKERS
    if _POOL is not None:
        try:
            _POOL.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            _POOL.shutdown(wait=False)
    _POOL = None
    _POOL_WORKERS = 0


def line_masks_parallel(frames, opts, workers=None):
    # frames: list/ndarray (B,H,W) float32 on CPU
    array = np.asarray(frames, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    batch = int(array.shape[0])
    if batch == 1:
        from core.image.ascii import line_mask

        return np.stack(
            [
                line_mask(
                    array[0],
                    threshold=opts["threshold"],
                    min_size=opts["min_size"],
                    step=opts["step"],
                    depth=opts["depth"],
                )
            ],
            axis=0,
        )
    pool, _n = ensure_mask_pool(workers=workers)
    payloads = [
        (
            np.ascontiguousarray(array[index]),
            opts["threshold"],
            opts["min_size"],
            opts["step"],
            opts["depth"],
        )
        for index in range(batch)
    ]
    futures = [pool.submit(_line_mask_job, payload) for payload in payloads]
    return np.stack([future.result() for future in futures], axis=0)


def submit_line_masks(frames, opts, workers=None):
    array = np.asarray(frames, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    pool, n_workers = ensure_mask_pool(workers=workers)
    futures = [
        pool.submit(
            _line_mask_job,
            (
                np.ascontiguousarray(array[index]),
                opts["threshold"],
                opts["min_size"],
                opts["step"],
                opts["depth"],
            ),
        )
        for index in range(int(array.shape[0]))
    ]
    return futures, n_workers


def collect_line_masks(futures):
    return np.stack([future.result() for future in futures], axis=0)


def quadtree_forms_parallel(frames, cols, rows, threshold, workers=None):
    array = np.asarray(frames, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    batch = int(array.shape[0])
    if batch == 1:
        from core.image.ascii import quadtree_form

        return np.stack([quadtree_form(array[0], cols, rows, threshold=threshold)], axis=0)
    pool, _n = ensure_mask_pool(workers=workers)
    futures = [
        pool.submit(
            _quadtree_form_job,
            (np.ascontiguousarray(array[index]), cols, rows, threshold),
        )
        for index in range(batch)
    ]
    return np.stack([future.result() for future in futures], axis=0)
