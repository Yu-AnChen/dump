"""
Benchmark: dask thread count vs OpenMP thread count for rolling ball POC.

Tests all (dask_workers, omp_threads) combinations across sf=1, sf=4, sf=8
to find the best split for each shrink regime.
"""

from __future__ import annotations

import math
import os
import time
from functools import partial
from itertools import product

import cv2
import numpy as np
import zarr
import dask.array as da
from skimage.restoration import rolling_ball as _skimage_rb

# ── copy of core helpers from rolling_ball_large.py ─────────────────────── #

def _shrink_params(radius):
    if radius <= 10:    return 1, 0.24
    elif radius <= 30:  return 2, 0.24
    elif radius <= 100: return 4, 0.32
    else:               return 8, 0.40


def _min_pool_2d(img, sf):
    if sf == 1:
        return img
    h, w = img.shape
    h_c, w_c = (h // sf) * sf, (w // sf) * sf
    return img[:h_c, :w_c].reshape(h // sf, sf, w // sf, sf).min(axis=(1, 3))


def _rolling_ball_block(block, *, radius, omp_threads):
    sf, _ = _shrink_params(radius)
    fp = block.astype(np.float32)
    fp = cv2.blur(fp, (3, 3))
    small = _min_pool_2d(fp, sf)
    small_bg = _skimage_rb(small, radius=max(radius / sf, 1.0),
                           num_threads=omp_threads)
    bg = cv2.resize(
        np.asarray(small_bg, dtype=np.float32),
        (fp.shape[1], fp.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return bg


def run(image, radius, chunk_size, dask_workers, omp_threads):
    x = da.from_array(image, chunks=(chunk_size, chunk_size))
    overlap = math.ceil(radius)
    bg = da.map_overlap(
        partial(_rolling_ball_block, radius=radius, omp_threads=omp_threads),
        x,
        depth=overlap,
        boundary="reflect",
        dtype=np.float32,
    )
    result = (x.astype(np.float32) - bg).clip(min=0)
    out = zarr.zeros(image.shape, chunks=(chunk_size, chunk_size), dtype=np.float32)
    t0 = time.perf_counter()
    da.store(result, out, scheduler="threads", num_workers=dask_workers)
    return time.perf_counter() - t0


# ── benchmark ────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    N_CORES = os.cpu_count()
    H, W = 4096, 4096
    CHUNK  = 1024
    WARMUP = 1   # runs before timing
    REPS   = 3   # timed runs per config

    rng = np.random.default_rng(0)
    noise = rng.poisson(300, (H, W)).astype(np.uint16)
    yy, xx = np.mgrid[0:H, 0:W]
    bg = (3000 * np.sin(yy / H * np.pi) * np.sin(xx / W * np.pi)).astype(np.int32)
    image = np.clip(noise.astype(np.int32) + bg, 0, 65535).astype(np.uint16)

    print(f"Image {H}×{W}  chunk={CHUNK}  CPU cores={N_CORES}\n")

    # Configurations: (dask_workers, omp_threads)
    # Include combinations where product exceeds core count to show over-subscription
    combos = sorted(set([
        (1, 1),
        (1, N_CORES),
        (2, N_CORES // 2),
        (4, 1),
        (4, 2),
        (4, N_CORES // 4),
        (N_CORES, 1),
        (N_CORES, 2),       # intentional over-subscription
        (N_CORES // 2, 2),
    ]))

    # One radius per sf regime
    test_cases = [
        ("sf=1  (r=10) ", 10),
        ("sf=4  (r=50) ", 50),
        ("sf=8  (r=150)", 150),
    ]

    header = f"{'config (dask×omp)':>22} | total threads | " + " | ".join(
        f"{label:^14}" for label, _ in test_cases
    )
    print(header)
    print("-" * len(header))

    for dask_w, omp_t in combos:
        label = f"dask={dask_w} × omp={omp_t}"
        total = dask_w * omp_t
        row = f"{label:>22} | {total:^13} |"
        for case_label, radius in test_cases:
            # warmup
            for _ in range(WARMUP):
                run(image, radius, CHUNK, dask_w, omp_t)
            # timed
            times = [run(image, radius, CHUNK, dask_w, omp_t) for _ in range(REPS)]
            best = min(times)
            row += f" {best:>6.2f}s        |"
        print(row)

    print()
    print("(showing best of", REPS, "runs per config)")
