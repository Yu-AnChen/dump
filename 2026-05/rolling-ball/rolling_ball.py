"""
rolling_ball_large.py

Port of imagec's rolling-ball background subtraction for large 2-D images.

Strategy (mirrors imagec's shrink → roll → enlarge approach):
  1. Min-pool image by shrinkFactor (sf) — preserves background minima
  2. Apply scikit-image rolling ball at radius/sf on the small image
  3. Bilinear upsample the estimated background back to full size
  4. Subtract from original, clip to zero

shrinkFactor thresholds (matching imagec):
  radius ≤  10  →  sf = 1  (no shrink)
  radius ≤  30  →  sf = 2
  radius ≤ 100  →  sf = 4
  radius > 100  →  sf = 8

Large-image handling:
  - Dask splits the image into chunks; each chunk is padded with `radius`
    pixels of overlap so the ball has full context at block boundaries.
  - Zarr provides chunked on-disk or in-memory storage.
  - Thread scheduler parallelises blocks; cv2, numpy, and the skimage Cython
    core all release the GIL, so multiple blocks run concurrently.

Thread allocation (empirically benchmarked on 4096×4096, chunk=1024):
  sf == 1 (radius ≤ 10): padded blocks are full-size (~1.1M pixels after no
    shrink), so skimage's OpenMP kernel has enough pixels to parallelise over.
    Best: split cores evenly — dask_workers = n//2, omp_threads = 2.
  sf >= 2 (radius > 10): after min-pool the small image is tiny (~280×280 for
    sf=4). OpenMP thread overhead dominates over useful work per block.
    Best: all cores to dask — dask_workers = n, omp_threads = 1.
  Using omp_threads=auto (default in skimage) with dask_workers=4 causes
  40 threads on 10 cores for sf=1, which is 2× slower than the split above.
"""

from __future__ import annotations

import math
import os
from functools import partial

import cv2
import numpy as np
import zarr
import dask.array as da
from skimage.restoration import rolling_ball as _skimage_rb


# ── shrink helpers ───────────────────────────────────────────────────────── #

def _shrink_params(radius: float) -> tuple[int, float]:
    """Return (shrinkFactor, arcTrimPct) matching imagec's radius thresholds."""
    if radius <= 10:
        return 1, 0.24
    elif radius <= 30:
        return 2, 0.24
    elif radius <= 100:
        return 4, 0.32
    else:
        return 8, 0.40


def _min_pool_2d(img: np.ndarray, sf: int) -> np.ndarray:
    """
    Non-overlapping min-pool by factor sf via numpy reshape — no Python loop.

    Clips to the largest H/W divisible by sf before pooling; the clipped
    border (< sf pixels) is negligible for background estimation.
    """
    if sf == 1:
        return img
    h, w = img.shape
    h_c, w_c = (h // sf) * sf, (w // sf) * sf
    return img[:h_c, :w_c].reshape(h // sf, sf, w // sf, sf).min(axis=(1, 3))


# ── per-block core ───────────────────────────────────────────────────────── #

def _rolling_ball_block(
    block: np.ndarray, *, radius: float, omp_threads: int
) -> np.ndarray:
    """
    Estimate rolling-ball background for one overlap-padded block.

    dask.map_overlap calls this with the block already padded by `radius`
    pixels on every side.  The function must return an array of the same
    shape; dask trims the overlap border from the output automatically.

    GIL notes — all inner operations release the GIL:
      cv2.blur, cv2.resize   : C extension, no GIL
      numpy reshape + min    : C loop, no GIL
      skimage apply_kernel   : Cython + OpenMP, no GIL
    """
    sf, _ = _shrink_params(radius)
    fp = block.astype(np.float32)

    # Step 1: pre-smooth with 3×3 mean (matches imagec doPresmooth)
    fp = cv2.blur(fp, (3, 3))

    # Step 2: min-pool → shrunk image
    small = _min_pool_2d(fp, sf)

    # Step 3: rolling ball at reduced scale
    small_bg = _skimage_rb(small, radius=max(radius / sf, 1.0),
                           num_threads=omp_threads)

    # Step 4: bilinear upsample back to block shape
    bg = cv2.resize(
        np.asarray(small_bg, dtype=np.float32),
        (fp.shape[1], fp.shape[0]),   # cv2 takes (width, height)
        interpolation=cv2.INTER_LINEAR,
    )
    return bg


def _thread_split(radius: float, n_cores: int) -> tuple[int, int]:
    """
    Return (dask_workers, omp_threads) that sum to n_cores without
    over-subscribing, based on shrink regime.

    sf == 1: blocks are full-size after no shrink — OpenMP has enough pixels
             to parallelise over, so split cores evenly between the two levels.
    sf >= 2: blocks shrink to a small image — OpenMP overhead dominates, so
             give all cores to dask and disable OpenMP inside each block.
    """
    sf, _ = _shrink_params(radius)
    if sf == 1:
        dask_workers = max(1, n_cores // 2)
        omp_threads  = max(1, n_cores // dask_workers)
    else:
        dask_workers = n_cores
        omp_threads  = 1
    return dask_workers, omp_threads


# ── public API ───────────────────────────────────────────────────────────── #

def rolling_ball_background(
    src: zarr.Array | np.ndarray,
    radius: float = 50.0,
    chunk_size: int = 2048,
    omp_threads: int = 1,
) -> da.Array:
    """
    Return a lazy dask array of the estimated rolling-ball background.

    The result is NOT yet computed — pass it to subtract_background() or
    call .compute() directly for images that fit in RAM.

    Parameters
    ----------
    src         : 2-D zarr.Array or numpy ndarray (uint16 or float32)
    radius      : rolling ball radius in full-image pixels
    chunk_size  : chunk edge in pixels; must be > 2 × radius
    omp_threads : OpenMP threads passed to skimage per block; use
                  _thread_split() to get the right value for your core count
    """
    overlap = math.ceil(radius)
    if chunk_size <= overlap * 2:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be > 2×radius ({2 * radius:.0f})"
        )

    x = (
        da.from_zarr(src, chunks=(chunk_size, chunk_size))
        if isinstance(src, zarr.Array)
        else da.from_array(src, chunks=(chunk_size, chunk_size))
    )

    return da.map_overlap(
        partial(_rolling_ball_block, radius=radius, omp_threads=omp_threads),
        x,
        depth=overlap,
        boundary="reflect",   # reflect at image edges instead of zero-padding
        dtype=np.float32,
    )


def subtract_background(
    src: zarr.Array | np.ndarray,
    radius: float = 50.0,
    chunk_size: int = 2048,
    num_workers: int | None = None,
    out_path: str | None = None,
) -> zarr.Array:
    """
    Subtract rolling-ball background from a large 2-D image, write to Zarr.

    Parameters
    ----------
    src         : 2-D zarr.Array or numpy ndarray (uint16 or float32)
    radius      : rolling ball radius in full-image pixels
    chunk_size  : chunk edge in pixels (must be > 2 × radius)
    num_workers : total CPU threads to use; None → os.cpu_count().
                  Automatically split between dask blocks and OpenMP based
                  on shrink regime (see _thread_split).
    out_path    : path for the output zarr array; None → in-memory

    Returns
    -------
    zarr.Array (float32) with background subtracted, clipped to ≥ 0
    """
    n_cores = num_workers or os.cpu_count() or 1
    dask_workers, omp_threads = _thread_split(radius, n_cores)

    x = (
        da.from_zarr(src, chunks=(chunk_size, chunk_size))
        if isinstance(src, zarr.Array)
        else da.from_array(src, chunks=(chunk_size, chunk_size))
    )

    bg = rolling_ball_background(src, radius=radius, chunk_size=chunk_size,
                                 omp_threads=omp_threads)
    result = (x.astype(np.float32) - bg).clip(min=0)

    out_arr = (
        zarr.open_array(out_path, mode="w", shape=x.shape,
                        chunks=(chunk_size, chunk_size), dtype=np.float32)
        if out_path
        else zarr.zeros(x.shape, chunks=(chunk_size, chunk_size), dtype=np.float32)
    )

    da.store(result, out_arr, scheduler="threads", num_workers=dask_workers)
    return out_arr


# ── smoke test ───────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import time

    rng = np.random.default_rng(42)
    H, W = 4096, 4096

    # Simulated 16-bit fluorescence: Poisson-noise spots + sinusoidal background
    noise = rng.poisson(300, size=(H, W)).astype(np.uint16)
    yy, xx = np.mgrid[0:H, 0:W]
    bg_true = (3000 * np.sin(yy / H * np.pi) * np.sin(xx / W * np.pi)).astype(np.int32)
    image = np.clip(noise.astype(np.int32) + bg_true, 0, 65535).astype(np.uint16)

    n_cores = os.cpu_count()
    print(f"Input: {image.shape}  dtype={image.dtype}  "
          f"range=[{image.min()}, {image.max()}]  cores={n_cores}")
    print()

    for radius in (10, 50, 150):
        sf, _ = _shrink_params(radius)
        dask_w, omp_t = _thread_split(radius, n_cores)
        t0 = time.perf_counter()
        out = subtract_background(image, radius=radius, chunk_size=1024)
        dt = time.perf_counter() - t0
        arr = out[:]
        print(f"radius={radius:4d}  sf={sf}  dask={dask_w} omp={omp_t}  "
              f"time={dt:.2f}s  out range=[{arr.min():.0f}, {arr.max():.0f}]")
