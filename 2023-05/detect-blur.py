import functools

import cv2
import numpy as np
import skimage.measure
import skimage.util
import tifffile
from joblib import Parallel, delayed


def shannon_entropy(img, block_size):
    assert img.ndim == 2
    wimg = skimage.util.view_as_windows(img, block_size, block_size)
    h, w, *_ = wimg.shape
    out = np.zeros((h, w))
    out.flat = Parallel(n_jobs=4)(
        delayed(skimage.measure.shannon_entropy)(wimg[i, j])
        for i, j in np.mgrid[:h, :w].reshape(2, -1).T
    )
    return out


def var_of_laplacian(img, block_size, sigma=0):
    assert img.ndim == 2
    wimg = skimage.util.view_as_windows(img, block_size, block_size)
    h, w, *_ = wimg.shape
    out = np.zeros((h, w))

    func = lambda x: x
    if sigma != 0:
        func = functools.partial(cv2.GaussianBlur, ksize=(0, 0), sigmaX=sigma)
    for i, j in np.mgrid[:h, :w].reshape(2, -1).T:
        out[i, j] = np.var(
            cv2.Laplacian(func(wimg[i, j]), cv2.CV_32F, ksize=1)
        )
    return out


def var_block(img, block_size):
    assert img.ndim == 2
    wimg = skimage.util.view_as_windows(img, block_size, block_size)
    return np.var(wimg, axis=(2, 3))


def mean_block(img, block_size):
    assert img.ndim == 2
    wimg = skimage.util.view_as_windows(img, block_size, block_size)
    return np.mean(wimg, axis=(2, 3))



import logging
import pathlib

import skimage.filters

logging.basicConfig( 
    format="%(asctime)s | %(levelname)-8s | %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S", 
    level=logging.INFO 
)


def process_file(img_path):
    img_path = pathlib.Path(img_path)
    
    logging.info(f"Reading {img_path}")
    img1 = tifffile.imread(img_path, key=0)

    logging.info("Computing entropy")
    entropy_img = shannon_entropy(img1, 128)
    tissue_mask = entropy_img > skimage.filters.threshold_triangle(entropy_img)

    logging.info("Computing variance")
    lvar_img = var_of_laplacian(img1, 128, 1)
    # qc_img = lvar_img / (mean_block(img1, 128)+1)
    # qc_img = np.nan_to_num(qc_img)
    # qc_mask = qc_img > skimage.filters.threshold_triangle(qc_img)
    qc_mask = lvar_img > skimage.filters.threshold_triangle(lvar_img)
    iou = (tissue_mask & qc_mask).sum() / (tissue_mask | qc_mask).sum()

    logging.info(
        f"IoU {iou*100:.0f}%"
    )

process_file(r"X:\cycif-production\149-Orion-Awad_Batch2\LSP16096_P54_A31_C100_HMS_Orion7@20230403_194642_809947.ome.tiff")
    
