from STalign import STalign
import torch
import tifffile
import skimage.exposure
import numpy as np

ref = tifffile.imread("../img_data/ref-c10.tif")
moving = tifffile.imread("../img_data/moving-c10.tif")

matched_moving = skimage.exposure.match_histograms(moving, ref)


device = "cuda:0"
device = "cpu"
ooo = STalign.LDDMM(
    xI=[np.arange(ss, dtype=float) for ss in matched_moving.shape],
    I=skimage.exposure.rescale_intensity(
        np.log1p(np.clip(matched_moving, 500, None)), out_range="float"
    )[np.newaxis],
    xJ=[np.arange(ss, dtype=float) for ss in ref.shape],
    J=skimage.exposure.rescale_intensity(
        np.log1p(np.clip(ref, 500, None)), out_range="float"
    )[np.newaxis],
    A=torch.eye(3, device=device),
    muB=torch.tensor([0], device=device),
    sigmaM=0.6,
    sigmaB=1e-7,
    device=device,
    niter=100,
    a=10,
    epV=10,
)

phiI = STalign.transform_image_source_to_target(
    ooo["xv"],
    ooo["v"],
    ooo["A"],
    [np.arange(ss, dtype=float) for ss in matched_moving.shape],
    skimage.exposure.rescale_intensity(
        np.log1p(np.clip(matched_moving, 500, None)), out_range="float"
    )[np.newaxis],
    [np.arange(ss, dtype=float) for ss in ref.shape],
)


"""best atm
ooo = STalign.LDDMM(
    xI=[np.arange(ss, dtype=float) for ss in matched_moving.shape],
    I=skimage.exposure.rescale_intensity(
        np.log1p(np.clip(matched_moving, 500, None)), out_range="float"
    )[np.newaxis],
    xJ=[np.arange(ss, dtype=float) for ss in ref.shape],
    J=skimage.exposure.rescale_intensity(
        np.log1p(np.clip(ref, 500, None)), out_range="float"
    )[np.newaxis],
    A=torch.eye(3, device=device),
    device=device,
    muB=torch.tensor([0], device=device),
    sigmaM=0.6,
    sigmaB=1e-7,
    niter=100,
    a=10,
    epV=10,
)
"""
