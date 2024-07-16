import napari
import numpy as np
import skimage.exposure
import skimage.transform
import tifffile
import torch

from STalign import STalign

v = napari.Viewer()


case_number = [1, 2, 3, 4, 6, 7, 8, 9, 10]
# ref = tifffile.imread("/Users/yuanchen/projects/STalign/docs/img_data/ref-c10.tif")
# moving = tifffile.imread("/Users/yuanchen/projects/STalign/docs/img_data/moving-c10.tif")


for nn in case_number[1:2]:
    print("processing", nn)

    ref = tifffile.imread(
        f"/Users/yuanchen/projects/STalign/docs/img_data/test-elastix-img-pair/C{nn:02}-ref.tif"
    )
    moving = tifffile.imread(
        f"/Users/yuanchen/projects/STalign/docs/img_data/test-elastix-img-pair/C{nn:02}-moving.tif"
    )

    # matched_moving = skimage.exposure.match_histograms(moving, ref)
    matched_moving = moving

    # device = "cuda:0"
    device = "cpu"
    ooo = STalign.LDDMM(
        xI=[np.arange(ss, dtype=float) for ss in matched_moving.shape],
        # I=skimage.exposure.rescale_intensity(
        #     np.log1p(np.clip(matched_moving, 500, None)), out_range="float"
        # )[np.newaxis],
        I=skimage.util.img_as_float32(matched_moving)[np.newaxis],
        xJ=[np.arange(ss, dtype=float) for ss in ref.shape],
        # J=skimage.exposure.rescale_intensity(
        #     np.log1p(np.clip(ref, 500, None)), out_range="float"
        # )[np.newaxis],
        J=skimage.util.img_as_float32(ref)[np.newaxis],
        A=torch.eye(3, device=device),
        muB=torch.tensor([0], device=device),
        sigmaM=0.4,
        sigmaB=1e-3,
        device=device,
        niter=100,
        a=100,
        epV=1_000,
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

    v.add_image(ref, colormap="bop blue", visible=False)
    v.add_image(
        matched_moving, blending="additive", colormap="bop purple", visible=False
    )
    v.add_image(
        phiI[0].numpy(), blending="additive", colormap="bop orange", visible=False
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
