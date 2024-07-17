import pathlib

import cellpose.denoise
import numpy as np
import scipy.ndimage as ndi
import skimage.exposure
import skimage.transform
import skimage.morphology
import tifffile
import tqdm

curr = pathlib.Path(
    r"Z:\yc296\computation\YC-20240429-cellpose_explore\FOR ALEX HUMAN ANNOTATIONS USE THESE2"
)

imgs_dna = sorted(curr.glob("*/*_DAPI.tif"))
imgs = [tifffile.imread(ii) for ii in imgs_dna]

dn_model = cellpose.denoise.CellposeDenoiseModel(
    # this seems to be the best atm
    gpu=True,
    model_type="cyto3",
    restore_type="deblur_cyto3",
)


def segment_tile(timg, diameter=15, normalize=True):
    timg = skimage.exposure.adjust_gamma(
        skimage.exposure.rescale_intensity(timg, out_range="float"),
        0.8,
    )
    tmask = dn_model.eval(
        timg,
        diameter=diameter,
        channels=[0, 0],
        normalize=normalize,
        flow_threshold=0,
        # GPU with 8 GB of RAM can handle 1024x1024 images
        tile=True,
    )[0]

    if np.all(tmask == 0):
        return tmask.astype("bool")

    struct_elem = ndi.generate_binary_structure(tmask.ndim, 1)
    contour = ndi.grey_dilation(tmask, footprint=struct_elem) != ndi.grey_erosion(
        tmask, footprint=struct_elem
    )
    return (tmask > 0) & ~contour


for pp, ii in zip(tqdm.tqdm(imgs_dna[:]), imgs):
    if ii.shape != (512, 512):
        ii = skimage.transform.resize(ii, (512, 512))
    mask = segment_tile(ii, 25, normalize=False)
    mask = skimage.morphology.label(mask)

    tifffile.imwrite(
        pp.parent / pp.name.replace("_DAPI.tif", "-mask-cellpose.tif"),
        mask,
        compression="zlib",
    )


# ---------------------------------------------------------------------------- #
#                               visual inspection                              #
# ---------------------------------------------------------------------------- #

import napari

v = napari.Viewer()

for pp, ii in zip(tqdm.tqdm(imgs_dna[:]), imgs):
    if ii.shape != (512, 512):
        ii = skimage.transform.resize(ii, (512, 512))
    mask = segment_tile(ii, 25, normalize=True)
    mask = skimage.morphology.label(mask)

    v.add_image(ii)
    v.add_labels(mask, name='normalized')
