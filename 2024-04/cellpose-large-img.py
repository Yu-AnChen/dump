# ---------------------------------------------------------------------------- #
#                              Process large image                             #
# ---------------------------------------------------------------------------- #
import datetime
import pathlib
import time

import cellpose.denoise
import dask.array as da
import dask.diagnostics
import dask_image.ndmeasure
import palom
import scipy.ndimage as ndi
import skimage.segmentation
import zarr
import numpy as np


def segment_tile(timg):
    timg = skimage.exposure.adjust_gamma(
        skimage.exposure.rescale_intensity(
            timg, in_range=(500, 40_000), out_range="float"
        ),
        0.7,
    )
    # return dn_model.eval(
    #     timg,
    #     diameter=15.0,
    #     channels=[0, 0],
    #     # inputs are globally normalized already
    #     normalize=False,
    #     flow_threshold=0,
    #     # GPU with 8 GB of RAM can handle 1024x1024 images
    #     tile=True,
    # )
    tmask = dn_model.eval(
        timg,
        diameter=15.0,
        channels=[0, 0],
        # inputs are globally normalized already
        normalize=False,
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


def da_to_zarr(da_img, zarr_store=None, num_workers=None, out_shape=None, chunks=None):
    if zarr_store is None:
        if out_shape is None:
            out_shape = da_img.shape
        if chunks is None:
            chunks = da_img.chunksize
        zarr_store = zarr.create(
            out_shape, chunks=chunks, dtype=da_img.dtype, overwrite=True
        )
    with dask.diagnostics.ProgressBar():
        da_img.to_zarr(zarr_store, compute=False).compute(num_workers=num_workers)
    return zarr_store


dn_model = cellpose.denoise.CellposeDenoiseModel(
    # this seems to be the best atm
    gpu=True,
    model_type="cyto3",
    restore_type="deblur_cyto3",
)

start = int(time.perf_counter())

img_path = r"Z:\RareCyte-S3\P37_CRCstudy_Round1\P37_S32_A24_C59kX_E15@20220106_014630_553652-zlib.ome.tiff"
out_dir = pathlib.Path(r"Z:\yc296\computation\YC-20240429-cellpose_explore")
out_dir.mkdir(exist_ok=True, parents=True)

reader = palom.reader.OmePyramidReader(img_path)

# img = reader.pyramid[0][0][:1024*5, 1024*40:1024*45]
img = reader.pyramid[0][0]

_binary_mask = img.map_overlap(
    segment_tile, depth={0: 128, 1: 128}, boundary="none", dtype=bool
)

_binary_mask = da_to_zarr(_binary_mask, num_workers=2)
binary_mask = da.from_zarr(_binary_mask)

end = int(time.perf_counter())
print("\nelapsed (cellpose):", datetime.timedelta(seconds=end - start))

_labeled_mask = dask_image.ndmeasure.label(binary_mask)[0]
labeled_mask = da_to_zarr(_labeled_mask)

end = int(time.perf_counter())
print("\nelapsed (label):", datetime.timedelta(seconds=end - start))

palom.pyramid.write_pyramid(
    [da.from_zarr(labeled_mask)],
    out_dir / f"{reader.path.name.split('.')[0]}-cellpose-mask-f0.ome.tif",
    pixel_size=reader.pixel_size,
    downscale_factor=2,
    compression="zlib",
    save_RAM=True,
    is_mask=True,
)

end = int(time.perf_counter())
print("\nelapsed (total):", datetime.timedelta(seconds=end - start))


# ---------------------------------------------------------------------------- #
#                                  dev section                                 #
# ---------------------------------------------------------------------------- #
limg = reader.pyramid[0][0, :5000, 40000 : 40000 + 5000].compute()
limg = da.from_array(limg, chunks=1024)

bimg = limg.map_overlap(
    segment_tile, depth={0: 128, 1: 128}, boundary="none", dtype=bool
)


with dask.diagnostics.ProgressBar():
    bimg = bimg.compute(num_workers=2)
    # bimg = bimg.compute()
    # bimg = bimg.compute(num_workers=4, scheduler="processes")
