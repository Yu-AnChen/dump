import re

import dask.array as da
import dask.diagnostics
import numpy as np
import palom
import scipy.ndimage
import skimage.filters
import skimage.morphology
import skimage.restoration
import skimage.util
import zarr

# Pre-calculate the Laplacian operator kernel. We'll always be using 2D images.
_laplace_kernel = skimage.restoration.uft.laplacian(2, (3, 3))[1]


def whiten(img, sigma):
    img = skimage.util.img_as_float32(img)
    if sigma == 0:
        output = scipy.ndimage.convolve(img, _laplace_kernel)
    else:
        output = scipy.ndimage.gaussian_laplace(img, sigma)
    return output


def local_maxima_peak(img, sigma):
    img = -1.0 * whiten(img, sigma)
    mask = skimage.morphology.local_maxima(img, footprint=np.ones((3, 3)))
    labeled_mask = skimage.morphology.label(mask, connectivity=1)
    _, idx = np.unique(labeled_mask, return_index=True)
    del labeled_mask, mask
    out = np.zeros_like(img)
    out[np.unravel_index(idx, out.shape)] = img[np.unravel_index(idx, out.shape)]
    return out


def process(img_path, puncta_channels, dna_channel, puncta_sigma):
    reader = palom.reader.OmePyramidReader(img_path)
    path = reader.path

    # reader.pyramid[0] = reader.pyramid[0][:, :10000, :10000]

    if dna_channel is not None:
        img_dna = reader.pyramid[0][dna_channel]

    imgs = reader.pyramid[0][puncta_channels]
    imgs_peak_zarr_stores = [
        zarr.TempStore(dir=path.parent, prefix=f"{ii:02}-zarr-")
        for ii in range(len(imgs))
    ]
    masks_peak = zarr.zeros(imgs.shape, chunks=imgs.chunksize, dtype="bool")

    for ii, (img, zstore) in enumerate(zip(imgs, imgs_peak_zarr_stores)):
        img_peak = img.map_overlap(
            local_maxima_peak,
            sigma=puncta_sigma,
            depth=32,
            boundary="reflect",
            dtype="float32",
        )

        with dask.diagnostics.ProgressBar():
            palom.pyramid.da_to_zarr(img_peak, zarr_store=zstore)

        coords = np.nonzero(zarr.Array(zstore))
        values = zarr.Array(zstore)[coords]

        threshold = skimage.filters.threshold_otsu(values)
        peaks = values > threshold

        masks_peak.vindex[ii, coords[0][peaks], coords[1][peaks]] = True

    # ------------------------- write puncta mask to disk ------------------------ #
    stem = re.sub(r"\.ome\.tiff?$", "", path.name)
    channel_str = "_".join([str(cc) for cc in puncta_channels])
    out_path_mask = path.parent / f"{stem}-spots-channel_{channel_str}.ome.tif"

    palom.pyramid.write_pyramid(
        da.from_zarr(masks_peak, name=False).astype("uint8"),
        out_path_mask,
        pixel_size=reader.pixel_size,
        channel_names="",
        downscale_factor=2,
        tile_size=1024,
        save_RAM=True,
        compression="zlib",
        is_mask=True,
    )

    # ------------------------------ write qc image ------------------------------ #
    out_qc_name = re.sub(r"\.ome\.tiff?$", "-qc.ome.tif", out_path_mask.name)
    out_qc_path = path.parent / "qc" / out_qc_name
    out_qc_path.parent.mkdir(exist_ok=True, parents=True)

    mosaics_qc = []
    channel_names_qc = []

    if dna_channel is not None:
        mosaics_qc.append(img_dna)
        channel_names_qc.append(f"{dna_channel}_DNA")

    mosaics_qc.append(imgs)
    channel_names_qc.append([f"{cc}" for cc in puncta_channels])

    # mosaics_qc.extend(
    #     [da.from_zarr(ss, name=False) for ss in imgs_peak_zarr_stores]
    # )
    mosaics_qc.extend(
        [
            mm.map_overlap(
                skimage.morphology.binary_dilation,
                footprint=skimage.morphology.disk(1),
                depth=4,
                boundary=False,
            )
            for mm in da.from_zarr(masks_peak, name=False)
        ]
    )
    channel_names_qc.extend([f"{cc}_puncta" for cc in puncta_channels])

    palom.pyramid.write_pyramid(
        mosaics_qc,
        out_qc_path,
        pixel_size=reader.pixel_size,
        channel_names=channel_names_qc,
        downscale_factor=2,
        tile_size=1024,
        save_RAM=True,
        compression="zlib",
        is_mask=False,
    )


"""
img_path = r"W:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\DNA-FISH-test1\Orion_FISH_test1\Tanjina_ORION_FISH_test_HGSOC\LSP31050\registration\LSP31050_P110_A107_P54N_HMS_TK.ome.tif"
puncta_channels = [22, 24, 27, 28, 29]
dna_channel = 20
puncta_sigma = 2
"""