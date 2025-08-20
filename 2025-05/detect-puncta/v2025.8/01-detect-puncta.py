import pathlib
import re

import dask.array as da
import dask.diagnostics
import numpy as np
import palom
import pandas as pd
import scipy.ndimage
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.restoration
import skimage.util
import tqdm.contrib
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


def process(
    img_path,
    out_dir,
    puncta_channels,
    dna_channel,
    puncta_sigma,
    save_RAM=False,
):
    reader = palom.reader.OmePyramidReader(img_path)
    path = reader.path

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # reader.pyramid[0] = reader.pyramid[0][:, :10000, :10000]

    if dna_channel is not None:
        img_dna = reader.pyramid[0][dna_channel]

    imgs = reader.pyramid[0][puncta_channels]

    imgs_log_peak_zarr_stores = [zarr.MemoryStore() for ii in range(len(imgs))]
    if save_RAM:
        imgs_log_peak_zarr_stores = [
            zarr.TempStore(dir=out_dir, prefix=f"log-{ii:02}-zarr-")
            for ii in range(len(imgs))
        ]
    masks_peak = zarr.zeros(imgs.shape, chunks=imgs.chunksize, dtype="bool")

    puncta_dfs = []
    for ii, (img, lzstore) in enumerate(zip(imgs, imgs_log_peak_zarr_stores)):
        img_log_peak = img.map_overlap(
            local_maxima_peak,
            sigma=puncta_sigma,
            depth=32,
            boundary="reflect",
            dtype="float32",
        )

        with dask.diagnostics.ProgressBar():
            palom.pyramid.da_to_zarr(img_log_peak, zarr_store=lzstore)

        coords = np.nonzero(zarr.Array(lzstore))
        values = zarr.Array(lzstore)[coords]

        threshold = skimage.filters.threshold_triangle(values)
        valid_peaks = values > threshold

        masks_peak.vindex[ii, coords[0][valid_peaks], coords[1][valid_peaks]] = True

        # -------------- save intensity properties for further filtering ------------- #
        df = pd.DataFrame()
        df[["Y", "X"]] = np.asarray(coords).T[valid_peaks].astype("int32")
        df["_LoG_intensity"] = zarr.Array(lzstore)[
            coords[0][valid_peaks], coords[1][valid_peaks]
        ]
        df["channel_intensity"] = img[
            da.from_zarr(masks_peak, name=False)[ii]
        ].compute()
        df["LoG_intensity"] = (
            skimage.exposure.rescale_intensity(
                df["_LoG_intensity"].values, out_range=str(img.dtype)
            )
            .round()
            .astype(img.dtype)
        )
        puncta_dfs.append(df)

    # ------------------------- write puncta mask to disk ------------------------ #
    stem = re.sub(r"\.ome\.tiff?$", "", path.name)
    max_digit_len = max([len(str(cc)) for cc in puncta_channels])
    channel_str = "_".join([f"{cc:0{max_digit_len}}" for cc in puncta_channels])
    out_path_mask = out_dir / f"{stem}-spots-channel_{channel_str}.ome.tif"

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

    for cc, dd in tqdm.contrib.tzip(
        puncta_channels, puncta_dfs, desc="Writing measurements"
    ):
        out_df_name = re.sub(
            r"\.ome\.tiff?$",
            f"-measurements_{cc:0{max_digit_len}}.parquet",
            out_path_mask.name,
        )
        out_df_path = out_dir / out_df_name
        out_df_path.parent.mkdir(exist_ok=True, parents=True)

        dd.to_parquet(out_df_path, engine="pyarrow")

    # ------------------------------ write qc image ------------------------------ #
    out_qc_name = re.sub(r"\.ome\.tiff?$", "-qc.ome.tif", out_path_mask.name)
    out_qc_path = out_dir / "qc" / out_qc_name
    out_qc_path.parent.mkdir(exist_ok=True, parents=True)

    mosaics_qc = []
    channel_names_qc = []

    if dna_channel is not None:
        mosaics_qc.append(img_dna)
        channel_names_qc.append(f"{dna_channel}_DNA")

    mosaics_qc.append(imgs)
    channel_names_qc.append([f"{cc}" for cc in puncta_channels])

    for mm in da.from_zarr(masks_peak, name=False):
        mask_dilated = mm.map_overlap(
            skimage.morphology.binary_dilation,
            footprint=skimage.morphology.disk(1),
            depth=4,
            boundary=False,
        )
        mosaics_qc.append(mask_dilated)
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


img_paths = r"""
T:\ORION_FISH_FT_fullpanel\LSP17773\registration\LSP17773.ome.tif
T:\ORION_FISH_FT_fullpanel\LSP33836_P110\registration\LSP33836_P110.ome.tif
""".strip().split("\n")

out_dirs = r"""
T:\ORION_FISH_FT_fullpanel\LSP17773\fish
T:\ORION_FISH_FT_fullpanel\LSP33836_P110\fish
""".strip().split("\n")

puncta_channels = [22, 24]
dna_channel = 20
puncta_sigma = 1.2


for pp, oo in zip(img_paths, out_dirs):
    process(pp, oo, puncta_channels, dna_channel, puncta_sigma)
