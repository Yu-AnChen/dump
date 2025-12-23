# ---------------------------------------------------------------------------- #
#                              Process large image                             #
# ---------------------------------------------------------------------------- #
import datetime
import inspect
import pathlib
import re
import time

import cellpose.models
import dask.array as da
import dask.diagnostics
import dask_image.ndmeasure
import numpy as np
import palom
import scipy.ndimage as ndi
import skimage.segmentation
import torch
import zarr


dn_model = cellpose.models.CellposeModel(gpu=True, pretrained_model="cpsam")


def segment_tile(timg, diameter, flow_threshold, **kwargs):
    # Get valid argument names for dn_model.eval
    valid_args = inspect.signature(dn_model.eval).parameters.keys()
    # Build the argument dict for eval
    eval_kwargs = {
        "x": timg,
        "diameter": diameter,
        "flow_threshold": flow_threshold,
        "normalize": False,
    }
    # Add only valid kwargs
    for k, v in kwargs.items():
        if k in valid_args:
            eval_kwargs[k] = v
        else:
            print(
                f"Warning: '{k}' is not a valid argument for model.eval and will be ignored."
            )

    with torch.no_grad():
        tmask = dn_model.eval(**eval_kwargs)[0]

    if np.all(tmask == 0):
        return tmask.astype("bool")

    struct_elem = ndi.generate_binary_structure(tmask.ndim, 1)
    contour = ndi.grey_dilation(tmask, footprint=struct_elem) != ndi.grey_erosion(
        tmask, footprint=struct_elem
    )
    return (tmask > 0) & ~contour


def adjust_intensity(img, intensity_in_range, intensity_gamma):
    img = skimage.exposure.rescale_intensity(
        img, in_range=intensity_in_range, out_range="float"
    )
    if intensity_gamma != 1.0:
        img = skimage.exposure.adjust_gamma(img, intensity_gamma)
    return img.astype("float32")


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


def segment_slide(
    img_path,
    channel,
    out_dir,
    intensity_p0=0,
    intensity_p1=100,
    intensity_gamma=1.0,
    diameter=15.0,
    flow_threshold=0.4,
    **kwargs,
):
    """
    Process a slide image with Cellpose denoising and segmentation.

    Parameters:
        img_path (str): Path to the input image.
        out_dir (str): Directory to save the output mask.
        intensity_p0 (int): Lower bound for intensity rescaling.
        intensity_p1 (int): Upper bound for intensity rescaling.
        intensity_gamma (float): Gamma correction factor for intensity adjustment.
        diameter (float): Diameter for Cellpose segmentation.
        flow_threshold (float): Flow threshold for Cellpose segmentation.
    """
    start = int(time.perf_counter())

    reader = palom.reader.OmePyramidReader(img_path)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    _img = reader.pyramid[0][channel]
    assert np.ndim(_img) in [2, 3]

    if np.ndim(_img) == 2:
        _img = _img[np.newaxis]
        channel = [channel]

    channels = []
    # cellpose SAM takes <=3 channels
    for cimg, ii, _ in zip(_img, channel, range(3)):
        with dask.diagnostics.ProgressBar():
            ss = np.ceil(np.prod(cimg.shape) / (10_000**2)).astype("int")
            p0, p1 = np.percentile(
                cimg[::ss, ::ss].compute(), [intensity_p0, intensity_p1]
            )
        in_range = (p0, p1)
        print(f"intensity range ({ii}):", np.round(in_range, decimals=2))

        chunksize = cimg.chunksize
        if all(np.less(chunksize, 2048)):
            r_factor = int(np.ceil(np.divide(2048, chunksize)).min())
            cimg = cimg.rechunk(tuple(cc * r_factor for cc in chunksize))
        cimg = cimg.map_blocks(
            adjust_intensity,
            intensity_in_range=in_range,
            intensity_gamma=intensity_gamma,
            dtype="float32",
        )
        channels.append(da_to_zarr(cimg, zarr_store=zarr.TempStore()))

    img = da.array([da.from_zarr(cc, name=False) for cc in channels])
    if len(img) > 1:
        img = img.rechunk((-1, *img.chunksize[1:]))

    _binary_mask = img.map_overlap(
        segment_tile,
        depth={0: 0, 1: 128, 2: 128},
        boundary="none",
        dtype=bool,
        drop_axis=0,
        diameter=diameter,
        flow_threshold=flow_threshold,
        **kwargs,
    )
    print("run cellpose; number of chunks:", _binary_mask.numblocks)

    _binary_mask = da_to_zarr(_binary_mask, num_workers=2)
    binary_mask = da.from_zarr(_binary_mask, name=False)

    out_path_binary = (
        out_dir / f"_{reader.path.name.split('.')[0]}-cellpose-nucleus-binary.ome.tif"
    )
    palom.pyramid.write_pyramid(
        [binary_mask.astype("uint8")],
        out_path_binary,
        pixel_size=reader.pixel_size,
        downscale_factor=2,
        tile_size=1024,
        compression="zlib",
        save_RAM=True,
        is_mask=True,
    )

    end = int(time.perf_counter())
    print("\nelapsed (cellpose):", datetime.timedelta(seconds=end - start))

    _labeled_mask = dask_image.ndmeasure.label(binary_mask.rechunk(4096))[0]
    labeled_mask = da_to_zarr(_labeled_mask)

    end_label = int(time.perf_counter())
    print("\nelapsed (label):", datetime.timedelta(seconds=end_label - end))

    out_path = out_dir / f"{reader.path.name.split('.')[0]}-cellpose-nucleus.ome.tif"
    palom.pyramid.write_pyramid(
        [da.from_zarr(labeled_mask, name=False)],
        out_path,
        pixel_size=reader.pixel_size,
        downscale_factor=2,
        tile_size=1024,
        compression="zlib",
        save_RAM=True,
        is_mask=True,
    )

    end = int(time.perf_counter())
    print("\nelapsed (total):", datetime.timedelta(seconds=end - start))

    return out_path


def mask_to_contour(mask):
    struct_elem_d = skimage.morphology.disk(1).astype("bool")
    struct_elem_e = skimage.morphology.disk(1).astype("bool")
    return np.where(
        ndi.grey_dilation(mask, footprint=struct_elem_d)
        == ndi.grey_erosion(mask, footprint=struct_elem_e),
        False,
        True,
    )


def dilate_mask(mask, radius=1):
    struct_elem = skimage.morphology.disk(radius).astype("bool")
    return ndi.grey_dilation(mask, footprint=struct_elem)


def dilate_slide_mask(mask_path, radius=1, out_dir=None):
    """
    Create a dilated mask from the given mask path.

    Parameters:
        mask_path (str): Path to the input mask.
        radius (int): Radius for dilation.

    Returns:
        da.Array: Dilation of the input mask.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    start = int(time.perf_counter())
    reader = palom.reader.OmePyramidReader(mask_path)
    assert "-cellpose-nucleus.ome.tif" in reader.path.name, "Invalid mask file name."
    mask = reader.pyramid[0][0].rechunk(2048)

    dilated_mask = mask.map_overlap(
        skimage.segmentation.expand_labels,
        distance=radius,
        depth={0: 128, 1: 128},
        boundary="none",
        dtype="int32",
    )
    out_path = out_dir / reader.path.name.replace(
        "-cellpose-nucleus.ome.tif", "-cellpose-cell.ome.tif"
    )
    palom.pyramid.write_pyramid(
        [dilated_mask],
        out_path,
        pixel_size=reader.pixel_size,
        downscale_factor=2,
        tile_size=1024,
        compression="zlib",
        save_RAM=True,
        is_mask=True,
    )
    end = int(time.perf_counter())
    print("\nelapsed (dilate):", datetime.timedelta(seconds=end - start))
    return out_path


def make_qc_image(
    img_path,
    mask_path,
    out_dir,
    channel=0,
    intensity_p0=0.1,
    intensity_p1=99.9,
):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    reader = palom.reader.OmePyramidReader(img_path)
    reader_mask = palom.reader.OmePyramidReader(mask_path)
    out_path = out_dir / f"{reader.path.name.split('.')[0]}-cellpose-qc.ome.tif"

    img = reader.pyramid[0][channel]
    mask = reader_mask.pyramid[0][0]

    outline = mask.map_blocks(mask_to_contour, dtype="bool").map_blocks(
        skimage.util.img_as_ubyte, dtype="uint8"
    )
    with dask.diagnostics.ProgressBar():
        p0, p1 = np.percentile(img.compute(), [intensity_p0, intensity_p1])

    img = img.map_blocks(
        lambda x: skimage.exposure.rescale_intensity(
            x, in_range=(p0, p1), out_range="uint8"
        ).astype("uint8"),
        dtype="uint8",
    )
    palom.pyramid.write_pyramid(
        [outline, img],
        out_path,
        pixel_size=reader.pixel_size,
        channel_names=["Mask outline", f"Image (channel {channel})"],
        downscale_factor=2,
        tile_size=1024,
        compression="zlib",
        save_RAM=True,
        is_mask=False,
    )
    return out_path


def difference_mask_from_file(path_1, path_2, out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    reader_1 = palom.reader.OmePyramidReader(path_1)
    reader_2 = palom.reader.OmePyramidReader(path_2)
    pixel_size = reader_1.pixel_size

    if reader_1.pixel_size != reader_2.pixel_size:
        raise ValueError("Pixel sizes of the two images do not match.")

    out_name = reader_1.path.name.split("-cellpose-")[0] + "-cellpose-cytoplasm.ome.tif"
    out_path = out_dir / out_name

    out_mask = da.map_blocks(
        # this assumes one mask is a subset of the other mask; computing
        # `np.sum(x>0)` instead of `x.sum()` to prevent int overflow
        lambda x, y: (x != y) * x if np.sum(x > 0) > np.sum(y > 0) else (x != y) * y,
        reader_1.pyramid[0][0],
        reader_2.pyramid[0][0],
        dtype="int32",
    )
    palom.pyramid.write_pyramid(
        [out_mask],
        out_path,
        pixel_size=pixel_size,
        downscale_factor=2,
        tile_size=1024,
        compression="zlib",
        save_RAM=True,
        is_mask=True,
    )
    return


config = r"""
P183_10_C1,\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\183-Gray_Breast\P183_Leif_BRCA_Pysed\registration\P183_10_C1.ome.tif,\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\183-Gray_Breast\P183_Leif_BRCA_Pysed\mcmicro
P183_84_C2,\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\183-Gray_Breast\P183_Leif_BRCA_Pysed\registration\P183_84_C2.ome.tif,\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\183-Gray_Breast\P183_Leif_BRCA_Pysed\mcmicro
P183_BOT0007,\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\183-Gray_Breast\P183_Leif_BRCA_Pysed\registration\P183_BOT0007.ome.tif,\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\183-Gray_Breast\P183_Leif_BRCA_Pysed\mcmicro
P183_MGH23058,\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\183-Gray_Breast\P183_Leif_BRCA_Pysed\registration\P183_MGH23058.ome.tif,\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\183-Gray_Breast\P183_Leif_BRCA_Pysed\mcmicro
""".strip().split("\n")


def run(config, idx):
    row = config[idx]

    name, img_path, _out_dir = row.split(",")
    img_path = pathlib.Path(img_path)
    _out_dir = pathlib.Path(_out_dir)
    img_name = re.sub(r"\.ome\.tif{1,2}$", "", img_path.name, flags=re.IGNORECASE)

    print(f"\nProcessing slide {name} ...")
    out_dir = _out_dir / name / "segmentation" / img_name
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = segment_slide(
        img_path,
        channel=[0],
        out_dir=out_dir,
        intensity_p0=0.1,
        intensity_p1=99.95,
        intensity_gamma=1,
        diameter=20.0,
        flow_threshold=0.6,
        batch_size=8,
        min_size=0.5 * (15.0 / 2) ** 2 * np.pi,
    )

    make_qc_image(
        img_path,
        mask_path,
        out_dir,
        channel=0,
        intensity_p0=0.1,
        intensity_p1=99.95,
    )
    d_mask_path = dilate_slide_mask(mask_path, radius=5, out_dir=out_dir)
    difference_mask_from_file(d_mask_path, mask_path, out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run segmentation pipeline for a specific config index."
    )
    parser.add_argument(
        "index", type=int, help="Index of the config row to process (starting from 0)."
    )
    args = parser.parse_args()

    if args.index < 0 or args.index >= len(config):
        print(f"Error: index {args.index} is out of range (0 to {len(config) - 1})")
    else:
        run(config, args.index)
