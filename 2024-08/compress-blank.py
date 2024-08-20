import pathlib
import re
import time

import numpy as np
import ome_types
import palom
import skimage.exposure
import skimage.filters
import skimage.filters.thresholding
import skimage.morphology
import skimage.segmentation
import tifffile
import zarr

PYRAMID_DEFAULTS = dict(
    downscale_factor=2,
    compression="zlib",
    tile_size=1024,
    save_RAM=True,
    is_mask=False,
)


def src_tif_tags(img_path):
    kwargs_tifffile = {}
    with tifffile.TiffFile(img_path) as tif:
        kwargs_tifffile.update(
            dict(
                photometric=tif.pages[0].photometric.value,
                resolution=tif.pages[0].resolution,
                resolutionunit=tif.pages[0].resolutionunit.value,
                software=tif.pages[0].software,
            )
        )
    return kwargs_tifffile


def get_img_path(img_path):
    img_path = pathlib.Path(img_path)
    assert img_path.exists
    assert re.search(r"(?i).ome.tiff?$", img_path.name) is not None
    return img_path


def get_output_path(output_path, img_path):
    img_name = re.sub(r"(?i).ome.tiff?$", ".ome.tif", img_path.name)
    if output_path is None:
        output_path = img_path.parent / re.sub(
            r".ome.tif$", "-tissue.ome.tif", img_name
        )
    output_path = pathlib.Path(output_path)
    assert output_path != img_path
    assert re.search(r"(?i).ome.tiff?$", output_path.name) is not None
    output_path.parent.mkdir(exist_ok=True, parents=True)
    return output_path


def local_entropy(img, kernel_size=9):
    img = skimage.exposure.rescale_intensity(img, out_range=(0, 1024 - 1)).astype(
        np.uint16
    )
    return skimage.filters.rank.entropy(img, np.ones((kernel_size, kernel_size)))


def make_tissue_mask(
    img_path: str | pathlib.Path,
    thumbnail_level: int,
    img_pyramid_downscale_factor: int,
    dilation_radius: int,
    plot: bool = False,
    level_center: float = -0.2,
    level_adjust: int = 0,
):
    assert thumbnail_level >= 0
    assert img_pyramid_downscale_factor >= 1
    assert dilation_radius >= 0
    # assert (level_center >= -0.5) & (level_center <= 0.5)

    assert level_adjust in np.arange(-2, 3, 1)

    reader = palom.reader.OmePyramidReader(img_path)
    LEVEL = thumbnail_level
    if LEVEL > len(reader.pyramid) - 1:
        LEVEL = len(reader.pyramid) - 1
    _thumbnail = reader.pyramid[LEVEL][:]
    thumbnail = np.array(
        [
            palom.img_util.cv2_downscale_local_mean(
                cc, img_pyramid_downscale_factor ** (thumbnail_level - LEVEL)
            )
            for cc in _thumbnail
        ]
    )
    thumbnail = np.log1p(thumbnail.sum(axis=0))
    entropy_thumbnail = local_entropy(thumbnail)

    erange = np.ptp(entropy_thumbnail)
    _threshold = skimage.filters.threshold_otsu(entropy_thumbnail)
    _max = entropy_thumbnail.max() - 0.1 * erange
    _min = entropy_thumbnail.min() + 0.1 * erange
    _threshold = np.clip(_threshold, _min, _max)

    level_center = np.clip(
        level_center, (_min - _threshold) / erange, (_max - _threshold) / erange
    )

    # forcing threshold to be within 10th and 90th percent of the range
    _threshold += level_center * erange
    print(_threshold)

    thresholds = np.concatenate(
        [
            np.linspace(entropy_thumbnail.min(), _threshold, 4)[1:],
            np.linspace(_threshold, entropy_thumbnail.max(), 4)[1:-1],
        ]
    )
    level_adjusts = np.arange(-2, 3, 1)
    masks = entropy_img_to_masks(entropy_thumbnail, thresholds, dilation_radius)
    mask = masks[list(level_adjusts).index(level_adjust)]

    if plot:
        fig = plot_tissue_mask(
            thumbnail,
            entropy_thumbnail,
            masks,
            thresholds,
            list(level_adjusts).index(level_adjust),
        )
        fig.suptitle(reader.path.name)

    return mask


def entropy_img_to_masks(entropy_img, thresholds, dilation_radius):
    masks = np.full((len(thresholds), *entropy_img.shape), fill_value=False, dtype=bool)
    for idx, tt in enumerate(thresholds):
        masks[idx] = skimage.morphology.binary_dilation(
            entropy_img > tt,
            footprint=skimage.morphology.disk(radius=dilation_radius),
        )
    return masks


def plot_tissue_mask(img, entropy_img, masks, thresholds, selected_mask_idx):
    import matplotlib.pyplot as plt

    # set contrast min to min value that is not 0
    vimg = skimage.exposure.rescale_intensity(
        img,
        in_range=(img[img > 0].min(), img.max()),
        out_range="float",
    )
    # pad images for mask outline drawing
    pad_size = np.ceil(np.max(img.shape) * 0.01).astype("int")
    vimg = np.pad(vimg, pad_size, constant_values=0)
    ventropy = np.pad(entropy_img, pad_size, constant_values=entropy_img.min())
    vmasks = np.pad(
        masks,
        [(0, 0), (pad_size, pad_size), (pad_size, pad_size)],
        constant_values=False,
    )
    vmask = vmasks[selected_mask_idx]

    subplot_shape = (2, 1)
    if np.divide(*img.shape) > 1.2:
        subplot_shape = (1, 2)

    fig, axs = plt.subplots(*subplot_shape, sharex=True, sharey=True)

    axs[0].imshow(vimg, cmap="cividis")
    axs[0].contour(vmask, levels=[0.5], colors=["w"], linewidths=1)
    # axs[1].imshow(ventropy, cmap="cividis", interpolation="none")

    _plot_entropy_mask_levels(
        ventropy, vmasks, thresholds, selected_mask_idx, img=vimg, ax=axs[1]
    )
    return fig


def _plot_entropy_mask_levels(
    entropy_img, masks, thresholds, selected_mask_idx=None, img=None, ax=None
):
    import itertools

    import matplotlib.cm
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    assert len(masks) == len(thresholds) == 5

    levels = np.arange(7) - 0.5
    tick_labels = np.arange(5) - 2

    if selected_mask_idx is None:
        selected_mask_idx = 2
    assert selected_mask_idx in range(5)

    if ax is None:
        _, ax = plt.subplots()
    fig = ax.get_figure()

    if img is None:
        img = [[0]]
    ax.imshow(np.log1p(img), cmap="gray")

    ax.contourf(masks.sum(axis=0), cmap="coolwarm_r", levels=levels, alpha=0.75)
    ax.contour(masks[selected_mask_idx], levels=[0.5], colors=["w"], linewidths=1)
    axins = inset_axes(
        ax,
        width=0.1,  # width: .1 inch
        height="100%",  # height: 100%
        loc="lower left",
        bbox_to_anchor=(1.05, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    colors = matplotlib.cm.coolwarm_r(np.linspace(0, 1, 6), alpha=0.75)
    yys = [entropy_img.min()] + list(thresholds) + [entropy_img.max()]

    for pp, cc in zip(itertools.pairwise(yys), colors):
        axins.fill_between([0, 1], *pp, color=cc, step="post")

    axins.set_xlim(0, 1)
    axins.set_ylim(yys[0], yys[-1])
    axins.axes.yaxis.tick_right()
    axins.set_yticks(yys[1:-1], labels=tick_labels)
    axins.set_xticks([])
    axins.axhline(thresholds[selected_mask_idx], color="w", linewidth=3)

    return fig


def write_masked(img_path, output_path, tissue_mask, mask_upscale_factor):
    reader = palom.reader.OmePyramidReader(img_path)

    # match mask shape to full res image
    _, H, W = reader.pyramid[0].shape
    mask_full_zarr = zarr.full((H, W), fill_value=False, chunks=1024)

    mask_full = palom.img_util.repeat_2d(tissue_mask, (mask_upscale_factor,) * 2)[
        :H, :W
    ]
    h, w = mask_full.shape
    # mask_full size might be smaller than image size
    mask_full_zarr[:h, :w] = mask_full[:h, :w]
    mask_full = None

    mosaic = reader.pyramid[0] * mask_full_zarr

    try:
        tif_tags = src_tif_tags(img_path)
    except Exception:
        tif_tags = {}

    software = "detect_tissue_v0"
    tif_tags["software"] = f"{tif_tags.get('software', None)}-{software}"

    palom.pyramid.write_pyramid(
        mosaics=[mosaic],
        output_path=output_path,
        **{
            **dict(
                pixel_size=reader.pixel_size,
                kwargs_tifffile=tif_tags,
            ),
            **PYRAMID_DEFAULTS,
        },
    )

    ome = ome_types.from_tiff(img_path)
    ome.creator = f"{ome.creator}-{software}"
    tifffile.tiffcomment(output_path, ome.to_xml().encode())

    return output_path


def process_file(
    img_path,
    thumbnail_level=6,
    img_pyramid_downscale_factor=2,
    dilation_radius=2,
    output_path=None,
):
    img_path = get_img_path(img_path)
    output_path = get_output_path(output_path, img_path)

    # tissue_mask = make_tissue_mask(img_path, thumbnail_factor, dilation_radius)
    tissue_mask = make_tissue_mask(
        img_path, thumbnail_level, img_pyramid_downscale_factor, dilation_radius
    )
    write_masked(
        img_path,
        output_path,
        tissue_mask,
        img_pyramid_downscale_factor**thumbnail_level,
    )


test_images = sorted(
    pathlib.Path(
        r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20240714-deform-registration-crc/img-data"
    ).glob("*-ori.tif")
)
"""
in_dir = r"X:\cycif-production\184-CRC_polyps\Orion_pilot-20240507"
files = sorted(pathlib.Path(in_dir).glob("*.ome.tiff"))[3:]

process_file(
    r"X:\cycif-production\184-CRC_polyps\Orion_pilot-20240507\LSP23366_001_P54_HMS_Artemis4_correct_001080.ome.tiff",
    64,
    2,
    r"X:\cycif-production\184-CRC_polyps\Orion_pilot-20240507\LSP23366_001_P54_HMS_Artemis4_correct_001080-test-ori.ome.tiff",
)
"""
