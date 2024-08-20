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
    return_entropy_thumbnail: bool = False,
    level_center: float = -0.2,
    level_adjust: int = 0,
):
    assert thumbnail_level >= 0
    assert img_pyramid_downscale_factor >= 1
    assert dilation_radius >= 0
    assert (level_center >= -0.5) & (level_center <= 0.5)

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

    erange = np.subtract(*np.percentile(entropy_thumbnail, [99, 1]))
    _threshold = skimage.filters.threshold_otsu(entropy_thumbnail)
    _threshold += level_center * erange
    print(_threshold)
    threshold = _threshold + 0.1 * level_adjust * erange
    threshold = np.clip(threshold, entropy_thumbnail.min(), entropy_thumbnail.max())

    mask = entropy_thumbnail > threshold
    skimage.morphology.binary_dilation(
        mask, footprint=skimage.morphology.disk(radius=dilation_radius), out=mask
    )
    if plot:
        masks = entropy_img_to_masks(
            entropy_thumbnail, _threshold, 5, dilation_radius=dilation_radius
        )
        fig = _plot_tissue_mask(thumbnail, mask, entropy_thumbnail, masks=masks)
    if return_entropy_thumbnail:
        return mask, entropy_thumbnail
    return mask


def entropy_img_to_masks(entropy_img, threshold_center, num_levels, dilation_radius=2):
    assert num_levels >= 1
    num_levels = 2 * (num_levels // 2) + 1  # must be symmetrical
    erange = np.subtract(*np.percentile(entropy_img, [99, 1]))

    adjustments = np.arange(num_levels) - np.arange(num_levels).mean()
    thresholds = adjustments * 0.1 * erange + threshold_center

    mask = np.full(entropy_img.shape, adjustments.min() - 1, dtype="int8")
    for tt in thresholds:
        mask += skimage.morphology.binary_dilation(
            entropy_img > tt, footprint=skimage.morphology.disk(radius=dilation_radius)
        )

    return mask


def _plot_mask_levels(img, mask, num_levels, ax=None, tick_labels=None):
    import matplotlib.cm
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    assert num_levels % 2 == 1
    assert num_levels >= 3

    # levels = np.arange(-0.5 * (num_levels - 1), 0.5 * (num_levels - 1) + 1, 1)
    # levels = np.arange(0.5, mask.max() + 1, 1)
    levels = np.arange(num_levels) - np.arange(num_levels).mean()
    ccmm = matplotlib.cm.bwr_r
    ccmm.set_under("k")
    ccmm.set_over(matplotlib.cm.bwr_r(1.0))

    if ax is None:
        _, ax = plt.subplots()
    fig = ax.get_figure()

    ax.imshow(np.log1p(img), cmap="gray")
    cplot = ax.contourf(mask - 0.5, levels=levels, cmap=ccmm, alpha=0.5, extend="both")

    axins = inset_axes(
        ax,
        width=0.1,  # width: .1 inch
        height="100%",  # height: 100%
        loc="lower left",
        bbox_to_anchor=(1.05, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cax = fig.colorbar(cplot, cax=axins, ticks=levels[:-1])
    if tick_labels is not None:
        if len(tick_labels) == len(levels[:-1]):
            # cax.ax.set_yticklabels(["--", "-", "", "+", "++"])
            cax.ax.set_yticklabels(tick_labels)
    return fig


def _plot_tissue_mask(img, mask, entropy_img, masks=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm

    # set contrast min to min value that is not 0
    vimg = skimage.exposure.rescale_intensity(
        img,
        in_range=(img[img > 0].min(), img.max()),
        out_range="float",
    )
    # pad images for mask outline drawing
    vimg = np.pad(vimg, 2, constant_values=0)
    ventropy = np.pad(entropy_img, 2, constant_values=entropy_img.min())
    vmask = np.pad(mask, 2, constant_values=0)
    if masks is not None:
        vmasks = np.pad(masks, 2, constant_values=masks.min())
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        _plot_mask_levels(
            vimg, vmasks, num_levels=2 * np.abs(masks.min()) - 1, ax=axs[2]
        )
    else:
        # one figure with two subplots
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    vimg = matplotlib.cm.cividis(vimg)[..., :3]
    outline = skimage.segmentation.find_boundaries(vmask, mode="thick", connectivity=2)
    # use magenta as outline color
    vimg[..., [0, 2]] += outline[..., np.newaxis]
    axs[0].imshow(np.clip(vimg, 0, 1), interpolation="none")
    axs[1].imshow(ventropy, cmap="cividis", interpolation="none")
    fig.tight_layout()
    return fig


def _make_tissue_mask(img_path: str | pathlib.Path, thumbnail_factor, dilation_radius):
    reader = palom.reader.OmePyramidReader(img_path)
    # FIXME assuming 2x downsizing in the pyramid
    LEVEL = int(np.log2(thumbnail_factor))
    if LEVEL > len(reader.pyramid) - 1:
        LEVEL = len(reader.pyramid) - 1
    _thumbnail = reader.pyramid[LEVEL][:]

    level_diff = int(np.log2(thumbnail_factor)) - LEVEL
    thumbnail = np.array(
        [
            palom.img_util.cv2_downscale_local_mean(cc, 2**level_diff)
            for cc in _thumbnail
        ]
    )
    entropy_thumbnail = local_entropy(np.log1p(thumbnail.sum(axis=0)))
    mask = entropy_thumbnail > skimage.filters.threshold_otsu(entropy_thumbnail)
    skimage.morphology.binary_dilation(
        mask, footprint=skimage.morphology.disk(radius=dilation_radius), out=mask
    )
    return mask


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
