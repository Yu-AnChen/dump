import dask
import dask.diagnostics
import numpy as np
import palom
import skimage.util
from palom.cli.align_he import get_reader


def digitize_img(img, n_bins, p0=0, p100=100):
    assert n_bins > 0
    mask = np.full(img.shape, True)
    if p0 > 0:
        mask[img <= np.percentile(img, p0)] = False
    if p100 < 100:
        mask[img >= np.percentile(img, p100)] = False
    assert np.any(mask)
    valid_pxs = img[mask]
    bin_edges = np.percentile(valid_pxs, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1
    digit_pxs = np.digitize(valid_pxs, bins=bin_edges)

    out = np.full(mask.shape, np.nan)
    out[mask] = digit_pxs
    return out


class WsiPatchSampler:
    # FIXME downsize_factor should be equal to crop_size
    # FIXME should remove level arg and select that based on crop size
    def __init__(
        self,
        img_path,
        channel=0,
        level=10,
        downsize_factor=1024,
        n_classes=8,
        p0=20,
        p100=100,
        crop_size=1024,
        n_patches=8,
    ):
        self.img_path = img_path
        self.channel = channel
        self.level = level
        self.downsize_factor = downsize_factor
        self.n_classes = n_classes
        self.p0 = p0
        self.p100 = p100
        self.crop_size = crop_size
        self.n_patches = n_patches

        # Load image reader
        self.reader = get_reader(self.img_path)(self.img_path)

    def _sample_patch_coordinates(self, dimg):
        sample_classes = np.unique(dimg[np.isfinite(dimg)])

        n_classes = len(sample_classes)

        coords = np.zeros((n_classes * self.n_patches, 2), dtype="int")
        for idx, cc in enumerate(sample_classes):
            options_rc = np.array(np.where(dimg == cc)).T
            options_rc *= self.downsize_factor

            n_pad = self.n_patches - len(options_rc)
            if n_pad > 0:
                options_rc = np.pad(options_rc, ([0, n_pad], [0, 0]), mode="edge")

            np.random.shuffle(options_rc)
            coords[idx * self.n_patches : (idx + 1) * self.n_patches] = options_rc[
                : self.n_patches
            ]
        return np.array(coords)

    def _mask_non_full_crop(self, dimg):
        rc = np.array(np.where(np.isfinite(dimg))).T
        ends = rc * self.downsize_factor + self.crop_size
        is_full_crop = np.all(ends < self.reader.pyramid[0].shape[1:], axis=1)

        rr, cc = rc[~is_full_crop].T
        dimg[rr, cc] = np.nan
        return dimg

    def extract_patches(self, channel=0, montage=False):
        """Extracts patches from the WSI based on histogram classification."""
        img = get_thumbnail_channels(self.img_path, self.level)[self.channel]
        dimg = digitize_img(img=img, n_bins=self.n_classes, p0=self.p0, p100=self.p100)
        dimg = self._mask_non_full_crop(dimg)
        coords = self._sample_patch_coordinates(dimg)

        tiles = [
            self.reader.pyramid[0][
                channel, rr : rr + self.crop_size, cc : cc + self.crop_size
            ]
            for rr, cc in coords
        ]

        with dask.diagnostics.ProgressBar():
            tiles = dask.compute(*tiles)

        if montage:
            return skimage.util.montage(
                tiles, grid_shape=(len(tiles) / self.n_patches, self.n_patches)
            )
        return tiles


def get_thumbnail_channels(img_path, thumbnail_level, _dfactor=None):
    Reader = get_reader(img_path)
    reader = Reader(img_path)

    if _dfactor is None:
        _prev, _next = (
            np.repeat(np.sort(list(reader.level_downsamples.values())), 2)[1:-1]
            .reshape(-1, 2)
            .T
        )
        _dfactor = np.unique(np.round(_next / _prev).astype("int"))
        assert len(_dfactor) == 1, (
            f"cannot find a single downsize factor for\n\t{img_path}"
            f"\n\tdetected level downsamples {reader.level_downsamples}"
            f"\n\tspecify it using `_dfactor`"
        )
        _dfactor = _dfactor[0]

    last_level = len(reader.pyramid) - 1
    if thumbnail_level > last_level:
        channels = reader.pyramid[last_level]
        channels = np.array(
            [
                palom.img_util.cv2_downscale_local_mean(
                    np.array(cc), factor=_dfactor ** (thumbnail_level - last_level)
                )
                for cc in channels
            ]
        )
    else:
        channels = reader.pyramid[thumbnail_level].compute()
    return channels


# Example Usage:
# sampler = WsiPatchSampler("path/to/image.ome.tiff")
# patches = sampler.extract_patches()
# montage = sampler.create_montage(patches)
