import dask
import dask.diagnostics
import numpy as np
import skimage.transform
import skimage.util
from palom.cli.align_he import get_reader


def digitize_img(
    img: np.ndarray, n_bins: int, p0: float = 0, p100: float = 100
) -> np.ndarray:
    """
    Digitizes a grayscale image into bins based on percentiles, optionally
    excluding the lowest and highest intensity pixels based on p0 and p100
    thresholds.

    Args:
        img: 2D NumPy array of image intensities.
        n_bins: Number of bins to divide the data into.
        p0: Lower percentile threshold (default 0).
        p100: Upper percentile threshold (default 100).

    Returns:
        A 2D NumPy array with same shape as `img`, where values are bin indices
        and NaN where the pixel was excluded.
    """
    assert n_bins > 0
    mask = np.full(img.shape, True)
    if p0 > 0:
        mask[img <= np.percentile(img, p0)] = False
    if p100 < 100:
        mask[img >= np.percentile(img, p100)] = False
    assert np.any(mask)
    valid_pxs = img[mask]
    bin_edges = np.percentile(valid_pxs, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1  # ensure upper bound

    digit_pxs = np.digitize(valid_pxs, bins=bin_edges)

    out = np.full(mask.shape, np.nan)
    out[mask] = digit_pxs
    return out


class WsiPatchSampler:
    """
    A utility class to sample representative patches from a Whole Slide Image
    (WSI) using intensity histogram-based binning.

    Args:
        img_path: Path to the WSI file.
        channel: Which channel to use (default 0).
        n_bins: Number of histogram bins/classes (default 8).
        patch_size: Size of each patch in pixels (default 1024).
        n_patches: Number of patches to sample per class (default 8).
    """

    def __init__(
        self,
        img_path: str,
        channel: int = 0,
        n_bins: int = 8,
        patch_size: int = 1024,
        n_patches: int = 8,
    ) -> None:
        self.img_path = img_path
        self.channel = channel
        self.n_bins = n_bins
        self.patch_size = patch_size
        self.n_patches = n_patches

        # Load image reader
        self.reader = get_reader(self.img_path)(self.img_path)

        level_downsamples = [
            (kk, vv)
            for kk, vv in self.reader.level_downsamples.items()
            if vv <= self.patch_size
        ]
        if not level_downsamples:
            raise ValueError("Patch size too small.")
        self._level, self._downsample = sorted(
            level_downsamples, key=lambda x: x[1], reverse=True
        )[0]

    def _sample_patch_coordinates(self, dimg: np.ndarray) -> np.ndarray:
        """Sample patch coordinates from each class/bin."""
        sample_classes = np.unique(dimg[np.isfinite(dimg)])
        n_bins = len(sample_classes)
        coords = np.zeros((n_bins * self.n_patches, 2), dtype="int")
        for idx, cc in enumerate(sample_classes):
            options_rc = np.array(np.where(dimg == cc)).T
            options_rc *= self.patch_size

            n_pad = self.n_patches - len(options_rc)
            if n_pad > 0:
                options_rc = np.pad(options_rc, ([0, n_pad], [0, 0]), mode="edge")

            np.random.shuffle(options_rc)
            coords[idx * self.n_patches : (idx + 1) * self.n_patches] = options_rc[
                : self.n_patches
            ]
        return coords

    def _mask_non_full_crop(self, dimg: np.ndarray) -> np.ndarray:
        """Mask out patches that would fall outside the image boundary."""
        rc = np.array(np.where(np.isfinite(dimg))).T
        ends = rc * self.patch_size + self.patch_size
        is_full_crop = np.all(ends < self.reader.pyramid[0].shape[1:], axis=1)
        rr, cc = rc[~is_full_crop].T
        dimg[rr, cc] = np.nan
        return dimg

    def extract_patches(
        self, channel: int = 0, montage: bool = False, p0: float = 30, p100: float = 100
    ) -> np.ndarray:
        """
        Extract representative patches based on intensity histogram binning.

        Args:
            channel: Channel index to extract (default 0).
            montage: Whether to return a visual montage (default False).
            p0: Lower percentile threshold for binning (default 30).
            p100: Upper percentile threshold for binning (default 100).

        Returns:
            Array of image patches (n_bins * n_patchs, patch_size, patch_size)
            or a single montage 2D image (if montage=True).
        """
        _img = np.array(self.reader.pyramid[self._level][self.channel])
        img = skimage.transform.rescale(
            _img,
            self._downsample / self.patch_size,
            anti_aliasing=True,
            preserve_range=True,
        )
        dimg = digitize_img(img=img, n_bins=self.n_bins, p0=p0, p100=p100)
        dimg = self._mask_non_full_crop(dimg)
        coords = self._sample_patch_coordinates(dimg)

        tiles = [
            self.reader.pyramid[0][
                channel, rr : rr + self.patch_size, cc : cc + self.patch_size
            ]
            for rr, cc in coords
        ]

        with dask.diagnostics.ProgressBar():
            tiles = dask.compute(*tiles)

        if montage:
            grid_shape = (len(tiles) // self.n_patches, self.n_patches)
            return skimage.util.montage(tiles, grid_shape=grid_shape)

        return np.asarray(tiles)


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
# sampler = WsiPatchSampler("path/to/image.ome.tif")
# patches = sampler.extract_patches(montage=True)
# plt.imshow(patches)
# plt.show()
