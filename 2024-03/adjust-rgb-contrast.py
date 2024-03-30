import palom
import skimage.exposure
import pathlib


def adjust_rgb_contrast(
    in_path: str | pathlib.Path,
    out_path: str | pathlib.Path,
    contrast_min: float,
    contrast_max: float,
    force=False,
):
    reader = palom.reader.OmePyramidReader(in_path)
    assert reader.pixel_dtype == "uint8", "input image must be a 8-bit image"
    assert in_path != out_path, "input path cannot be the same as output path"
    if not force:
        assert not pathlib.Path(
            out_path
        ).exists(), "output file already exists and force is `False`"
    mosaic = reader.pyramid[0].map_blocks(
        lambda x: skimage.exposure.rescale_intensity(
            x, in_range=(contrast_min, contrast_max), out_range=(0, 255)
        ).astype("uint8")
    )
    palom.pyramid.write_pyramid(
        [mosaic],
        out_path,
        pixel_size=reader.pixel_size,
        downscale_factor=2,
        compression="zlib",
    )
    return out_path


if __name__ == '__main__':
    import fire

    fire.Fire(adjust_rgb_contrast)
