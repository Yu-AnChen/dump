import concurrent.futures
import itertools
import multiprocessing
import os
import uuid

import numpy as np
import ome_types
import palom
import skimage.transform
import tifffile
import tqdm
import zarr


def mask_to_binary(in_path, out_path):
    reader = palom.reader.OmePyramidReader(in_path)
    palom.pyramid.write_pyramid(
        [(reader.pyramid[0] > 0).astype("uint8") * 255],
        out_path,
        is_mask=True,
        downscale_factor=2,
        compression="zlib",
        save_RAM=True,
        pixel_size=reader.pixel_size,
    )


def adjust_rgb_contrast(in_path, in_range, out_range="uint8", jpeg_compression=True):
    import skimage.exposure

    reader = palom.reader.OmePyramidReader(in_path)
    assert reader.pyramid[0].shape[0] == 3
    assert reader.pixel_dtype == "uint8"
    assert ".ome.tif" in reader.path.name
    out_path = reader.path.parent / reader.path.name.replace(
        ".ome.tif", "-adjusted.ome.tif"
    )
    mosaic = reader.pyramid[0].map_blocks(
        lambda x: skimage.exposure.rescale_intensity(
            x, in_range=in_range, out_range=out_range
        ).astype("uint8")
    )
    palom.pyramid.write_pyramid(
        [mosaic],
        out_path,
        pixel_size=reader.pixel_size,
        downscale_factor=2,
        compression="zlib",
        save_RAM=True,
    )
    if jpeg_compression:
        out_path_jpeg = out_path.parent / out_path.name.replace(
            ".ome.tif", "-jpeg.ome.tif"
        )
        merge_rgb(out_path, out_path_jpeg)


def error(path, msg):
    print(f"\nERROR: {path}: {msg}")
    raise RuntimeError
    # sys.exit(1)


def merge_rgb(input, output, num_threads=0):
    tiff = tifffile.TiffFile(input)
    if len(tiff.series) != 1:
        error(
            input,
            f"Input must contain only one OME image; found {len(tiff.series)}"
            " instead",
        )
    series = tiff.series[0]
    if series.axes != "CYX":
        error(
            input,
            "Input must have shape (channel, height, width); found"
            f" {series.dims} = {series.shape} instead",
        )
    if series.shape[0] != 3:
        error(
            input,
            f"Input must have exactly 3 channels; found {series.shape[0]}" " instead",
        )
    if series.dtype != "uint8":
        error(input, f"Input must have pixel type uint8; found {series.dtype}")

    if output.exists():
        error(output, "Output file exists, remove before continuing")

    if num_threads == 0:
        if hasattr(os, "sched_getaffinity"):
            num_threads = len(os.sched_getaffinity(0))
        else:
            num_threads = multiprocessing.cpu_count()
        print(f"Using {num_threads} worker threads based on available CPUs")
        print()

    image0 = zarr.open(series.aszarr(level=0))
    metadata = ome_types.from_xml(tiff.ome_metadata, parser="xmlschema")

    base_shape = image0.shape[1:]
    tile_size = 1024
    num_levels = np.ceil(np.log2(max(base_shape) / tile_size)) + 1
    factors = 2 ** np.arange(num_levels)
    shapes = [
        tuple(s) for s in (np.ceil(np.array(base_shape) / factors[:, None])).astype(int)
    ]
    cshapes = [tuple(s) for s in np.ceil(np.divide(shapes, tile_size)).astype(int)]
    print("Pyramid level sizes:")
    for i, shape in enumerate(shapes):
        shape_fmt = "%d x %d" % (shape[1], shape[0])
        print(f"    Level {i + 1}: {shape_fmt}", end="")
        if i == 0:
            print(" (original size)", end="")
        print()
    print()

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

    def tiles0():
        zimg = image0
        ts = tile_size
        ch, cw = cshapes[0]
        for j in range(ch):
            for i in range(cw):
                tile = zimg[:, ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
                tile = tile.transpose(1, 2, 0)
                # Must copy() to provide contiguous array for jpeg encoder.
                yield tile.copy()

    def tiles(level):
        if level == 0:
            yield from tiles0()
        tiff_out = tifffile.TiffFile(output, is_ome=False)
        zimg = zarr.open(tiff_out.series[0].aszarr(level=level - 1))
        ts = tile_size * 2

        def tile(coords):
            j, i = coords
            tile = zimg[ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
            tile = skimage.transform.downscale_local_mean(tile, (2, 2, 1))
            tile = np.round(tile).astype(np.uint8)
            return tile

        ch, cw = cshapes[level]
        coords = itertools.product(range(ch), range(cw))
        yield from pool.map(tile, coords)

    def progress(level):
        ch, cw = cshapes[level]
        t = tqdm.tqdm(
            tiles(level),
            desc=f"    Level {level + 1}",
            total=ch * cw,
            unit="tile",
        )
        # Fix issue with tifffile's peek_iterator causing a missed update.
        t.update()
        return iter(t)

    metadata.uuid = uuid.uuid4().urn
    # Reconfigure metadata for a single 3-sample channel.
    mpixels = metadata.images[0].pixels
    del mpixels.channels[1:]
    del mpixels.planes[1:]
    mpixels.channels[0].name = None
    mpixels.channels[0].samples_per_pixel = 3
    mpixels.tiff_data_blocks = [ome_types.model.TiffData(plane_count=1)]
    # Drop the optional PyramidResolution annotation rather than recompute it.
    metadata.structured_annotations = [
        a
        for a in metadata.structured_annotations
        if a.namespace != "openmicroscopy.org/PyramidResolution"
    ]
    ome_xml = metadata.to_xml()
    # Hack to work around ome_types always writing the default color.
    ome_xml = ome_xml.replace('Color="-1"', "")

    software = tiff.pages[0].software
    print("Writing new OME-TIFF:")
    with tifffile.TiffWriter(output, ome=False, bigtiff=True) as writer:
        writer.write(
            data=progress(0),
            shape=shapes[0] + (3,),
            subifds=num_levels - 1,
            dtype="uint8",
            tile=(tile_size, tile_size),
            compression="jpeg",
            compressionargs={"level": 90},
            software=software,
            description=ome_xml.encode(),
            metadata=None,
        )
        for level, shape in enumerate(shapes[1:], 1):
            writer.write(
                data=progress(level),
                shape=shape + (3,),
                subfiletype=1,
                dtype="uint8",
                tile=(tile_size, tile_size),
                compression="jpeg",
                compressionargs={"level": 90},
            )
        print()
