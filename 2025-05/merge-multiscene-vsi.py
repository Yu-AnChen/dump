import pathlib
import re

import dask.array as da
import numpy as np
import palom
import tqdm
import zarr


def parse_scene_origins(metadata_string):
    # Regex pattern
    pattern = r'"tag":\s*2018,\s*"name":\s*"RWC frame origin",\s*"value":\s*"\(([^,]+),\s*([^)]+)\)"'

    # Find matches
    matches = re.findall(pattern, metadata_string)
    matches = np.array([(float(xx), float(yy)) for xx, yy in matches])

    _, idx, counts = np.unique(matches, return_counts=True, return_index=True, axis=0)
    mask = counts == 2
    return matches[sorted(idx[mask])]


def merge_scenes(vsi_path, out_path):
    reader = palom.reader.VsiReader(vsi_path)

    if out_path is None:
        out_path = reader.path.parent / reader.path.name.replace(
            ".vsi", "-scene_merged.ome.tif"
        )
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    assert not out_path.exists()

    slide = reader.store._slide
    if slide.num_scenes == 1:
        print(f"Only 1 scene detected in {vsi_path}, aborting")
        return

    positions = parse_scene_origins(slide.raw_metadata)
    positions /= reader.pixel_size
    positions = np.fliplr(positions).round().astype("int")
    offset = positions.min(axis=0)
    positions -= offset

    scene_shapes = [
        (
            slide.get_scene(ii).get_zoom_level_info(0).size.height,
            slide.get_scene(ii).get_zoom_level_info(0).size.width,
        )
        for ii in range(slide.num_scenes)
    ]

    extent = np.add(positions, scene_shapes).max(axis=0)

    num_channels = reader.pyramid[0].shape[0]
    # FIXME hardcoded to use max, does not work for images w/ dark background
    fill_values = np.array(np.percentile(reader.pyramid[-1], 90, axis=(1, 2))).astype(
        reader.pixel_dtype
    )
    out = zarr.zeros(
        (num_channels, *extent),
        chunks=(1, 2048, 2048),
        dtype=reader.pixel_dtype,
    )
    for ii, vv in enumerate(fill_values):
        out[ii] = vv

    for ii in tqdm.trange(slide.num_scenes):
        img = slide.get_scene(ii).read_block()
        rs, cs = positions[ii]
        re, ce = np.add(positions[ii], img.shape[:2])
        out[:, rs:re, cs:ce] = np.moveaxis(img, 2, 0)

    da_out = da.from_zarr(out, name=False)

    palom.pyramid.write_pyramid(
        [da_out],
        out_path,
        pixel_size=reader.pixel_size,
        downscale_factor=2,
        compression="zlib",
        save_RAM=True,
        tile_size=1024,
    )
