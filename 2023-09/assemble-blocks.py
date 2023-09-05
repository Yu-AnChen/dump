import pathlib

import dask.array as da
import joblib
import numpy as np
import tifffile
import zarr

import palom

tiles = [
    [(int(p.name.split('_')[1]), int(p.name.split('_')[2])), p]
    for p in sorted(pathlib.Path('.').glob('*.tiff'))
]

sorted_tiles = sorted(tiles, key=lambda x: x[0])

n_rows, n_cols = sorted_tiles[-1][0]

_img = tifffile.imread(sorted_tiles[0][1])
dtype = _img.dtype
tile_height, tile_width, n_channels = _img.shape

# if RAM is limited
# z = zarr.open(
#     'mosaic.zarr',
#     mode='w',
#     shape=(n_channels, (n_rows+1)*tile_height, (n_cols+1)*tile_width),
#     dtype='uint8',
#     chunks=(1, 1024, 1024)
# )

z = zarr.zeros(
    (n_channels, (n_rows+1)*tile_height, (n_cols+1)*tile_width),
    dtype='uint8',
    chunks=(1, 1024, 1024)
)

def wrap(tile):
    (rs, cs), p = tile
    rs *= tile_height
    cs *= tile_width
    re = rs + tile_height
    ce = cs + tile_width
    z[:, rs:re, cs:ce] = np.moveaxis(tifffile.imread(p), 2, 0)

_ = joblib.Parallel(n_jobs=12, verbose=1, backend='threading')(
    joblib.delayed(wrap)(t) for t in sorted_tiles
)

zimg = da.from_zarr(z)
palom.pyramid.write_pyramid(
    [zimg], 'C06-fake.ome.tif',
    pixel_size=0.325, compression='zlib',
    tile_size=1024, save_RAM=True
)
