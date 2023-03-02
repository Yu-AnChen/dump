import skimage.measure
import numpy as np
import dask.array as da
import dask
import dask.diagnostics

_all_mask_props = [
    "label", "centroid", "area",
    "major_axis_length", "minor_axis_length",
    "eccentricity", "solidity", "extent", "orientation"
]

size = 50_000

# ---------------------------------------------------------------------------- #
#                             Run with scikit-image                            #
# ---------------------------------------------------------------------------- #
tmask = np.arange(size, dtype=np.int32).reshape(1, size)
# %%time
_ = skimage.measure.regionprops_table(tmask, properties=_all_mask_props)


# ---------------------------------------------------------------------------- #
#                   Process-based parallelization using dask                   #
# ---------------------------------------------------------------------------- #
# this doesn't seem to matter
import threadpoolctl
threadpoolctl.threadpool_limits(1)

damask = da.arange(size, dtype=np.int32, chunks=1000).reshape(1, size)
results = [
    dask.delayed(skimage.measure.regionprops_table)(b, properties=_all_mask_props)
    for b in damask.blocks.ravel()
]
with dask.diagnostics.ProgressBar():
    _ = dask.compute(*results, scheduler='processes', num_workers=4)


# ---------------------------------------------------------------------------- #
#                         Next is to monitor RAM usage                         #
# ---------------------------------------------------------------------------- #