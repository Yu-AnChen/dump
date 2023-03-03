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
#                      Find out how much overlap is needed                     #
# ---------------------------------------------------------------------------- #
import skimage.segmentation

def edger_bbox_to_edge(mask):
    '''
    For objects touching edges, compute their distances to the touching front
    '''
    mask = np.array(mask)
    h, w = mask.shape
    cleared = skimage.segmentation.clear_border(mask)
    mask[cleared > 0] = 0
    if np.all(mask == 0):
        return np.array(0)
    # return mask
    results = skimage.measure.regionprops_table(mask)
    rs = results
    return np.concatenate([
        rs['bbox-2'][rs['bbox-0'] == 0],
        rs['bbox-3'][rs['bbox-1'] == 0],
        h - rs['bbox-0'][rs['bbox-2'] == h],
        w - rs['bbox-1'][rs['bbox-3'] == w],
    ])


# ---------------------------------------------------------------------------- #
#                         Next is to monitor RAM usage                         #
# ---------------------------------------------------------------------------- #