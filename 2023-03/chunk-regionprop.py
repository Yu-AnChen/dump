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


import palom

img_path = '/mnt/orion/Mercury-3/20230227/B6_8EOHXW/segmentation/B6_8EOHXW/nucleiRing.ome.tif'
reader = palom.reader.OmePyramidReader(img_path)
mask = reader.pyramid[0][0]
_all_mask_props = [
    "label", "centroid", "area",
]
full_result = skimage.measure.regionprops_table(np.array(mask), properties=_all_mask_props)
full_result.keys()


edge_size = [dask.delayed(edger_bbox_to_edge)(m) for m in mask.blocks.ravel()]
with dask.diagnostics.ProgressBar():
    # FIXME should use process-based parallelization
    edge_size = dask.compute(*edge_size)

es = np.max([np.max(e) for e in edge_size])

omask = da.overlap.overlap(mask, 48, 'none')
block_results = [dask.delayed(skimage.measure.regionprops_table)(b, properties=_all_mask_props) for b in omask.blocks.ravel()]

with dask.diagnostics.ProgressBar():
    # FIXME should use process-based parallelization
    block_results = dask.compute(*block_results)

import pandas as pd

df_block = (pd.concat([pd.DataFrame(r) for r in block_results])
    .sort_values(['label', 'area'])
    .drop_duplicates(keep='last', subset='label')
    .set_index('label'))


df = pd.DataFrame(full_result).set_index('label')
# compare result
np.sum(df_block.area != df.area)


# use block_info to get offsets for cropping (process parallelization)
# and apply coordinates offsets
def test(b, block_info=None):
    print(block_info[None]['array-location'])
    return np.atleast_2d(1)

mask.map_blocks(test, dtype=int).compute()

# ---------------------------------------------------------------------------- #
#                      joblib for process parallelization                      #
# ---------------------------------------------------------------------------- #
rechunked = mask.rechunk(2048)
rr, cc = np.indices(rechunked.numblocks)
rsize, csize = rechunked.chunksize

es = 48
rs = np.clip(rr * rsize - es, 0, mask.shape[0])
re = np.clip((rr+1) * rsize + es, 0, mask.shape[0])
cs = np.clip(cc * csize - es, 0, None)
ce = np.clip((cc+1) * csize + es, None, mask.shape[1])

import joblib

def wrap(rrs, rre, ccs, cce):
    mm = reader.pyramid[0][0][rrs:rre, ccs:cce]
    result = skimage.measure.regionprops_table(mm.compute(), properties=_all_mask_props)
    result['offset_row'] = np.ones(len(result['label']), dtype=int) * rrs
    result['offset_col'] = np.ones(len(result['label']), dtype=int) * ccs
    return result
_ = joblib.Parallel(verbose=1, backend='loky', n_jobs=12, return_as='list')(
    joblib.delayed(wrap)(rrs, rre, ccs, cce)
    for rrs, rre, ccs, cce in zip(*np.array([rs, re, cs, ce]).reshape(4, -1))
)

# ---------------------------------------------------------------------------- #
#                         Next is to monitor RAM usage                         #
# ---------------------------------------------------------------------------- #
