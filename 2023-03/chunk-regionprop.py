import dask.array as da
import joblib
import numpy as np
import skimage.measure
import skimage.segmentation
import tifffile
import tqdm
import zarr


_all_mask_props = [
    "label",
    "centroid",
    "area",
    "major_axis_length",
    "minor_axis_length",
    "eccentricity",
    "solidity",
    "extent",
    "orientation",
]

# _all_mask_props = [
#     "label",
#     "centroid",
#     "area",
# ]

# ---------------------------------------------------------------------------- #
#                      Find out how much overlap is needed                     #
# ---------------------------------------------------------------------------- #


def edger_bbox_to_edge(mask):
    """
    For objects touching edges, compute their distances to the touching front
    """
    mask = np.array(mask)
    h, w = mask.shape
    cleared = skimage.segmentation.clear_border(mask)
    mask[cleared > 0] = 0
    if np.all(mask == 0):
        return np.array([0])
    # return mask
    results = skimage.measure.regionprops_table(mask)
    rs = results
    return np.concatenate(
        [
            rs["bbox-2"][rs["bbox-0"] == 0],
            rs["bbox-3"][rs["bbox-1"] == 0],
            h - rs["bbox-0"][rs["bbox-2"] == h],
            w - rs["bbox-1"][rs["bbox-3"] == w],
        ]
    )


img_path = "/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/ORION-CRC04-manual-cell-type/segmentation/cellRing.ome.tif"

zimg = zarr.open(tifffile.imread(img_path, aszarr=True, level=0), mode="r")
mask = da.from_zarr(zimg)
## %%time
full_result = skimage.measure.regionprops_table(
    np.array(mask), properties=_all_mask_props
)

# ---------------------------------------------------------------------------- #
#                       compute amount of overlap needed                       #
# ---------------------------------------------------------------------------- #
block_size = 1024 * 4
nr, nc = np.ceil(np.divide(zimg.shape, block_size)).astype("int")
ri, ci = np.indices([nr, nc]).reshape(2, -1)

rs = block_size * ri
cs = block_size * ci
re = block_size * (ri + 1)
ce = block_size * (ci + 1)


def wrap_edger(rrs, rre, ccs, cce):
    zimg = zarr.open(tifffile.imread(img_path, aszarr=True, level=0), mode="r")
    mm = zimg[rrs:rre, ccs:cce]
    return edger_bbox_to_edge(mm)


## %%time
edge_size_gen = joblib.Parallel(
    verbose=0, backend="loky", n_jobs=4, return_as="generator"
)(
    joblib.delayed(wrap_edger)(rrs, rre, ccs, cce)
    for rrs, rre, ccs, cce in zip(*np.array([rs, re, cs, ce]))
)
edge_size = [rr for rr in tqdm.tqdm(edge_size_gen, total=nr * nc)]

es = np.max([np.max(e) for e in edge_size])
es = 16 * np.ceil(es / 16).astype("int")

# ---------------------------------------------------------------------------- #
#                      joblib for process parallelization                      #
# ---------------------------------------------------------------------------- #
rechunked = mask.rechunk(1024 * 4)
rr, cc = np.indices(rechunked.numblocks)
rsize, csize = rechunked.chunksize

# es = 128
rs = np.clip(rr * rsize - es, 0, mask.shape[0])
re = np.clip((rr + 1) * rsize + es, 0, mask.shape[0])
cs = np.clip(cc * csize - es, 0, None)
ce = np.clip((cc + 1) * csize + es, None, mask.shape[1])


def wrap(rrs, rre, ccs, cce):
    zimg = zarr.open(tifffile.imread(img_path, aszarr=True, level=0), mode="r")
    mm = zimg[rrs:rre, ccs:cce]
    result = skimage.measure.regionprops_table(mm, properties=_all_mask_props)
    result["offset_row"] = np.ones(len(result["label"]), dtype=int) * rrs
    result["offset_col"] = np.ones(len(result["label"]), dtype=int) * ccs
    return result


## %%time
_block_results = joblib.Parallel(
    verbose=0, backend="loky", n_jobs=4, return_as="generator"
)(
    joblib.delayed(wrap)(rrs, rre, ccs, cce)
    for rrs, rre, ccs, cce in zip(*np.array([rs, re, cs, ce]).reshape(4, -1))
)
block_results = [
    rr for rr in tqdm.tqdm(_block_results, total=np.multiply(*rechunked.numblocks))
]


import pandas as pd

df_block = (
    pd.concat([pd.DataFrame(r) for r in block_results])
    .sort_values(["label", "area"])
    .drop_duplicates(keep="last", subset="label")
    .set_index("label")
)

df = pd.DataFrame(full_result).set_index("label")
# compare result
np.sum(df_block.area != df.area)


# ---------------------------------------------------------------------------- #
#                       quantify channel intensity props                       #
# ---------------------------------------------------------------------------- #
orion_img_path = "/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/ORION-CRC04-manual-cell-type/image/P37_S32_A24_C59kX_E15@20220106_014630_553652-zlib-viz-300.ome.tiff"


def quartiles(regionmask, intensity):
    return np.percentile(intensity[regionmask], q=(25, 50, 75))


def wrap_int(rrs, rre, ccs, cce, channel=0):
    zimg = zarr.open(tifffile.imread(img_path, aszarr=True, level=0), mode="r")
    _img = zarr.open(tifffile.imread(orion_img_path, aszarr=True, level=0), mode="r")
    mm = zimg[rrs:rre, ccs:cce]
    img = _img[channel, rrs:rre, ccs:cce]
    result = skimage.measure.regionprops_table(
        mm,
        intensity_image=img,
        properties=("label", "area", "intensity_mean"),
        extra_properties=(quartiles,),
    )
    return result


rechunked = mask.rechunk(1024 * 4)
rr, cc = np.indices(rechunked.numblocks)
rsize, csize = rechunked.chunksize

es = 128
rs = np.clip(rr * rsize - es, 0, mask.shape[0])
re = np.clip((rr + 1) * rsize + es, 0, mask.shape[0])
cs = np.clip(cc * csize - es, 0, None)
ce = np.clip((cc + 1) * csize + es, None, mask.shape[1])


## %%time
block_results_int = joblib.Parallel(
    verbose=1, backend="loky", n_jobs=6, return_as="list"
)(
    joblib.delayed(wrap_int)(rrs, rre, ccs, cce)
    for rrs, rre, ccs, cce in zip(*np.array([rs, re, cs, ce]).reshape(4, -1))
)

df_block_int = (
    pd.concat([pd.DataFrame(r) for r in block_results_int])
    .sort_values(["label", "area"])
    .drop_duplicates(keep="last", subset="label")
    .set_index("label")
)
