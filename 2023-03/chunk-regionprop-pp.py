import concurrent.futures
import datetime
import time

import numpy as np
import pandas as pd
import skimage.measure
import skimage.segmentation
import tifffile
import tqdm
import zarr

_all_mask_props = [
    "label",
    "centroid",
    "area",
]
img_path = "/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/ORION-CRC04-manual-cell-type/segmentation/cellRing.ome.tif"

zimg = zarr.open(tifffile.imread(img_path, aszarr=True, level=0), mode="r")

block_size = 1024 * 4
nr, nc = np.ceil(np.divide(zimg.shape, block_size)).astype("int")
ri, ci = np.indices([nr, nc]).reshape(2, -1)

es = 128

rs = np.clip(ri * block_size - es, 0, zimg.shape[0])
re = np.clip((ri + 1) * block_size + es, 0, zimg.shape[0])
cs = np.clip(ci * block_size - es, 0, None)
ce = np.clip((ci + 1) * block_size + es, None, zimg.shape[1])


def wrap_processes(slice_coords):
    zimg = zarr.open(tifffile.imread(img_path, aszarr=True, level=0), mode="r")
    rrs, rre, ccs, cce = slice_coords
    mm = zimg[rrs:rre, ccs:cce]
    result = skimage.measure.regionprops_table(mm, properties=_all_mask_props)
    result["offset_row"] = np.ones(len(result["label"]), dtype=int) * rrs
    result["offset_col"] = np.ones(len(result["label"]), dtype=int) * ccs
    return result


def wrap_threads(slice_coords):
    rrs, rre, ccs, cce = slice_coords
    mm = zimg[rrs:rre, ccs:cce]
    result = skimage.measure.regionprops_table(mm, properties=_all_mask_props)
    result["offset_row"] = np.ones(len(result["label"]), dtype=int) * rrs
    result["offset_col"] = np.ones(len(result["label"]), dtype=int) * ccs
    return result


def main():
    mask = zimg[:]
    start_time = int(time.perf_counter())
    full_result = skimage.measure.regionprops_table(mask, properties=_all_mask_props)
    end_time = int(time.perf_counter())
    print("elapsed", datetime.timedelta(seconds=end_time - start_time))

    start_time = int(time.perf_counter())
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        block_results = [
            rrr
            for rrr in tqdm.tqdm(
                executor.map(
                    wrap_processes, np.array([rs, re, cs, ce]).reshape(4, -1).T
                ),
                total=nr * nc,
            )
        ]
    end_time = int(time.perf_counter())
    print("elapsed (process)", datetime.timedelta(seconds=end_time - start_time))

    start_time = int(time.perf_counter())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        block_results_2 = [
            rrr
            for rrr in tqdm.tqdm(
                executor.map(wrap_threads, np.array([rs, re, cs, ce]).reshape(4, -1).T),
                total=nr * nc,
            )
        ]
    end_time = int(time.perf_counter())
    print("elapsed (thread)", datetime.timedelta(seconds=end_time - start_time))

    df_block = (
        pd.concat([pd.DataFrame(r) for r in block_results])
        .sort_values(["label", "area"])
        .drop_duplicates(keep="last", subset="label")
        .set_index("label")
    )

    df = pd.DataFrame(full_result).set_index("label")
    # compare result
    print(f"Num of differences: {np.sum(df_block.area != df.area)}")


if __name__ == "__main__":
    main()
