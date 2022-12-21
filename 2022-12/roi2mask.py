import omero_roi
import pandas as pd
import numpy as np
import cv2


def roi_path_to_mask(
    roi_path,
    mask_shape=None,
    fill_value=None
):
    _roi = pd.read_csv(roi_path)
    roi = _roi.apply(omero_roi.add_mpatch, axis=1)

    if mask_shape is None:
        mask_shape = [
            roi.parsed_points.apply(
                lambda x: np.array(x).max(axis=0)[i]
            ).max()
            for i in range(2)
        ][::-1]
    h, w = mask_shape
    
    if fill_value is None:
        fill_value = 1
    fill_value = int(fill_value)

    mask = np.zeros((int(h), int(w)), np.uint8)
    cv2.fillPoly(
        mask,
        roi.parsed_points.apply(lambda x: np.round(x).astype(int)),
        fill_value
    )
    return mask


def save_as_tif(filepath, mask):
    import tifffile
    import math
    
    def num_levels(base_shape):
        max_pyramid_img_size = 1024
        downscale_factor = 2
        factor = max(base_shape) / max_pyramid_img_size
        return math.ceil(math.log(factor, downscale_factor)) + 1

    subresolutions = num_levels(mask.shape) - 1
    with tifffile.TiffWriter(filepath, bigtiff=True) as tif:
        metadata={
            # this is loose, 2-bit contains 0, 1, 2, 3
            'SignificantBits': 2
        }
        options = dict(
            tile=(1024, 1024),
            compression='lzw'
        )
        tif.write(
            mask,
            subifds=subresolutions,
            metadata=metadata,
            **options
        )
        for level in range(subresolutions):
            mag = 2**(level + 1)
            tif.write(
                mask[::mag, ::mag],
                subfiletype=1,
                metadata=metadata,
                **options
            )


def run_all():
    import pathlib
    roi_paths = sorted(pathlib.Path(
        r'Y:\sorger\data\computation\Yu-An\YC-20221220-Tuulia_omero_roi\rois'
    ).glob('*-rois.csv'))
    img_shapes = pd.read_csv(
        r'Y:\sorger\data\computation\Yu-An\YC-20221220-Tuulia_omero_roi\img_shapes.csv',
        index_col='id'
    )
    img_ids = [
        int(pp.stem.split('-')[-2])
        for pp in roi_paths
    ]
    out_dir = pathlib.Path(
        r'Y:\sorger\data\computation\Yu-An\YC-20221220-Tuulia_omero_roi\masks'
    )
    for pp, id in zip(roi_paths, img_ids):
        height, width = img_shapes.loc[id]
        print('Processing:', pp.name)
        mask = roi_path_to_mask(pp, mask_shape=(height, width))
        out_name = f"{pp.stem.split('.ome')[0]}-tumor_mask.ome.tif"
        save_as_tif(out_dir / out_name, mask)
