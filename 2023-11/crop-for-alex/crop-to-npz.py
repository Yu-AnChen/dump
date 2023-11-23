import pathlib

import palom
import numpy as np
import dask.diagnostics
import fire


def crop(img_path, mask_path, out_dir, ul_x, ul_y, size):
    img_path = pathlib.Path(img_path)
    mask_path = pathlib.Path(mask_path)
    out_dir = pathlib.Path(out_dir)

    out_dir.mkdir(exist_ok=True, parents=True)

    r1 = palom.reader.OmePyramidReader(img_path)
    r2 = palom.reader.OmePyramidReader(mask_path)

    assert r1.pyramid[0][0].shape == r2.pyramid[0][0].shape

    row_s, col_s = int(ul_y), int(ul_x)
    row_e, col_e = row_s + int(size), col_s + int(size)

    assert row_s >= 0
    assert col_s >= 0
    assert row_e < r1.pyramid[0].shape[1]
    assert col_e < r1.pyramid[0].shape[2]

    img = r1.pyramid[0][:, row_s:row_e, col_s:col_e]
    img = np.moveaxis(img, 0, 2)
    mask = r2.pyramid[0][0, row_s:row_e, col_s:col_e]


    out_img_name = f"image-x_{ul_x}-y_{ul_y}.npz"
    out_mask_name = f"mask-x_{ul_x}-y_{ul_y}.npz"


    with dask.diagnostics.ProgressBar():
        np.savez(
            out_dir / out_img_name,
            data=img.compute()
        )
        np.savez(
            out_dir / out_mask_name,
            data=mask.compute()
        )


if __name__ == '__main__':
    fire.Fire(crop)

    r'''
    img_path = r"Z:\JL503_JERRY\TNP_2020\Addtional_CRC_Ometiff\CRC12.ome.tif"
    mask_path = r"Z:\JL503_JERRY\TNP_2020\Addtional_CRC_MCMICRO\mcmicro\CRC12_n\segmentation\unmicst-TNPCRC_11\cellRingMask.tif"
    out_dir = r"Y:\sorger\data\computation\Yu-An\YC-20231122-Alex_recyze\output"

    UL_X = 9000
    UL_Y = 4000
    SIZE = 5000

    crop(img_path, mask_path, out_dir, UL_X, UL_Y, SIZE)

    Example
    python crop-to-npz.py Z:\JL503_JERRY\TNP_2020\Addtional_CRC_Ometiff\CRC12.ome.tif Z:\JL503_JERRY\TNP_2020\Addtional_CRC_MCMICRO\mcmicro\CRC12_n\segmentation\unmicst-TNPCRC_11\cellRingMask.tif Y:\sorger\data\computation\Yu-An\YC-20231122-Alex_recyze\output\CRC12 9000 4000 5000
    '''
