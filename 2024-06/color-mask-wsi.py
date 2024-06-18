import napari
import palom
import pandas as pd
import numpy as np


def mapping_indexer(df, column_name=None, value_for_missing_key=0):
    if column_name is None:
        assert df.shape[1] == 1
        column_name = df.columns[0]
    indexer = np.full(df.index.max() + 1, fill_value=value_for_missing_key)
    indexer[df.index.values] = df[column_name].values
    return indexer


def map_mask_labels(mask_pyramid, df_mapper):
    
    def recolor(mask, indexer):
        return indexer[mask].astype(indexer.dtype)

    mapped_masks = {}
    for kk in df_mapper:
        idxr = mapping_indexer(df_mapper[[kk]], column_name=kk).astype(
            df_mapper[kk].dtype
        )
        mapped_masks[kk] = [
            ll.map_blocks(recolor, indexer=idxr, dtype=idxr.dtype)
            for ll in mask_pyramid
        ]

    return mapped_masks


df = pd.read_csv(
    r"U:\YC-20240214-orion_c4\CRC09\P37_S37_cellRingMask.csv.zip", index_col="CellID"
)
mask = palom.reader.OmePyramidReader(
    r"U:\YC-20240214-orion_c4\CRC09\P37_S37_nucleiRingMask.ome.tif"
)

markers = df.loc[:, "Hoechst":"SMA"].columns
marker_masks = map_mask_labels(
    mask.pyramid, df.loc[:, "Hoechst":"SMA"].astype("uint16")
)


reader = palom.reader.OmePyramidReader(
    r"\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round1\P37_S37_A24_C59kX_E15@20220108_012113_953544-zlib.ome.tiff"
)

v = napari.Viewer()
v.add_image(
    reader.pyramid,
    name=markers,
    visible=False,
    contrast_limits=(100, 10000),
    channel_axis=0,
)
v.add_image(marker_masks["CD45"], colormap="cividis")
