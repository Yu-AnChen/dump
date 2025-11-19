import argparse

import matplotlib.cm
import matplotlib.pyplot as plt
import napari
import napari.utils
import numpy as np
import ome_types
import palom
import pandas as pd
import scipy.ndimage as ndi
import skimage.morphology
import tqdm


def mask_to_contour(mask):
    struct_elem_d = skimage.morphology.disk(1).astype("bool")
    struct_elem_e = skimage.morphology.disk(2).astype("bool")
    return np.where(
        ndi.grey_dilation(mask, footprint=struct_elem_d)
        == ndi.grey_erosion(mask, footprint=struct_elem_e),
        0,
        mask,
    )


def mapping_indexer(df, column_name=None, missing_value=0, bg_value=0):
    if column_name is None:
        assert df.shape[1] == 1
        column_name = df.columns[0]
    indexer = np.full(df.index.max() + 100, fill_value=missing_value, dtype="uint16")
    indexer[df.index.values] = df[column_name].values
    indexer[0] = bg_value
    return indexer


def map_mask_labels(mask_pyramid, df_mapper):
    for ll in mask_pyramid:
        assert ll.ndim == 2

    def recolor(mask, indexer):
        return indexer[mask].astype(indexer.dtype)

    mapped_masks = {}
    for kk in tqdm.tqdm(df_mapper.columns):
        idxr = mapping_indexer(df_mapper[[kk]], column_name=kk)
        mapped_masks[kk] = [
            ll.map_blocks(mask_to_contour, dtype=mask_pyramid[0].dtype).map_blocks(
                recolor, indexer=idxr, dtype="uint16"
            )
            for ll in mask_pyramid
        ]

    return mapped_masks


img_paths = r"""
//research.files.med.harvard.edu/hits/lsp-analysis/cycif-production/110-BRCA-Mutant-Ovarian-Precursors/ORION_FISH_FT_p53_Tanjina_81925/FT_iSTIC/LSP18303/fish-spotiflow/qc/LSP18303-spots-channel_22_24-qc.ome.tif
//research.files.med.harvard.edu/hits/lsp-analysis/cycif-production/110-BRCA-Mutant-Ovarian-Precursors/ORION_FISH_FT_p53_Tanjina_81925/FT_iSTIC/LSP18315/fish-spotiflow/qc/LSP18315-spots-channel_22_24-qc.ome.tif
""".strip().split("\n")

mask_paths = r"""
//research.files.med.harvard.edu/hits/lsp-analysis/cycif-production/110-BRCA-Mutant-Ovarian-Precursors/ORION_FISH_FT_p53_Tanjina_81925/FT_iSTIC/LSP18303/segmentation/mccellpose-LSP18303/cellpose-nucleus.ome.tif
//research.files.med.harvard.edu/hits/lsp-analysis/cycif-production/110-BRCA-Mutant-Ovarian-Precursors/ORION_FISH_FT_p53_Tanjina_81925/FT_iSTIC/LSP18315/segmentation/cellpose-nucleus.ome.tif
""".strip().split("\n")

counts_paths = r"""
//research.files.med.harvard.edu/hits/lsp-analysis/cycif-production/110-BRCA-Mutant-Ovarian-Precursors/ORION_FISH_FT_p53_Tanjina_81925/FT_iSTIC/LSP18303/fish-spotiflow/counts/cellpose-nucleus.parquet
//research.files.med.harvard.edu/hits/lsp-analysis/cycif-production/110-BRCA-Mutant-Ovarian-Precursors/ORION_FISH_FT_p53_Tanjina_81925/FT_iSTIC/LSP18315/fish-spotiflow/counts/cellpose-nucleus.parquet
""".strip().split("\n")

out_paths = r"""
//research.files.med.harvard.edu/hits/lsp-analysis/cycif-production/110-BRCA-Mutant-Ovarian-Precursors/ORION_FISH_FT_p53_Tanjina_81925/FT_iSTIC/LSP18303/fish-spotiflow/qc/LSP18303-spots-channel_22_24-qc-single-cell.ome.tif
//research.files.med.harvard.edu/hits/lsp-analysis/cycif-production/110-BRCA-Mutant-Ovarian-Precursors/ORION_FISH_FT_p53_Tanjina_81925/FT_iSTIC/LSP18315/fish-spotiflow/qc/LSP18315-spots-channel_22_24-qc-single-cell.ome.tif
""".strip().split("\n")

for img_path, mask_path, counts_path, out_path in zip(
    img_paths[:], mask_paths[:], counts_paths[:], out_paths[:]
):
    reader = palom.reader.OmePyramidReader(img_path)
    mask_reader = palom.reader.OmePyramidReader(mask_path)
    counts = pd.read_parquet(counts_path, engine="pyarrow")

    # manually adding 1 because background is 0
    counts_render = counts.filter(like="count").round().astype("uint16") + 1
    # counts_render.loc[:, :] = np.where(counts_render.values == 0, 200, counts_render.values)

    print(f"re-coloring mask based on {counts_render.columns.tolist()}")
    counts_imgs = map_mask_labels(
        [pp.squeeze() for pp in mask_reader.pyramid],
        counts_render,
    )

    mosaics = [reader.pyramid[0]]
    mosaics.extend([counts_imgs[nn][0] for nn in counts_render.columns.tolist()])
    channel_names = [
        [cc.name for cc in ome_types.from_tiff(reader.path).images[0].pixels.channels]
    ]
    channel_names.extend(counts_render.columns.tolist())
    palom.pyramid.write_pyramid(
        mosaics,
        output_path=out_path,
        pixel_size=reader.pixel_size,
        channel_names=channel_names,
        is_mask=True,
    )

