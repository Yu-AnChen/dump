import argparse
import sys

import matplotlib.pyplot as plt
import napari
import napari.utils
import numpy as np
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


def mapping_indexer(df, column_name=None, value_for_missing_key=0):
    if column_name is None:
        assert df.shape[1] == 1
        column_name = df.columns[0]
    indexer = np.full(
        df.index.max() + 1, fill_value=value_for_missing_key, dtype="float16"
    )
    indexer[df.index.values] = df[column_name].values
    indexer[0] = np.nan
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
                recolor, indexer=idxr, dtype="float16"
            )
            for ll in mask_pyramid
        ]
        mapped_masks[f"{kk}-m"] = [
            ll.map_blocks(recolor, indexer=idxr, dtype="float16") for ll in mask_pyramid
        ]

    return mapped_masks


def main():
    from qtpy.QtWidgets import QApplication

    parser = argparse.ArgumentParser(
        description="Visual QC for puncta detection: overlay spot counts per cell."
    )
    parser.add_argument(
        "--img-path",
        type=str,
        required=True,
        help="Path to the multi-channel .ome.tif image",
    )
    parser.add_argument(
        "--mask-path", type=str, required=True, help="Path to the labeled mask .ome.tif"
    )
    parser.add_argument(
        "--counts-path",
        type=str,
        required=True,
        help="Path to the per-cell counts .parquet file",
    )
    parser.add_argument(
        "--channel-names",
        type=str,
        required=True,
        help="Comma separated list of all channel names in the image",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="tab10",
        help="Name of the colormap for count-per-cell rendering",
    )
    args = parser.parse_args()

    reader = palom.reader.OmePyramidReader(args.img_path)
    mask_reader = palom.reader.OmePyramidReader(args.mask_path)
    counts = pd.read_parquet(args.counts_path, engine="pyarrow")
    mpl_cmap = plt.get_cmap(args.colormap)

    channel_names = args.channel_names.split(",")
    if len(channel_names) != len(reader.pyramid[0]):
        print(
            f"WARNING: {len(reader.pyramid[0])} channel(s) in image but {len(channel_names)} channel name(s) provided"
        )
        channel_names = None

    v = napari.Viewer()
    app = QApplication.instance()
    app.lastWindowClosed.connect(lambda: sys.exit(0))

    v.scale_bar.visible = True
    v.scale_bar.unit = "µm"
    scale = (reader.pixel_size, reader.pixel_size)

    v.add_image(
        reader.pyramid,
        channel_axis=0,
        visible=False,
        name=channel_names,
        scale=scale,
    )

    colors = mpl_cmap(np.linspace(0, 1, 10))
    cmap = napari.utils.Colormap(
        colors=((0.1, 0.1, 0.1, 1),) + tuple(colors),
        high_color=(1, 1, 1, 1),
        # BUG nan_color does not seem to work as of napari v0.6.1
        nan_color=np.zeros(4),
        low_color=(0, 0, 0, 0),
        interpolation="zero",
    )

    print(f"re-coloring mask based on {counts.filter(like='count').columns.tolist()}")
    counts_imgs = map_mask_labels(
        [pp.squeeze() for pp in mask_reader.pyramid],
        counts.filter(like="count").round().astype("int"),
    )

    for kk, vv in counts_imgs.items():
        v.add_image(
            vv,
            name=kk,
            visible=False,
            colormap=cmap,
            contrast_limits=(-1, 11),
            scale=scale,
        )

    plt.figure()
    plt.imshow(
        np.reshape(
            np.vstack([[0.1, 0.1, 0.1, 1], colors, [1, 1, 1, 1]]),
            (-1, 1, 4),
        )
    )
    plt.gca().set(xticks=[], yticks=range(12), yticklabels=list(range(11)) + ["11+"])
    plt.gca().invert_yaxis()
    plt.title("Colormap for counts")
    plt.show()

    napari.run()


if __name__ == "__main__":
    main()


"""
python \\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION-FISH-72525_Tanjina\October2025-Batch2\scripts-fish-v2025-11\04-viz-for-parquet.py
  --img-path "\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION-FISH-72525_Tanjina\October2025-Batch2\LSP18304\fish-spotiflow\qc\LSP18304-spots-channel_22_23_24_27-qc.ome.tif"
  --mask-path "\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION-FISH-72525_Tanjina\October2025-Batch2\LSP18304\segmentation\LSP18304-cellpose-nucleus.ome.tif"
  --counts-path "\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION-FISH-72525_Tanjina\October2025-Batch2\LSP18304\fish-spotiflow\counts\cellpose-nucleus.parquet"
  --channel-names "DNA,CEP8,CCNE1,MYC,CEP19,CEP8_peak,CCNE1_peak,MYC_peak,CEP19_peak"


python "/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20250520-Tanjina-puncta-detection/tanjina-fish-presentation/viz.py" --img-path "/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20250520-Tanjina-puncta-detection/tanjina-fish-presentation/LSP18304-spots-channel_22_23_24_27-qc.ome.tif" --mask-path "/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20250520-Tanjina-puncta-detection/tanjina-fish-presentation/LSP18304-cellpose-nucleus.ome.tif" --counts-path "/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20250520-Tanjina-puncta-detection/tanjina-fish-presentation/counts/cellpose-nucleus.parquet" --channel-names "DNA,CEP8,CCNE1,MYC,CEP19,CEP8_peak,CCNE1_peak,MYC_peak,CEP19_peak"
"""
