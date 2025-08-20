import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import napari
import napari.utils
import pandas as pd
import scipy.ndimage as ndi
import skimage.morphology
import palom
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
            mask_to_contour(ll).astype(mask_pyramid[0].dtype)
            if hasattr(ll, "map_blocks")
            else mask_to_contour(ll)
            for ll in mask_pyramid
        ]
        mapped_masks[kk] = [recolor(ll, idxr) for ll in mapped_masks[kk]]
    return mapped_masks


def main():
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
        help="Path to the per-cell counts .csv file",
    )
    parser.add_argument(
        "--channel-names",
        type=str,
        required=True,
        help="Comma separated list of all channel names in the image",
    )
    args = parser.parse_args()

    reader = palom.reader.OmePyramidReader(args.img_path)
    mask_reader = palom.reader.OmePyramidReader(args.mask_path)
    counts = pd.read_csv(args.counts_path, engine="pyarrow", index_col="CellID")

    channel_names = args.channel_names.split(",")
    if len(channel_names) != len(reader.pyramid[0]):
        print(
            f"WARNING: {len(reader.pyramid[0])} channel(s) in image but {len(channel_names)} channel name(s) provided"
        )
        channel_names = None

    v = napari.Viewer()
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

    cmap = napari.utils.Colormap(
        colors=((0.1, 0.1, 0.1),) + matplotlib.cm.tab10.colors,
        high_color=(1, 1, 1, 1),
        # BUG nan_color does not seem to work as of napari v0.6.1
        nan_color=np.zeros(4),
        low_color=(0, 0, 0, 0),
        interpolation="zero",
    )

    print(f"re-coloring mask based on {counts.filter(like='count').columns.tolist()}")
    counts_imgs = map_mask_labels(
        [pp.squeeze() for pp in mask_reader.pyramid], counts.filter(like="count")
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
            np.vstack([[0, 0, 0, 1], matplotlib.cm.tab10(np.arange(10)), [1, 1, 1, 1]]),
            (-1, 1, 4),
        )
    )
    plt.gca().set(xticks=[], yticks=range(12), yticklabels=list(range(11)) + ["11+"])
    plt.title("Colormap for counts")

    napari.run()


if __name__ == "__main__":
    main()


"""
python visual_check_cli.py
  --img-path "T:\Orion_FISH_test1\Tanjina_ORION_FISH_test_HGSOC\LSP14728\fish\qc\LSP14728_001_A107_P54N_HMS_TK-spots-channel_22_24_27_28_29-qc.ome.tif"
  --mask-path "T:\Orion_FISH_test1\Tanjina_ORION_FISH_test_HGSOC\LSP14728\segmentation\LSP14728_001_A107_P54N_HMS_TK\nucleiRing.ome.tif"
  --counts-path "T:\Orion_FISH_test1\Tanjina_ORION_FISH_test_HGSOC\LSP14728\fish\counts\LSP14728-quantification-fish.csv"
  --channel-names "DNA,CEP8,CCNE1,MYC,CEP19_602,CEP19_624,peak_CEP8,peak_CCNE1,peak_MYC,peak_CEP19_602,peak_CEP19_624"
"""
