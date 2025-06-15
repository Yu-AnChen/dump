import functools
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.feature
import skimage.filters
import sklearn.linear_model
import tifffile
import tqdm


def peak_intensity_to_count(
    log_intensities,
    intensity_p0=0,
    intensity_p1=99.9,
    bin_size=0.25,
    gaussian_sigma=1,
    first_n_peaks=5,
    plot=False,
):
    log_intensities = np.asarray(log_intensities)
    p0, p1 = np.percentile(log_intensities, [intensity_p0, intensity_p1])
    counts, edges = np.histogram(
        log_intensities,
        bins=np.linspace(p0, p1, np.floor((p1 - p0) / bin_size).astype("int") + 1),
    )

    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = skimage.filters.gaussian(counts, gaussian_sigma)
    peak_idx = skimage.feature.peak_local_max(
        smoothed, num_peaks=first_n_peaks
    ).flatten()
    peak_locations = centers[peak_idx]

    lr = sklearn.linear_model.LinearRegression()
    lr.fit(
        peak_locations.reshape(-1, 1), np.arange(1, 1 + first_n_peaks).reshape(-1, 1)
    )

    foci_counts = lr.predict(log_intensities.reshape(-1, 1))

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axs = plt.subplots(1, 2)
        sns.histplot(log_intensities, bins=edges, ax=axs[0])
        plt.sca(axs[0])
        plt.vlines(
            peak_locations, 0, counts.max(), colors="k", linestyles="dashed", alpha=0.5
        )
        axs[0].set_xlabel(r"$\sum\log_{10}\left( \text{Peak intensity} \right)$")

        sns.regplot(x=peak_locations, y=np.arange(1, 1 + first_n_peaks), ax=axs[1])
        axs[1].set_xlabel(r"$\sum\log_{10}\left( \text{Peak intensity} \right)$")
        axs[1].set_ylabel("Number of dots per cell")
        axs[1].set(
            xlim=(0, peak_locations.max() + peak_locations.min()),
            ylim=(0, first_n_peaks + 2),
        )

    return foci_counts


def assign_spots_to_mask(peak_parquet, labeled_mask, marker_name):
    df = pd.read_parquet(peak_parquet, engine="pyarrow")
    df["mask_id"] = labeled_mask[tuple(df[["Y", "X"]].values.T)]

    df_valid = df.query("(mask_id > 0) & (channel_intensity > 0)")
    total_peak_intensity_per_cell = (
        df_valid[["channel_intensity"]]
        .transform(np.log10)
        .groupby(df_valid["mask_id"])
        .sum()
    )
    total_peak_intensity_per_cell[f"count_{marker_name}"] = peak_intensity_to_count(
        total_peak_intensity_per_cell["channel_intensity"], plot=True
    )
    plt.gcf().suptitle(marker_name)
    total_peak_intensity_per_cell.rename(
        columns={"channel_intensity": marker_name.replace("count_", "peak_int_")},
        inplace=True,
    )
    return total_peak_intensity_per_cell


def process_one_slide(
    peak_parquet_dir, marker_names, labeled_mask_path, out_path, qc_dir
):
    from palom.cli.align_he import save_all_figs, set_matplotlib_font

    set_matplotlib_font(10)

    parquets = sorted(
        pathlib.Path(peak_parquet_dir).glob("*-spots-channel_*-measurements_*.parquet")
    )
    assert len(marker_names) == len(set(marker_names)), "marker names must be unique"

    out_path = pathlib.Path(out_path)
    assert out_path.name.endswith(".parquet")
    qc_dir = pathlib.Path(qc_dir)

    out_path.parent.mkdir(exist_ok=True, parents=True)
    qc_dir.mkdir(exist_ok=True, parents=True)

    mask = tifffile.imread(labeled_mask_path)

    tlength = max([len(nn) for nn in marker_names])
    dfs = []
    for pp, nn in zip(parquets, marker_names):
        print(f"Processing marker: {nn:{tlength}} - {pp.name}")
        dfs.append(assign_spots_to_mask(pp, mask, nn))

    df_all = (
        functools.reduce(
            lambda left, right: pd.merge(
                left, right, left_index=True, right_index=True, how="outer"
            ),
            dfs,
        )
        .fillna(0)
        .round(2)
    )

    set_matplotlib_font(6)
    df_all.filter(like="count_").hist(bins=np.linspace(-0.5, 19.5, 21))

    fig = plt.gcf()
    fig.suptitle("Distribution of spot counts per cell")
    for aa in fig.axes:
        aa.set_xticks(range(21))
    fig.tight_layout()
    save_all_figs(format="jpg", out_dir=qc_dir)

    df_all.to_parquet(out_path)


process_one_slide(
    "fish",
    "CEP8,CCNE1,MYC,CEP19_602,CEP19_624".split(","),
    labeled_mask_path="segmentation/nucleiRing.ome.tif",
    out_path="fish/count-per-cell/nucleiRing.parquet",
    qc_dir="fish/count-per-cell/qc",
)

# ---------------------------------------------------------------------------- #
#                                 visual check                                 #
# ---------------------------------------------------------------------------- #
import matplotlib.cm
import napari
import napari.utils
import numpy as np
import palom
import pandas as pd
import scipy.ndimage as ndi
import skimage.morphology


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
    for kk in df_mapper:
        idxr = mapping_indexer(df_mapper[[kk]], column_name=kk)
        mapped_masks[kk] = [
            ll.map_blocks(mask_to_contour, dtype=mask_pyramid[0].dtype).map_blocks(
                recolor, indexer=idxr, dtype="float16"
            )
            for ll in mask_pyramid
        ]

    return mapped_masks


reader = palom.reader.OmePyramidReader(
    r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20250520-Tanjina-puncta-detection/fish/qc/LSP31050_P110_A107_P54N_HMS_TK-spots-channel_22_24_27_28_29-qc.ome.tif"
)
mask_reader = palom.reader.OmePyramidReader(
    r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20250520-Tanjina-puncta-detection/segmentation/nucleiRing.ome.tif"
)
counts = pd.read_parquet(
    r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20250520-Tanjina-puncta-detection/fish/count-per-cell/nucleiRing.parquet"
)


v = napari.Viewer()
v.scale_bar.visible = True
v.scale_bar.unit = "µm"

v.add_image(
    reader.pyramid,
    channel_axis=0,
    visible=False,
    name="DNA,CEP8,CCNE1,MYC,CEP19_602,CEP19_624,peak_CEP8,peak_CCNE1,peak_MYC,peak_CEP19_602,peak_CEP19_624".split(
        ","
    ),
    scale=(0.325, 0.325),
)

_max = mask_reader.pyramid[0][0].max().compute()
idxr = np.full(_max + 1, fill_value=np.nan, dtype="float16")
idxr[counts.index] = counts["count_CEP8"]

cmap = napari.utils.Colormap(
    colors=((0.1, 0.1, 0.1),) + matplotlib.cm.tab10.colors,
    high_color=(1, 1, 1, 1),
    # BUG nan_color does not seem to work as of napari v0.6.1
    nan_color=np.zeros(4),
    low_color=(0, 0, 0, 0),
    interpolation="zero",
)

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
        scale=(0.325, 0.325),
    )


plt.figure()
plt.imshow(
    np.reshape(
        np.vstack([[0, 0, 0, 1], matplotlib.cm.tab10(np.arange(10)), [1, 1, 1, 1]]),
        (-1, 1, 4),
    )
)
plt.gca().set(xticks=[], yticks=range(12), yticklabels=list(range(11)) + ["11+"])
