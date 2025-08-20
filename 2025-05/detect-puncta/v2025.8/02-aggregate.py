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
    bin_size=0.4,
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
    df_all.filter(like="count_").hist(
        bins=np.linspace(-0.5, 19.5, 21), log=True, sharey=True, sharex=True
    )

    fig = plt.gcf()
    fig.suptitle("Distribution of spot counts per cell")
    for aa in fig.axes:
        aa.set_xticks(range(21))
    fig.tight_layout()
    save_all_figs(format="jpg", out_dir=qc_dir)

    df_all.to_parquet(out_path)


process_one_slide(
    r"T:\ORION_FISH_FT_fullpanel\LSP33836_P110\fish",
    "Green-CEP1,Orange-MDM4".split(","),
    r"T:\ORION_FISH_FT_fullpanel\LSP33836_P110\segmentation\unmicst-LSP33836_P110\nucleiRing.ome.tif",
    r"T:\ORION_FISH_FT_fullpanel\LSP33836_P110\fish\counts\nucleiRing.parquet",
    r"T:\ORION_FISH_FT_fullpanel\LSP33836_P110\fish\counts\qc",
)

process_one_slide(
    r"T:\ORION_FISH_FT_fullpanel\LSP17773\fish",
    "Green-CEP1,Orange-MDM4".split(","),
    r"T:\ORION_FISH_FT_fullpanel\LSP17773\segmentation\unmicst-LSP17773\nucleiRing.ome.tif",
    r"T:\ORION_FISH_FT_fullpanel\LSP17773\fish\counts\nucleiRing.parquet",
    r"T:\ORION_FISH_FT_fullpanel\LSP17773\fish\counts\qc",
)
