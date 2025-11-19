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


def assign_spots_to_mask(peak_parquet, labeled_mask, marker_name):
    df = pd.read_parquet(peak_parquet, engine="pyarrow")
    df["mask_id"] = labeled_mask[tuple(df[["Y", "X"]].values.T)]

    df_valid = df.query("(mask_id > 0) & (channel_intensity > 0)")
    gb = df_valid[["channel_intensity"]].groupby(df_valid["mask_id"])

    total_peak_intensity_per_cell = gb.sum()
    total_peak_intensity_per_cell[f"count_{marker_name}"] = gb.size()
    total_peak_intensity_per_cell.rename(
        columns={"channel_intensity": f"sum_peak_int_{marker_name}"},
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
    save_all_figs(format="pdf", out_dir=qc_dir)

    df_all.to_parquet(out_path)


process_one_slide(
    r"\\research.files.med.harvard.edu\hits\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION_FISH_FT_p53_Tanjina_81925\FT_iSTIC\LSP18303\fish-spotiflow",
    ["CEP1", "MDM4"],
    r"\\research.files.med.harvard.edu\hits\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION_FISH_FT_p53_Tanjina_81925\FT_iSTIC\LSP18303\segmentation\mccellpose-LSP18303\cellpose-nucleus.ome.tif",
    r"\\research.files.med.harvard.edu\hits\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION_FISH_FT_p53_Tanjina_81925\FT_iSTIC\LSP18303\fish-spotiflow\counts\cellpose-nucleus.parquet",
    r"\\research.files.med.harvard.edu\hits\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION_FISH_FT_p53_Tanjina_81925\FT_iSTIC\LSP18303\fish-spotiflow\counts\qc",
)

process_one_slide(
    r"\\research.files.med.harvard.edu\hits\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION_FISH_FT_p53_Tanjina_81925\FT_iSTIC\LSP18315\fish-spotiflow",
    ["CEP1", "MDM4"],
    r"\\research.files.med.harvard.edu\hits\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION_FISH_FT_p53_Tanjina_81925\FT_iSTIC\LSP18315\segmentation\cellpose-nucleus.ome.tif",
    r"\\research.files.med.harvard.edu\hits\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION_FISH_FT_p53_Tanjina_81925\FT_iSTIC\LSP18315\fish-spotiflow\counts\cellpose-nucleus.parquet",
    r"\\research.files.med.harvard.edu\hits\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION_FISH_FT_p53_Tanjina_81925\FT_iSTIC\LSP18315\fish-spotiflow\counts\qc",
)
