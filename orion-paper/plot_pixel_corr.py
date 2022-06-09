import numpy as np
import pathlib
import pandas as pd
import seaborn as sns
import tifffile
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches


def set_matplotlib_font():
    font_families = matplotlib.rcParams['font.sans-serif']
    if font_families[0] != 'Arial':
        font_families.insert(0, 'Arial')
    matplotlib.rcParams['pdf.fonttype'] = 42


def plot_corr(
    corr, name='', 
    grouping=None, grouping_colors=None,
    highlighted_fluors=None,
    heatmap_kwargs=None
):
    set_matplotlib_font()

    plt.figure(figsize=(6.5, 7))
    plt.gcf().patch.set_alpha(0)
    _heatmap_kwargs = dict(
        cmap='coolwarm',
        center=0,
        linewidths=1,
        square=True,
        cbar=False
    )
    if heatmap_kwargs is None:
        heatmap_kwargs = {}
    heatmap_kwargs = {**_heatmap_kwargs, **heatmap_kwargs}
    ax = sns.heatmap(corr, **heatmap_kwargs)
    ax.set_xlim(np.array([-.2, .2])+ax.get_xlim())
    ax.set_ylim(np.array([.2, -.2])+ax.get_ylim())
    plt.suptitle(name)

    if grouping is not None:
        orion_ori = np.cumsum([0] + grouping)

        ax = plt.gca()
        for o, w, c in zip(orion_ori, grouping, grouping_colors):
            ax.add_patch(mpatches.Rectangle(
                (o, o), w, w, edgecolor=c,        
                fill=False, lw=5
            ))
    if highlighted_fluors is not None:
        for o in highlighted_fluors:
            ax.add_patch(mpatches.Rectangle(
                (o, o), 1, 1, edgecolor='black',        
                fill=False, lw=3
            ))

paths = [
    r"Z:\RareCyte-S3\P37_CRCstudy_SpectrallySeperatedControls\P37_S19_01_A24",
    r"Z:\RareCyte-S3\P37_CRCstudy_SpectrallySeperatedControls\P37_S20_01_A24",
    r"Z:\RareCyte-S3\P37_CRCstudy_SpectrallySeperatedControls\P37_S21_01_A24",
    r"Z:\RareCyte-S3\P37_CRCstudy_SpectrallySeperatedControls\P37_S22_01_A24",
    r"Z:\RareCyte-S3\P37_CRCstudy_SpectrallySeperatedControls\P37_S23_01_A24",
    r"Z:\RareCyte-S3\P37_CRCstudy_SpectrallySeperatedControls\P37_S24_01_A24",
]

num_channels = 19

import zarr
import dask.array as da

def read_as_da(path, num_channels):
    img = tifffile.imread(path, aszarr=True)
    zimg = zarr.open(img)
    da_img = da.from_zarr(zimg)
    return da_img.reshape(-1, 19, *da_img.shape[1:])


def heatmap_ticklabel(markers, highlighted=None):
    if highlighted is None:
        highlighted = np.arange(len(markers))
    labels = np.arange(1, len(markers)+1).astype(str)
    labels[highlighted] = np.asarray(markers)[highlighted]
    return labels


raw_paths = [
    sorted(pathlib.Path(p).glob('C00_E14*.pysed.ome.tif'))[0]
    for p in paths
]

sed_paths = [
    sorted(pathlib.Path(p).glob('C59kX_E14*.pysed.ome.tif'))[0]
    for p in paths
]

fluors = [
    [0, 6, 10, 14, 18],
    [0, 2, 7, 11, 15],
    [0, 3, 8, 12, 16],
    [0, 4, 9, 13, 17],
    [],
    list(range(19))
]

colors_hex = [
    '#9933ff',
    '#00e6e6',
    '#00e600',
    '#ffff00',
    '#ffbf00',
    '#ff4000',
    '#cc0000'
]
colors_rgb = [mcolors.to_rgb(c) for c in colors_hex]

orion_ex_group = [1, 1, 2, 2, 5, 4, 4]

markers = [
    'Hoechst',
    'AF1',
    'CD31',
    'CD45',
    'CD68',
    '(Empty)',
    'CD4',
    'FOXP3',
    'CD8a',
    'CD45RO',
    'CD20',
    'PD-L1',
    'CD3e',
    'CD163',
    'E-cadherin',
    'PD-1',
    'Ki-67',
    'Pan-CK',
    'SMA'
]


# use all the tiles, do center crop to avoid overlap
import dask.diagnostics

out_dir = pathlib.Path(
    r'Z:\RareCyte-S3\YC-analysis\P37_CRCstudy_SpectrallySeperatedControls\20220609'
)
for fls, r, s in zip(fluors[:], raw_paths[:], sed_paths[:]):
    for t, n in zip([r, s], ['Raw', 'Subtracted']):
        tiles = read_as_da(t, None)
        name = t.name.split('--')[-1].replace('_01_A24.pysed.ome.tif', '')
        ticklabels = heatmap_ticklabel(markers, fls)
        with dask.diagnostics.ProgressBar():
            plot_corr(
                np.mean(
                    [
                        np.corrcoef(t[:, 200:-200, 200:-200].reshape(19, -1))
                        for t in tiles
                    ],
                    axis=0
                ),
                f'{n} - {name}',
                grouping=orion_ex_group,
                grouping_colors=colors_rgb,
                highlighted_fluors=fls,
                heatmap_kwargs=dict(
                    xticklabels=ticklabels,
                    yticklabels=ticklabels
                )
            )
            plt.savefig(out_dir / f'{n}-{name}.pdf')