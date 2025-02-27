import itertools
import pathlib

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import seaborn as sns
import tqdm
from ashlar import reg, utils
from ashlar.scripts import ashlar


def set_matplotlib_font():
    font_families = matplotlib.rcParams["font.sans-serif"]
    if font_families[0] != "Arial":
        font_families.insert(0, "Arial")
    matplotlib.rcParams["pdf.fonttype"] = 42


def plot_corr(
    corr,
    name="",
    grouping=None,
    grouping_colors=None,
    highlighted_fluors=None,
    heatmap_kwargs=None,
):
    set_matplotlib_font()

    plt.figure(figsize=(6.5, 7))
    plt.gcf().patch.set_alpha(0)
    _heatmap_kwargs = dict(
        cmap="coolwarm", center=0, linewidths=1, square=True, cbar=False
    )
    if heatmap_kwargs is None:
        heatmap_kwargs = {}
    heatmap_kwargs = {**_heatmap_kwargs, **heatmap_kwargs}
    ax = sns.heatmap(corr, **heatmap_kwargs)
    ax.set_xlim(np.array([-0.2, 0.2]) + ax.get_xlim())
    ax.set_ylim(np.array([0.2, -0.2]) + ax.get_ylim())
    plt.suptitle(name)

    if grouping is not None:
        orion_ori = np.cumsum([0] + grouping)

        ax = plt.gca()
        for o, w, c in zip(orion_ori, grouping, grouping_colors):
            ax.add_patch(
                mpatches.Rectangle(
                    (0, o + 0.05), 0, w - 0.1, edgecolor=c, fill=False, lw=5
                )
            )
    if highlighted_fluors is not None:
        for o in highlighted_fluors:
            ax.add_patch(
                mpatches.Rectangle((o, o), 1, 1, edgecolor="black", fill=False, lw=3)
            )


# TILE = 32
TILE = 45


c1r = reg.BioformatsReader(
    r"X:\cycif-production\152-Kidney_Imaging_ProspectiveCasesRound2-2\LSP22828_001\Kidney_V4_2024-09_PysedOn_001398\LSP22828_001_Kidney_V4_2024-09_PysedOn_001398.pysed.ome.tif"
)
ashlar.process_axis_flip(c1r, False, True)
c1e = reg.EdgeAligner(c1r, filter_sigma=1, verbose=True)
c1e.run()

reg.plot_edge_quality(c1e, img=np.log1p(c1e.reader.thumbnail))

unmixed = np.array(
    [c1r.read(TILE, cc) for cc in tqdm.trange(c1r.metadata.num_channels)]
)
# plt.figure()
# plt.imshow(
#     np.corrcoef(unmixed.reshape(19, -1)),
#     vmin=-1,
#     vmax=1,
#     cmap="coolwarm",
# )

markers = [
    "DNA-Hoechst",
    "AF-AF1",
    "C3-FITC",
    "PODXL-Argo555L",
    "Albumin-Argo535",
    "Lambda-Argo580L",
    "Fibrinogen-Argo660L",
    "C1q-Argo572",
    "Nephrin-Argo602",
    "IgG-Argo624",
    "IgM-Argo662",
    "Synaptopodin-Argo686",
    "KIM1-Argo730",
    "Kappa-Argo760",
    "AQP1-Argo784",
    "IgA-Argo810",
    "ColIII-Argo845",
    "ColIV-Argo874",
    "SMA-Argo890",
]

ex_em = [
    "405-452",
    "445-486",
    "470-516",
    "470-560",
    "520-542",
    "520-582",
    "520-672",
    "555-574",
    "555-602",
    "555-628",
    "640-664",
    "640-688",
    "640-722",
    "730-764",
    "730-784",
    "730-814",
    "730-848",
    "730-872",
    "730-890",
]

colors_hex = [
    "#9933ff",
    "#00e6e6",
    "#00e600",
    "#ffff00",
    "#ffbf00",
    "#ff4000",
    "#cc0000",
]
colors_rgb = [mcolors.to_rgb(c) for c in colors_hex]


rcpnls = [
    r"X:\cycif-production\152-Kidney_Imaging_ProspectiveCasesRound2-2\LSP22828_001\Scan_20240918_110026_01x19x00130_405.rcpnl",
    r"X:\cycif-production\152-Kidney_Imaging_ProspectiveCasesRound2-2\LSP22828_001\Scan_20240918_110026_01x19x00130_445.rcpnl",
    r"X:\cycif-production\152-Kidney_Imaging_ProspectiveCasesRound2-2\LSP22828_001\Scan_20240918_110026_01x19x00130_470.rcpnl",
    r"X:\cycif-production\152-Kidney_Imaging_ProspectiveCasesRound2-2\LSP22828_001\Scan_20240918_110026_01x19x00130_520.rcpnl",
    r"X:\cycif-production\152-Kidney_Imaging_ProspectiveCasesRound2-2\LSP22828_001\Scan_20240918_110026_01x19x00130_555.rcpnl",
    r"X:\cycif-production\152-Kidney_Imaging_ProspectiveCasesRound2-2\LSP22828_001\Scan_20240918_110026_01x19x00130_640.rcpnl",
    r"X:\cycif-production\152-Kidney_Imaging_ProspectiveCasesRound2-2\LSP22828_001\Scan_20240918_110026_01x19x00130_730.rcpnl",
]
imgs = []

for rr in rcpnls:
    rreader = reg.BioformatsReader(str(rr))
    # print(rreader.metadata.num_channels)
    for nn in range(rreader.metadata.num_channels):
        imgs.append(rreader.read(TILE, nn))

shifts = np.cumsum(
    [(0, 0)]
    + [
        utils.register(i1, i2, 1, 10)[0]
        for i1, i2 in itertools.pairwise(tqdm.tqdm(imgs))
    ],
    axis=0,
).round(1)
rimgs = [scipy.ndimage.shift(ii, ss) for ii, ss in zip(tqdm.tqdm(imgs), shifts)]
# plt.figure()
# plt.imshow(
#     np.corrcoef(np.array(rimgs)[:, 200:-200, 200:-200].reshape(19, -1)),
#     vmin=-1,
#     vmax=1,
#     cmap="coolwarm",
# )

plot_corr(
    np.corrcoef(np.array(rimgs)[:, 200:-200, 200:-200].reshape(19, -1)),
    grouping=[1, 1, 2, 3, 3, 3, 6],
    grouping_colors=colors_rgb,
    heatmap_kwargs=dict(
        xticklabels=markers, yticklabels=markers, vmin=-1, vmax=1, cbar=True
    ),
)
fig = plt.gcf()
fig.tight_layout()
fig.suptitle(f"LSP22828_001 (tile {TILE}) RAW")
fig.savefig(f"LSP22828_001 (tile {TILE}) RAW.pdf")

plot_corr(
    np.corrcoef(unmixed.reshape(19, -1)),
    grouping=[1, 1, 2, 3, 3, 3, 6],
    grouping_colors=colors_rgb,
    heatmap_kwargs=dict(
        xticklabels=markers, yticklabels=markers, vmin=-1, vmax=1, cbar=True
    ),
)
fig = plt.gcf()
fig.tight_layout()
fig.suptitle(f"LSP22828_001 (tile {TILE}) UNMIXED")
fig.savefig(f"LSP22828_001 (tile {TILE}) UNMIXED.pdf")


import napari

v = napari.Viewer()
v.add_image(np.array(imgs))
v.add_image(np.array(rimgs))
