import itertools
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.exposure
import sklearn.decomposition
import tqdm
import umap

import gmm

tqdm.auto.tqdm.pandas()


def load_data(filepath, markers, markers_min=0, dna_like=None, dna_min=0):
    _df = (
        pd.read_parquet(filepath, engine="pyarrow")
        .reset_index()
        .set_index(["Sample", "CellID"])
    )
    mdf = _df[markers]
    if dna_like is not None:
        ddf = _df.filter(like=dna_like)

    mmask = np.full(len(_df), bool, dtype="bool")
    dmask = np.full(len(_df), bool, dtype="bool")
    if markers_min > 0:
        mmask = mdf.min(axis=1) > markers_min
    if (dna_like is not None) & (dna_min > 0):
        dmask = ddf.min(axis=1) > dna_min
    return mdf[mmask & dmask]


def bottom_top(data, p0, p1):
    data = np.asarray(data)
    p0, p1 = np.percentile(data.ravel(), [p0, p1])
    vmin = np.mean(data, where=data <= p0)
    vmax = np.mean(data, where=data >= p1)
    if p0 == 0:
        vmin = data.min()
    if p1 == 100:
        vmax = data.max()
    return vmin, vmax


def rescale_markers(df, p0=0, p1=100):
    return df.apply(
        lambda x: skimage.exposure.rescale_intensity(
            x, in_range=bottom_top(x.values, p0, p1)
        ),
    )


markers = "CD11c,GranzymeB,CD68,CD45,CD45RO,panCK,CD20,Ecad,aSMA,Vimentin,CD4,FOXP3,PDL1,CD15,CD3d,CD16,CD44,CD8a,PCNA,Desmin,PD1,Catenin,H3K27me3,HLAA,Nestin,CD163,Iba1,H2sx,pSTAT1,STING,pTBK1,PDPN".split(
    ","
)[:12]

load_columns = markers + ["Sample"]

dfs_log = [
    # simple QC - marker min intensity > 200 and DNA min intensity > 1000
    # nature laog transform the data
    load_data(pp, markers, 200, "Hoechst", 1000).transform(np.log1p)
    for pp in ["LSP10532.parquet", "LSP10543.parquet"]
]

# keep track of indicies for subsetting each dataset (fraction=0.05)
dfs_subset = [dd.index.to_frame().sample(frac=0.05) for dd in dfs_log]


def compute_anchors(data):
    """
    compute bunch of reference points
    p0: min
    p100: max
    mp0.01/mp0.05: mean of values <= 0.01/0.05 precentile
    mp99.99/mp99.95: mean of values >= 99.99/99.95 precentile
    gmm_peakX: mean of gaussian modes
    """
    max_n = 200_000

    data = np.asarray(data)

    out = {}

    out["p0"], out["p100"] = data.min(), data.max()
    out["mp0.01"], out["mp99.99"] = bottom_top(data, 0.01, 99.99)
    out["mp0.05"], out["mp99.95"] = bottom_top(data, 0.05, 99.95)

    _data = data
    if data.size > max_n:
        rng = np.random.default_rng()
        _data = rng.choice(data, max_n, replace=False, shuffle=False)

    out["gmm_peak1"], out["gmm_peak2"] = np.sort(
        gmm.get_gmm_and_pos_label(_data)[0].means_.ravel()
    )

    return out


anchors = [dd.progress_apply(lambda x: pd.Series(compute_anchors(x))) for dd in dfs_log]

scalings = [
    ("p0", "p100"),
    ("mp0.01", "mp99.99"),
    ("mp0.05", "mp99.95"),
    ("mp0.05", "gmm_peak1"),
    ("gmm_peak1", "gmm_peak2"),
    ("mp0.05", "gmm_peak1", "gmm_peak2", "mp99.95"),
]


# use anchor point pairs and compute the linear transformation (slope & offset)
# for each marker
scaling_funcs = []
for ii, aa in enumerate(anchors):
    dict_scaling = {}

    for ss, mm in itertools.product(scalings, markers):
        if ii == 0:
            poly1d_fn = np.asarray
        else:
            coef = np.polyfit(aa.loc[list(ss), mm], anchors[0].loc[list(ss), mm], 1)
            poly1d_fn = np.poly1d(coef)
        if ss not in dict_scaling:
            dict_scaling[ss] = {}

        dict_scaling[ss].update({mm: poly1d_fn})

    scaling_funcs.append(dict_scaling)


# viz linear transformed data with histogram
for scal in scalings[:]:
    fig, axs = plt.subplots(4, np.ceil(len(markers) / 4).astype("int"))
    for ss, ax in zip(markers, axs.ravel()):
        for ii, (dd, sf) in enumerate(zip(dfs_log, scaling_funcs)):
            dd = dd[ss]
            dd = sf[scal][ss](dd)
            bins = np.linspace(*np.percentile(dd, [0.01, 99.99]), 200)
            _ = ax.hist(dd, bins=bins, histtype="step", density=True)
        ax.set_title(ss)
        fig.suptitle(scal)
    fig.tight_layout()


# ---------------------------------------------------------------------------- #
#                  transform -> rescale -> PCA (5 PCs) -> UMAP                 #
# ---------------------------------------------------------------------------- #
scaling_sets = []
# basic "min/max" normalization
for ss in [("p0", "p100"), ("mp0.01", "mp99.99"), ("mp0.05", "mp99.95")]:
    scaling_sets.append({mm: scaling_funcs[1][ss][mm] for mm in markers})

# hand-configured gmm normalization
selected_scaling = {mm: ("gmm_peak1", "gmm_peak2") for mm in markers}
selected_scaling["CD45"] = ("mp0.05", "gmm_peak1")
selected_scaling["CD20"] = ("mp0.05", "gmm_peak1")

scaling_sets.append({mm: scaling_funcs[1][selected_scaling[mm]][mm] for mm in markers})


sdfs = []
umap_tforms = []
for sf in tqdm.tqdm(scaling_sets[:]):
    _df = [dd.loc[ii.index].copy() for dd, ii in zip(dfs_log, dfs_subset)]
    _df[1] = _df[1].progress_apply(lambda x: sf[x.name](x))
    df = pd.concat(_df).progress_apply(
        lambda x: skimage.exposure.rescale_intensity(
            x, in_range=bottom_top(x, 0.01, 99.99)
        ),
    )
    pca = sklearn.decomposition.PCA(n_components=5)
    pca_tformed = pca.fit_transform(df)

    umapper = umap.UMAP(densmap=True)
    umap_tformed = umapper.fit_transform(pca_tformed)

    sdfs.append(df)
    umap_tforms.append(umap_tformed)


# plot single-cell in the umap space and color by -
# 1) sample density
# 2) marker expression level
set_name = [
    ("p0", "p100"),
    ("mp0.01", "mp99.99"),
    ("mp0.05", "mp99.95"),
    f'{("gmm_peak1", "gmm_peak2")}*',
]
for umap_tformed, sdf, nn in zip(umap_tforms, sdfs, set_name):
    img_size = 500
    grouper = pd.DataFrame(umap_tformed).apply(
        lambda x: np.digitize(x, bins=np.linspace(x.min(), x.max(), img_size))
    )
    expr = sdf.groupby(grouper.T.values.tolist()).mean()

    count_by_sample = (
        df.index.to_frame()[["Sample"]]
        .astype("category")
        .groupby(grouper.T.values.tolist())["Sample"]
        .value_counts()
        .reset_index(level="Sample")
        .pivot(columns="Sample")
    )

    fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
    fig.suptitle(nn)
    vmin, vmax = 1, 3
    kwarg_args = dict(cmap="turbo", interpolation="none", vmin=vmin, vmax=vmax)
    for ii, (ss, ax) in enumerate(zip(count_by_sample, axs[1:, 0].ravel())):
        xx, yy = count_by_sample.index.to_frame().values.T
        img = np.zeros((img_size + 1,) * 2)
        img[yy, xx] = count_by_sample[ss]
        img = np.where(img == 0, np.nan, img)
        ax.imshow(np.log1p(img), **kwarg_args)
        ax.set_title(ss[1])
    img[yy, xx] = count_by_sample.sum(axis=1)
    img = np.where(img == 0, np.nan, img)
    axs[0, 0].imshow(np.log1p(img), **kwarg_args)
    axs[0, 0].set_title("All samples")
    axs[-1, 0].axis("off")

    for ss, ax in zip(expr, axs[:, 1:].ravel()):
        xx, yy = expr.index.to_frame().values.T
        img = np.full((img_size + 1,) * 2, np.nan)
        img[yy, xx] = expr[ss]
        ax.imshow(img, cmap="cividis", interpolation="none")
        ax.set_title(ss)
    fig.tight_layout()


# ---------------------------------------------------------------------------- #
#                                 save figures                                 #
# ---------------------------------------------------------------------------- #
def save_all_figs(dpi=300, format="pdf", out_dir=None, prefix=None, close=True):
    figs = [plt.figure(i) for i in plt.get_fignums()]
    if prefix is not None:
        for f in figs:
            if f._suptitle:
                f.suptitle(f"{prefix} {f._suptitle.get_text()}")
            else:
                f.suptitle(prefix)
    names = [f._suptitle.get_text() if f._suptitle else "" for f in figs]
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for f, n, nm in zip(figs, plt.get_fignums(), names):
        f.savefig(out_dir / f"{n}-{nm}.{format}", dpi=dpi, bbox_inches="tight")
        if close:
            plt.close(f)


for ii in range(1, 7):
    ff = plt.figure(ii)
    ff.set_size_inches(6, 7)
    ff.tight_layout()

for ii in range(7, 11):
    ff = plt.figure(ii)
    ff.set_size_inches(9, 10)
    ff.tight_layout()

save_all_figs(format="pdf", out_dir="plots")

for dd, ii in zip(dfs_log, dfs_subset):
    gmm.plot_hist_gmm(
        dd.loc[ii.index].transform(np.expm1),
        markers,
        subplot_grid_shape=(4, 3)
    )

for ii, nn in zip(range(1, 3), ["LSP10532", "LSP10543"]):
    ff = plt.figure(ii)
    ff.set_size_inches(9, 10)
    ff.suptitle(nn)
    ff.tight_layout()
    
save_all_figs(format="pdf", out_dir="plots")
