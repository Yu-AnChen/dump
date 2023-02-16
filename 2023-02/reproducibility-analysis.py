import pandas as pd
import scipy.stats
import numpy as np

table_paths = pd.read_csv(r'Z:\RareCyte-S3\YC-analysis\P37_CRCstudy_Round1\scripts-analysis\c40-tables.csv', header=None)[0].to_list()

dfs = [pd.read_csv(p, index_col='CellID') for p in table_paths]

markers =  list(dfs[0].columns[:19])
# markers = list(filter(lambda x: x not in ['Hoechst', 'AF1', 'Argo550'], markers))

# a different sample as reference
df_test = pd.read_csv(
    # 
    # r'Z:\RareCyte-S3\YC-analysis\P37_CRCstudy_Round1\P37_S37-CRC09\quantification\raw\P37_S37_A24_C59kX_E15@20220108_012113_953544_cellRingMask_intensity.csv'
    r'Z:\RareCyte-S3\YC-analysis\P37_CRCstudy_Round1\P37_S29-CRC01\quantification\raw\P37_S29_A24_C59kX_E15@20220106_014304_946511_cellRingMask_intensity.csv'
)

# scipy.stats.wasserstein_distance(
#     dfs[0][mm].nlargest(200_000).transform(np.array).values,
#     df_test[mm].sample(frac=0.5).transform(np.array).values
# )



# coarse register between C40 images
def coords2img(coords, out_shape=None, scale=1, as_counts=False):
    # coords in the form of (N, 2)
    assert coords.ndim == 2
    assert coords.shape[1] == 2
    coords = np.asarray(coords)
    coords = np.around(coords * scale).astype(int)
    if out_shape is None:
        out_shape = coords.max(axis=0) + 1
    
    idxs, counts = coords, True
    dtype = bool
    if as_counts:
        idxs, counts = np.unique(coords, axis=0, return_counts=True)
        dtype = np.min_scalar_type(max(counts))
    img = np.zeros(out_shape, dtype=dtype)
    img.flat[np.ravel_multi_index(idxs.T, img.shape)] = counts

    return img

imgs = [
    coords2img(df[['Y_centroid', 'X_centroid']], scale=1/64, as_counts=True)
    for df in dfs
]
import palom

affine_mxs = [np.eye(3)]
for ii in imgs[1:]:
    mx = palom.register.feature_based_registration(imgs[0], ii, plot_match_result=True, n_keypoints=4000)
    mx = np.vstack([mx, [0, 0, 1]])
    affine_mxs.append(mx)


import skimage.transform
tforms = [skimage.transform.AffineTransform(matrix=mm) for mm in affine_mxs]

# mattched coordinates are now in X and Y
for tform, df in zip(tforms, dfs):
    df.loc[:, ['X', 'Y']] = tform(df[['X_centroid', 'Y_centroid']]/64) * 64

# visual inspection to select common region without artifacts
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 4, sharex=True, sharey=True)
for ax, df in zip(axs, dfs):
    ax.imshow([[0]], alpha=0)
    ax.scatter(*df[['X', 'Y']].values.T, linewidths=0, s=1)

ylims = [1_113, 45_693]
xlims = [31_613, 69_875]

# finalize the df based on common region
valid_dfs = [
    df[df['X'].between(*xlims) & df['Y'].between(*ylims)]
    for df in dfs
]

plt.figure()
plt.imshow([[0]])
for df in valid_dfs:
    plt.scatter(*df[['X', 'Y']].values.T, linewidths=0, s=1)


# quick check with qq plot
fig, axs = plt.subplots(5, 4)
for mm, ax in zip(markers, axs.flatten()):
    ax.set_title(mm)
    ax.plot(
        np.log1p(np.percentile(valid_dfs[0][mm], np.linspace(90, 99.9, 1000))),
        np.log1p(np.percentile(valid_dfs[0][mm], np.linspace(90, 99.9, 1000))),
        'k--'
    )
    for i in range(4):
        ax.scatter(
            np.log1p(np.percentile(valid_dfs[0][mm], np.linspace(90, 99.9, 1000))),
            np.log1p(np.percentile(valid_dfs[i][mm], np.linspace(90, 99.9, 1000))),
            linewidths=0, s=4
        )
    ax.scatter(
        np.log1p(np.percentile(valid_dfs[0][mm], np.linspace(90, 99.9, 1000))),
        np.log1p(np.percentile(df_test[mm], np.linspace(90, 99.9, 1000))),
        linewidths=0, s=4
    )


def pad_size(dfs):
    num_rows = [df.shape[0] for df in dfs]
    max_row = max(num_rows)
    return max_row - np.array(num_rows, dtype=int)


pad_valid_df = pad_size(valid_dfs+[df_test])

fig, axs = plt.subplots(5, 4)
aucs = []
for mm, ax in zip(markers, axs.flatten()):
    ax.set_title(mm)
    hist_data = np.array([
        np.pad(df[mm], (0, pp), constant_values=np.nan)
        for df, pp in zip(valid_dfs+[df_test], pad_valid_df)
    ])
    _hist_data = ax.hist(np.array(hist_data).T, bins=30, histtype='step', density=True, log=(True, True))

    col_start = _hist_data[0].max(axis=0).argmax()
    # col_start = 0
    aucs.append([
        # np.min(_hist_data[0][[0, i], col_start:], axis=0).sum() / _hist_data[0][0, col_start:].sum()
        np.min(_hist_data[0][[0, i], col_start:], axis=0).sum() / np.max(_hist_data[0][[0, i], col_start:], axis=0).sum()
        for i in range(len(_hist_data[0]))
    ])
    # ax.set_xlim(4, 8)

import seaborn as sns
fig, axs = plt.subplots(5, 4)
for mm, ax in zip(markers[:], axs.flatten()):
    ax.set_title(mm)
    hist_data = np.array([
        np.pad(df[mm], (0, pp), constant_values=np.nan)
        for df, pp in zip(valid_dfs+[df_test], pad_valid_df)
    ])
    xlims = np.nanpercentile(hist_data[0], (1, 99.9))
    sns.histplot(data=hist_data.T[:, :] + 100, element='step', fill=False, log_scale=(True, False), stat='percent', ax=ax, common_norm=False)
    ax.set_xlim(*xlims)

p90s = [
    np.percentile(valid_dfs[0][mm] + 100, 90)
    for mm in markers
]

aucs = []
for ax, pp in zip(axs.flatten()[:len(markers)], p90s):
    mask = ax.lines[-1].get_xydata()[:, 0] > pp
    # mask = True
    # MUST invert the lines since the new line are prepended to the list
    hhights = np.array([line.get_xydata()[:, 1] for line in ax.lines[::-1]])
    areas = [
        np.min([hhights[0], row], axis=0)[mask].sum() / np.max([hhights[0], row], axis=0)[mask].sum()
        for row in hhights
    ]
    aucs.append(areas)

fig, ax = plt.subplots()
sns.heatmap(pd.DataFrame(aucs, index=markers), annot=True)


# EMD
# cutoffs = pd.DataFrame(np.array([df.mean()[:19] for df in dfs]).mean(axis=0).reshape(1, -1), columns=dfs[0].columns[:19])
cutoffs = pd.DataFrame([p90s], columns=markers)

import tqdm
emd_markers = []
for mm in tqdm.tqdm(markers):
    emd = []
    for i in range(0, 4):
        emd.append(scipy.stats.wasserstein_distance(
            valid_dfs[0].query(f"`{mm}` > {cutoffs[mm].values[0]}")[mm].transform(np.array).values,
            valid_dfs[i].query(f"`{mm}` > {cutoffs[mm].values[0]}")[mm].transform(np.array).values
        ))
    emd.append(
        scipy.stats.wasserstein_distance(
            valid_dfs[0].query(f"`{mm}` > {cutoffs[mm].values[0]}")[mm].transform(np.array).values,
            df_test.query(f"`{mm}` > {cutoffs[mm].values[0]}")[mm].transform(np.array).values
        )
    )
    emd_markers.append(emd)


fig, ax = plt.subplots()
sns.heatmap(pd.DataFrame(emd_markers, index=markers), annot=True, cmap='magma_r')

