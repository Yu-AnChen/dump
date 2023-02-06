import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pathlib


sc.settings.verbosity = 3


def process_qupath_class(qupath_classes, exclude=None):
    if exclude is None:
        exclude = 'Ignore*'
    qupath_classes = np.asarray(qupath_classes, dtype=str)
    qupath_annotations = np.char.split(np.unique(qupath_classes).astype(str), ': ')
    markers = np.unique(np.concatenate(qupath_annotations))
    out = np.zeros((len(qupath_classes), markers.size), dtype=bool)
    for ii, mm in enumerate(markers):
        out[:, ii] = np.char.find(qupath_classes, mm) > -1
    if exclude in markers:
        return pd.DataFrame(out, columns=markers).drop(columns=exclude)
    return pd.DataFrame(out, columns=markers)


def plot_marker_interaction(
    df_positivity,
    markers,
    min_subset_size=100,
    min_degree=1,
    max_degree=None
):
    import upsetplot
    df = df_positivity
    
    upset_data = df.groupby(markers).size()
    assert 'COUNT' not in 'markers'
    upset_data.name = 'COUNT'
    # can use multi-index-ed df too 
    # upset_data = pd.DataFrame(index=pd.MultiIndex.from_frame(df))
    sorted_data = upsetplot.reformat.query(
        upset_data,
        sort_by='degree',
        sort_categories_by='input',
        min_subset_size=min_subset_size,
        min_degree=min_degree,
        max_degree=max_degree
    ).subset_sizes
    upset_plot = upsetplot.UpSet(
        sorted_data,
        sort_by='input',
        sort_categories_by='input',
        orientation='vertical',
        show_counts=True
    )
    upset_plot.plot()
    return pd.DataFrame(sorted_data)


def write_annotation_table(path, df_interaction, overwrite=False):
    path = pathlib.Path(path)
    assert 'COUNT' in df_interaction.columns
    assert path.name.lower().endswith('.csv')
    if path.exists() and not overwrite:
        print(
            'File',
            path,
            'already exists. Set `overwrite=True` to overwrite it.'
        )
        return path
    df_interaction['CELL TYPE'] = pd.Series(dtype='category')
    df_interaction.to_csv(path)
    return path


def cluster_with_positivity_map(
    df_positivity,
    condition_map,
    fillna_with=np.nan
):
    out_df = pd.DataFrame(
        index=df_positivity.index, columns=['Cell type']
    )
    marker_names = condition_map.index.names
    df = pd.DataFrame(
        data=df_positivity.index,
        index=pd.MultiIndex.from_frame(df_positivity[marker_names]),
        columns=['idxs']
    )
    for condition in condition_map.index:
        idxs = df.xs(condition, level=marker_names)['idxs'].values
        out_df.loc[idxs, 'Cell type'] = condition_map.loc[condition, 'CELL TYPE']
    return out_df.fillna(fillna_with)


def df2adata(df, markers_valid=None):

    if markers_valid is None:
        markers_valid = slice(None)

    # intensity columns and rename column name
    mean_intensities = (
        df
        .filter(like='ROI: 0.33 µm per pixel: ')
        .rename(columns=lambda x: x.split('_')[1].replace(': Mean', ''))
    )[markers_valid]

    positions = (
        df
        .filter(like='Centroid')
        .rename(columns={
            'Centroid X µm': 'X_centroid',
            'Centroid Y µm': 'Y_centroid'
        })
    )

    adata = ad.AnnData(mean_intensities)
    adata.raw = adata
    sc.pp.log1p(adata)

    adata.obs['qupath class'] = df.Class.astype('category').values
    adata.obsm['spatial'] = positions[['X_centroid', 'Y_centroid']].values

    return adata


def process_classification(
    cls_array, min_num_cells=500, other_class_name='Ignore*'
):
    cls_array = cls_array.astype(str)
    _names, idxs, counts = np.unique(
        cls_array,
        return_inverse=True,
        return_counts=True
    )
    names = np.array([
        'Others' if n == other_class_name else n
        for n in _names
    ])
    for idx, cc in enumerate(counts):
        if cc < min_num_cells:
            names[idx] = 'Others'
    cls_array[:] = names[idxs]
    return cls_array
    

markers = 'Hoechst, AF1, CD31, Ki-67, AF2, CD163, CD20, CD4, CD8a, CD68, PD-L1, FOXP3, E-Cadherin, PD-1, Blank, CD45, Pan-CK, Blank, SMA'.split(', ')

markers_valid = list(filter(lambda x: x != 'Blank', markers))

SAMPLE_ID = 'P68_S07'
CLUSTER = 'Cell type'
df = pd.read_csv(f'{SAMPLE_ID}_Classification_measurements.csv')
adata = df2adata(df, markers_valid=markers_valid)
marker_positivity = process_qupath_class(
    adata.obs['qupath class'], exclude='Ignore*'
)
adata.obs = adata.obs.join(marker_positivity, on=marker_positivity.index)


# choose markers to assign cell types - CLL story
moi = [
    'CD20+', 'Ki-67+', 'CD68+', 'CD163+'
]

marker_interaction = plot_marker_interaction(
    adata.obs.filter(like='+'),
    moi,
    min_subset_size=200,
    min_degree=1,
    max_degree=None
)
annotation_path = 'annotation-cll.csv'
write_annotation_table(
    annotation_path, marker_interaction
)
celltype_map = pd.read_csv(annotation_path).fillna('Others')
celltype_map.set_index(celltype_map.columns.to_list()[:-2], inplace=True)


# choose markers to assign cell types - Treg story
moi = [
    'CD4+', 'FOXP3+', 'CD8a+'
]

marker_interaction = plot_marker_interaction(
    adata.obs.filter(like='+'),
    moi,
    min_subset_size=200,
    min_degree=1,
    max_degree=None
)
annotation_path = 'annotation-treg.csv'
write_annotation_table(
    annotation_path, marker_interaction
)
celltype_map = pd.read_csv(annotation_path).fillna('Others')
celltype_map.set_index(celltype_map.columns.to_list()[:-2], inplace=True)


fig, ax = plt.subplots()
ax.axis('off')
ax.table(
    cellText=celltype_map.index.to_numpy(),
    rowLabels=celltype_map['CELL TYPE'].values,
    colLabels=celltype_map.index.names, loc='center'
)
fig.tight_layout()
fig.suptitle(f"{CLUSTER} dictionary")


# plot_marker_interaction(adata.obs.filter(like='+'), celltype_map.index.names)
celltypes = cluster_with_positivity_map(
    adata.obs[celltype_map.index.names],
    celltype_map,
    fillna_with='Others'
)
adata.obs[CLUSTER] = celltypes.values

max_n_obs = min(50_000, len(adata.X))
sc.pl.spatial(
    sc.pp.subsample(adata, n_obs=max_n_obs, copy=True),
    spot_size=60,
    color=CLUSTER
)
sc.pl.heatmap(
    adata, adata.var_names,
    groupby=CLUSTER, use_raw=False,
    standard_scale='var'
)



import squidpy as sq

NHOOD_R = 20
sq.gr.spatial_neighbors(adata, radius=(0, NHOOD_R), coord_type='generic', delaunay=False)

sq.gr.nhood_enrichment(adata, cluster_key=CLUSTER)
enrichment_img = adata.uns[f'{CLUSTER}_nhood_enrichment']['zscore']
vmin, vmax = enrichment_img.min(), enrichment_img.max()
vmax = np.abs([vmin, vmax]).max() * 1.05
sq.pl.nhood_enrichment(adata, cluster_key=CLUSTER, cmap='coolwarm', figsize=(5, 5), vmax=vmax, vmin=-vmax, annotate=True)    

sq.gr.interaction_matrix(adata, cluster_key=CLUSTER, normalized=True)
sq.pl.interaction_matrix(adata, cluster_key=CLUSTER, figsize=(5, 5), annotate=True, vmin=0, vmax=1)



# 
# neighborhood composition vector
# 
from numba import njit, prange  # noqa: F401
import numpy as np
import numba.types as nt


dt = nt.uint32  # data type aliases (both for numpy and numba should match)
ndt = np.uint32


@njit(dt[:, :](dt[:], dt[:], dt[:]), parallel=False, fastmath=True)
def _nhood_cluster_count(indices, indptr, clustering):
    res = np.zeros((indptr.shape[0] - 1, len(np.unique(clustering))), dtype=ndt)

    for i in prange(res.shape[0]):
        xs, xe = indptr[i], indptr[i + 1]
        cols = indices[xs:xe]
        for c in cols:
            res[i, clustering[c]] += 1
    
    return res


def nhood_cluster_composition(anndata, cluster_key):

    adj = anndata.obsp['spatial_connectivities']
    indices, indptr = (adj.indices.astype(ndt), adj.indptr.astype(ndt))
    cluster_cats = anndata.obs[cluster_key].cat.categories
    int_clust = anndata.obs[cluster_key].replace(
        cluster_cats, range(len(cluster_cats))
    ).astype(np.uint32).values

    cluster_count = _nhood_cluster_count(indices, indptr, int_clust)
    anndata.obsm[f"{cluster_key}_nhood_cluster_count"] = cluster_count
    anndata.uns[f"{cluster_key}_nhood_cluster_count_columns"] = cluster_cats

    cluster_count_mean = np.zeros((len(cluster_cats), len(cluster_cats)))
    for rr, cc in zip(range(len(cluster_count_mean)), cluster_cats):
        mask = anndata.obs[f"{cluster_key}"] == cc
        cluster_count_mean[rr] = cluster_count[mask].mean(axis=0)
    anndata.uns[f"{cluster_key}_nhood_cluster_count_mean"] = cluster_count_mean

    cluster_sum = cluster_count.sum(axis=1).reshape(-1, 1)
    cluster_fraction = cluster_count.astype(float) / cluster_sum
    anndata.obsm[f'{cluster_key}_nhood_cluster_fraction'] = cluster_fraction
    anndata.uns[f"{cluster_key}_nhood_cluster_fraction_columns"] = cluster_cats
    return 

# 
# mean distance between objects between/within clusters
# 
import sklearn.neighbors
import matplotlib
import matplotlib.pyplot as plt

def knn_distance_from_to(from_points, to_points, n_neighbors, plot=False, exclude_self=True):
    processing_self = False
    if from_points.shape == to_points.shape:
        if np.all(np.equal(from_points, to_points)):
            processing_self = True
    if processing_self and exclude_self:
        n_neighbors += 1

    knn_from = sklearn.neighbors.NearestNeighbors()
    knn_from.fit(to_points)

    from_points_nn = knn_from.kneighbors(
        from_points, n_neighbors=n_neighbors, return_distance=True
    )

    if processing_self and exclude_self:
        from_points_nn = (
            from_points_nn[0][:, 1:], from_points_nn[1][:, 1:]
        )
        
    if plot:
        plt.figure()
        plt.plot(to_points[:, 0], to_points[:, 1], "og", markersize=14)
        plt.plot(from_points[:, 0], from_points[:, 1], "xk", markersize=14)

        for i, col in enumerate(from_points_nn[1].T):
            color = matplotlib.cm.viridis(
                np.linspace(0, 1, from_points_nn[1].shape[1])[i]
            )
            nn_points = to_points[col]
            plt.plot(
                np.vstack([from_points, nn_points]).T.reshape(4, -1)[:2],
                np.vstack([from_points, nn_points]).T.reshape(4, -1)[2:],
                color=color
            )

    return from_points_nn


def paired_knn_distance_mean(coords, cluster_label, labels_of_interest=None, n_neighbors=1):
    labels = np.unique(cluster_label)
    if labels_of_interest is None:
        labels_of_interest = labels

    size = labels_of_interest.size
    out = np.ones((len(cluster_label), size))

    out_heatmap = np.zeros([size]*2)
    combinations = np.mgrid[:size, :size].reshape(2, -1).T

    for ii, jj in combinations:
        if (labels_of_interest[ii] in labels) & (labels_of_interest[jj] in labels):
            distances = knn_distance_from_to(
                coords[cluster_label == labels_of_interest[ii]],
                coords[cluster_label == labels_of_interest[jj]],
                n_neighbors=n_neighbors, exclude_self=True, plot=False
            )[0]
            out_heatmap[ii, jj] = distances.mean()
            out[
                cluster_label == labels_of_interest[ii], jj
            ] = distances.mean(axis=1)

    return out_heatmap, out



# 
# nhoood composition plotting
#
nhood_cluster_composition(adata, cluster_key=CLUSTER)
sq.pl._utils._heatmap(
    ad.AnnData(
        adata.uns[f'{CLUSTER}_nhood_cluster_count_mean'],
        obs={CLUSTER: pd.Categorical(adata.obs[CLUSTER].cat.categories)},
        uns={f"{CLUSTER}_colors": adata.uns[f"{CLUSTER}_colors"]}
    ),
    key=CLUSTER,
    title='Mean neighborhood composition',
    figsize=(5, 5), annotate=True
)


# 
# mean distance between objects between/within clusters
# 
N_NEIGHBORS = 2
distance_heatmap, sc_distances = paired_knn_distance_mean(
    adata.obsm['spatial'],
    adata.obs[CLUSTER],
    labels_of_interest=adata.obs[CLUSTER].cat.categories,
    n_neighbors=N_NEIGHBORS
)
adata.uns[f'{CLUSTER}_paired_distance_heatmap'] = distance_heatmap
adata.obsm[f'{CLUSTER}_paired_nn_distance_{N_NEIGHBORS}'] = sc_distances

sq.pl._utils._heatmap(
    ad.AnnData(
        adata.uns[f'{CLUSTER}_paired_distance_heatmap'],
        obs={CLUSTER: pd.Categorical(adata.obs[CLUSTER].cat.categories)},
        uns={f"{CLUSTER}_colors": adata.uns[f"{CLUSTER}_colors"]}
    ),
    key=CLUSTER,
    title=f'Mean NN({N_NEIGHBORS}) distance',
    cont_cmap='cividis_r',
    figsize=(5, 5), annotate=True
)


# 
# save figure block
# 
import matplotlib.pyplot as plt
import matplotlib
import pathlib

def set_matplotlib_font():
    font_families = matplotlib.rcParams['font.sans-serif']
    if font_families[0] != 'Arial':
        font_families.insert(0, 'Arial')
    matplotlib.rcParams['pdf.fonttype'] = 42

def save_all_fig(figs=None, dpi=144, label=None, close=True, out_dir=None):
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        num = fig.number
        title = '' if fig._suptitle is None else fig._suptitle.get_text()
        if label is None: label = ''
        title = label + title
        fig.suptitle(title)
        filename = f"fig {num}"
        if title:
            filename += f" - {title}"
        out_dir = pathlib.Path('.') if out_dir is None else pathlib.Path(out_dir)
        fig.savefig(out_dir / f"{filename.strip()}.png", bbox_inches='tight', dpi=dpi)
        if close:
            plt.close(num)

set_matplotlib_font()
save_all_fig(label=f"{SAMPLE_ID}\n", dpi=180)


# 
# napari visualization block
# 

# CLL story
TARGET_NHOODS = ['CD20+', 'CD20+ Ki-67+']
COMPOSITION_CLASSES = ['CD68+', 'CD163+', 'CD68+ CD163+']
# Treg story
# TARGET_NHOODS = ['Treg']
# COMPOSITION_CLASSES = ['TC']

import palom
import napari
import pathlib

v = napari.Viewer()
v.scale_bar.visible = True
v.scale_bar.unit = "um"

# v.add_points(df_core[['Y_centroid', 'X_centroid']], face_color=df_core.filter(like='Status').values)

import matplotlib.cm
cluster_cats = adata.obs[CLUSTER].cat.categories
matched_colors = np.array(adata.uns[f'{CLUSTER}_colors'])[
    adata.obs[CLUSTER].replace(
        cluster_cats, range(len(cluster_cats))
    ).astype(int)
]
# v.add_points(df_core[['Y_centroid', 'X_centroid']], face_color=matched_colors, size=3)
v.add_points(np.fliplr(adata.obsm['spatial']), face_color=matched_colors, size=3)
v.add_points(np.fliplr(adata.obsm['spatial']), face_color=matched_colors, size=30)


node_of_interest_mask = adata.obs.eval(
    f'`{CLUSTER}` in @TARGET_NHOODS'
)

noi_coor = np.fliplr(adata.obsm['spatial'][node_of_interest_mask])
connections = adata.obsp['spatial_connectivities'][node_of_interest_mask].toarray().astype(bool)
conn_coor = [np.fliplr(adata.obsm['spatial'][c]) for c in connections]

lines = []
for nn, cc in zip(noi_coor, conn_coor):
    for icc in cc:
        lines.append([nn, icc])

v.add_shapes(lines, shape_type='line', edge_color='white', edge_width=0.5)
v.add_points(noi_coor, size=30)


# napari viz nhood composition
import skimage.exposure
cluster_fraction = adata.obsm[f'{CLUSTER}_nhood_cluster_fraction']
celltype_columns = adata.uns[f'{CLUSTER}_nhood_cluster_fraction_columns']
composition_mask = [
    t in COMPOSITION_CLASSES for t in celltype_columns
]
ints = cluster_fraction[node_of_interest_mask][
    :, composition_mask
].sum(axis=1)
fcs = matplotlib.cm.cividis(
    skimage.exposure.rescale_intensity(
        ints,
        in_range=(0, 0.4),
        out_range=(0, 1)
    )
)
for w in [1, 5]:
    v.add_shapes(
        [np.vstack([c, [NHOOD_R, NHOOD_R]]) for c in noi_coor],
        edge_color=fcs,
        shape_type='ellipse',
        edge_width=w,
        face_color=np.array((0, 0, 0, 0))
    )


# napari viz NN distance
lines = []
coords_to_mask = adata.obs.eval(
    f'`{CLUSTER}` in @COMPOSITION_CLASSES'
)
coords_from_mask = adata.obs.eval(
    f'`{CLUSTER}` in @TARGET_NHOODS'
)
coords_to = adata[coords_to_mask].obsm['spatial']
coords_from = adata[coords_from_mask].obsm['spatial']
distances, idxs = knn_distance_from_to(coords_from, coords_to, 2)
print(id, np.percentile(distances, 1), np.percentile(distances, 99))
for dd, ii, ff in zip(distances, idxs, coords_from):
    for iii in ii:
        lines.append([ff[::-1], coords_to[iii][::-1]])
colors = matplotlib.cm.cividis_r(
    skimage.exposure.rescale_intensity(distances.flatten(), in_range=(5, 90), out_range=(0, 1))
)
v.add_shapes(lines, shape_type='line', edge_color=colors, edge_width=4, name=id)

