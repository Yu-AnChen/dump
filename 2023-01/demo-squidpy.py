import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('demo-P68_S07.csv')
adata = ad.AnnData(df.iloc[:, 1:-1])

adata.raw = adata
adata = adata[:, 2:]

adata.obsm['spatial'] = adata.raw[:, ['X_centroid', 'Y_centroid']].X
adata.obs['Cell type'] = df['Cell type'].astype('category').values

CLUSTER = 'Cell type'
sc.pl.spatial(adata, spot_size=8, color=CLUSTER)
plt.tight_layout()
sc.pl.heatmap(adata, adata.var_names, groupby=CLUSTER, use_raw=False, standard_scale='var')


import napari

v = napari.Viewer()
v.scale_bar.visible = True
v.scale_bar.unit = "um"

cluster_cats = adata.obs[CLUSTER].cat.categories
matched_colors = np.array(adata.uns['Cell type_colors'])[
    adata.obs[CLUSTER].replace(
        cluster_cats, range(len(cluster_cats))
    ).astype(int)
]

v.add_points(np.fliplr(adata.obsm['spatial']), face_color=matched_colors, size=3)


import squidpy as sq

NHOOD_R = 20
sq.gr.spatial_neighbors(adata, radius=(0, NHOOD_R), coord_type='generic', delaunay=False)


node_of_interest_mask = adata.obs.eval(f'`{CLUSTER}` == "FOXP3+ CD4+"')

noi_coor = np.fliplr(adata.obsm['spatial'][node_of_interest_mask])
connections = adata.obsp['spatial_connectivities'][node_of_interest_mask].toarray().astype(bool)
conn_coor = [np.fliplr(adata.obsm['spatial'][c]) for c in connections]

lines = []
for nn, cc in zip(noi_coor, conn_coor):
    for icc in cc:
        lines.append([nn, icc])

v.add_shapes(lines, shape_type='line', edge_color='white', edge_width=0.5)
v.add_points(noi_coor, size=3)


sq.gr.nhood_enrichment(adata, cluster_key=CLUSTER)
enrichment_img = adata.uns[f'{CLUSTER}_nhood_enrichment']['zscore']
vmin, vmax = enrichment_img.min(), enrichment_img.max()
vmax=150
sq.pl.nhood_enrichment(adata, cluster_key=CLUSTER, cmap='coolwarm', figsize=(3, 3), vmax=vmax, vmin=-vmax, annotate=True)

sq.gr.interaction_matrix(adata, cluster_key=CLUSTER, normalized=True)
sq.pl.interaction_matrix(adata, cluster_key=CLUSTER, figsize=(3, 3), annotate=True, vmin=0, vmax=1)


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


def nhood_cluster_count(anndata, cluster_key):

    adj = anndata.obsp['spatial_connectivities']
    indices, indptr = (adj.indices.astype(ndt), adj.indptr.astype(ndt))
    cluster_cats = anndata.obs[cluster_key].cat.categories
    int_clust = anndata.obs[cluster_key].replace(
        cluster_cats, range(len(cluster_cats))
    ).astype(np.uint32).values

    anndata.obsm[f"{cluster_key}_nhood_cluster_count"] = _nhood_cluster_count(indices, indptr, int_clust)
    return 

nhood_cluster_count(adata, cluster_key=CLUSTER)
cluster_count = adata.obsm[f'{CLUSTER}_nhood_cluster_count'].astype(float)
cluster_count /= cluster_count.sum(axis=1).reshape(-1, 1)
adata.obsm[f'{CLUSTER}_nhood_cluster_fraction'] = cluster_count

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

N_NEIGHBORS = 2
distance_heatmap, sc_distances = paired_knn_distance_mean(
    adata.obsm['spatial'],
    adata.obs[CLUSTER],
    labels_of_interest=adata.obs[CLUSTER].cat.categories,
    n_neighbors=N_NEIGHBORS
)
adata.uns[f'{CLUSTER}_paired_distance_heatmap'] = distance_heatmap
adata.obsm[f'{CLUSTER}_paired_nn_distance_{N_NEIGHBORS}'] = sc_distances


import skimage.exposure
# fraction of CD8a+ T cell in Treg nhood
for i, n in enumerate(adata.obs['Cell type'].cat.categories):
    print(i, n)
ints = adata.obsm[f'{CLUSTER}_nhood_cluster_fraction'][node_of_interest_mask][:, [9, 10]].sum(axis=1)
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


distance_heatmap, sc_distances = paired_knn_distance_mean(
    adata.obsm['spatial'],
    adata.obs[CLUSTER],
    labels_of_interest=adata.obs[CLUSTER].cat.categories,
    n_neighbors=N_NEIGHBORS
)
adata.uns[f'{CLUSTER}_paired_distance_heatmap'] = distance_heatmap
adata.obsm[f'{CLUSTER}_paired_nn_distance_{N_NEIGHBORS}'] = sc_distances
# hijack interaction uns for plotting purpose
adata.uns[f'{CLUSTER}_interactions'] = distance_heatmap
sq.pl.interaction_matrix(adata, cluster_key=CLUSTER, figsize=(3, 3), annotate=True)
plt.gca().get_figure().axes[2].set_title('NN distance matrix')