import pandas as pd
import numpy as np

def hist_2d(
    df, bin_size, query=None,
    spatial_x_name='Xt', spatial_y_name='Yt',
    kde=False,
    kde_kwargs=None
):
    # snap coordinates to grid
    df_coords = df[[spatial_y_name, spatial_x_name]] / bin_size
    df_coords = df_coords.apply(lambda x: pd.to_numeric(x.round(), downcast='integer'))
    # set output image size
    h, w = df_coords.max() + 1
    # query by condition string
    if query is not None:
        df_coords = df_coords[df.eval(query)]
    counts = df_coords.groupby([spatial_y_name, spatial_x_name]).size()
    y, x = counts.index.to_frame().values.T
    img = np.zeros((h, w))
    img[y, x] = counts

    if kde:
        img_kde = _kde_2d(img, kde_kwargs)
        return img, img_kde

    return img


import KDEpy

def _kde_2d(img, kde_kwargs=None):
    assert img.ndim == 2
    rs, cs = img.nonzero()
    values = img[rs, cs]

    h, w = img.shape
    if kde_kwargs is None: kde_kwargs = {}
    kde = KDEpy.FFTKDE(**kde_kwargs)
    # 1. FIXME does it need to be shifted by half pixel?
    # 2. ValueError: Every data point must be inside of the grid.
    grid_pts = np.mgrid[-1:h+1, -1:w+1].reshape(2, -1).T
    return (
        kde
        .fit(np.vstack([rs, cs]).T, weights=values)
        .evaluate(grid_pts)
        .reshape(h+2, w+2)[1:-1, 1:-1]
    )


import matplotlib.pyplot as plt
import napari
import matplotlib.cm

def napari_contour(
    img_contour, n_levels,
    cmap_name='viridis',
    viewer=None, add_shape_kwargs=None
):
    _, ax = plt.subplots()
    ax.imshow([[0]])
    ctr = ax.contour(img_contour, levels=n_levels, cmap=cmap_name)

    colors = matplotlib.cm.get_cmap(cmap_name)(np.linspace(0, 1, n_levels+2))[1:-1]
    level_values = ctr.levels[1:-1]

    if viewer is None:
        viewer = napari.Viewer()
    
    kwargs = dict(
        shape_type='polygon',
        face_color=[0]*4,
    )
    if add_shape_kwargs is None: add_shape_kwargs = {}

    for level, color, lv in zip(ctr.allsegs[1:-1], colors, level_values):
        kwargs.update(dict(edge_color=color, name=str(lv)))
        kwargs.update(add_shape_kwargs)
        viewer.add_shapes(
            [np.fliplr(seg)[:-1, :] for seg in level],
            **kwargs
        )
    return viewer

# jdf = pd.read_csv(r"Y:\sorger\data\computation\Yu-An\YC-20230126-squidpy_demo\dataC04.csv")

jdf = pd.read_csv(r"W:\crc-scans\C1-C40-sc-tables\demo-c4.csv")
jdf = jdf.set_index('CellID')

topic8 = hist_2d(jdf, bin_size=25, query='topics == 8', kde=True, kde_kwargs=dict(bw=5))
v = napari_contour(topic8[1], n_levels=5, cmap_name='gist_earth')
v.add_image(np.array(topic8), channel_axis=0)






def spatial_cat_count(
    df, col_name, col_value,
    spatial_x_name='tXt', spatial_y_name='tYt'
):
    count = (
        df
        .query(f'{col_name} == {col_value}')
        .groupby([spatial_x_name, spatial_y_name])[col_name]
        .count()
    )
    x, y = count.index.to_frame().values.T
    img = np.zeros((df[spatial_y_name].max()+1, df[spatial_x_name].max()+1))
    img[y, x] = count
    return img

import matplotlib.pyplot as plt


n_levels = 5
_, ax = plt.subplots()
ax.imshow([[0]])
ctr = ax.contour(topic8[1], levels=n_levels)


import napari
import matplotlib.cm

colors = matplotlib.cm.viridis(np.linspace(0, 1, n_levels))
level_values = ctr.levels[1:-1]
vvv = napari.Viewer()

for level, color, lv in zip(ctr.allsegs[1:-1], colors, level_values):
    vvv.add_shapes(
        [np.fliplr(seg)[:-1, :] for seg in level],
        edge_color=color,
        name=str(lv),
        shape_type='polygon',
        face_color=[0]*4,
    )







def spatial_cat_kde(
    df, col_name, col_value,
    spatial_x_name='tXt', spatial_y_name='tYt',
    kde_kwargs=None
):
    height, width = np.ceil(
        df[[spatial_y_name, spatial_x_name]].max()
    ).astype(int) + 2
    target_df = df.query(f'{col_name} == {col_value}')
    data = target_df[[spatial_y_name, spatial_x_name]].values
    if kde_kwargs is None: kde_kwargs = {}
    kde = KDEpy.FFTKDE(**kde_kwargs)
    # FIXME does it need to be shifted by half pixel?
    grid_pts = np.mgrid[-.5:height-.5, -.5:width-.5].reshape(2, -1).T
    return kde.fit(data).evaluate(grid_pts).reshape(height, width)[1:, 1:]


jdf = pd.read_csv(r"Y:\sorger\data\computation\Yu-An\YC-20230126-squidpy_demo\dataC04.csv")
jdf = jdf.set_index('CellID')

jdf['tYt'] = np.around(jdf.Yt / 25).astype(int)
jdf['tXt'] = np.around(jdf.Xt / 25).astype(int)






import skimage.filters
import skimage.segmentation
jdf = pd.read_csv(r"Y:\sorger\data\computation\Yu-An\YC-20230126-squidpy_demo\dataC04.csv")

jdf['tYt'] = (jdf.Yt / 25).astype(int)
jdf['tXt'] = (jdf.Xt / 25).astype(int)

tb = skimage.filters.gaussian((topic_count(jdf, 'Region', 2)>0).astype(float), sigma=1)
figure(); imshow(skimage.segmentation.find_boundaries(tb>0.3, mode='inner'))

ti = topic_count(jdf, 'Region', 1) > 0
tb = topic_count(jdf, 'Region', 2) > 0
np.where(ti, False, tb)



t7 = jdf.query('topics == 7').groupby(['tXt', 'tYt']).topics.count()

import scipy.stats
timg = np.zeros((300, 300))
timg[_18.index.get_level_values('tYt'), t7.index.get_level_values('tXt')] = t7.values


scipy.stats.gaussian_kde(t7.index.to_frame().values.T, weights=t7.values)
e7 = scipy.stats.gaussian_kde(t7.index.to_frame().values.T, weights=t7.values)
Z = e7(np.mgrid[:300, :300].reshape(2, -1)[::-1])
figure(); imshow(Z.reshape(300, 300), cmap='gist_earth_r')
scatter(*t7.index.to_frame().values.T, linewidths=0, s=8, c='k')
