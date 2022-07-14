import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.spatial.distance as sdistance


def parse_roi_points(all_points):
    return np.array(
        re.findall(r'-?\d+\.?\d+', all_points), dtype=float
    ).reshape(-1, 2)

def ellipse_points_to_patch(
    vertex_1, vertex_2,
    co_vertex_1, co_vertex_2,
    patch_kwargs={}
):
    """
    Parameters
    ----------
    vertex_1, vertex_2, co_vertex_1, co_vertex_2: array like, in the form of (x-coordinate, y-coordinate)

    """
    v_and_co_v = np.array([
        vertex_1, vertex_2,
        co_vertex_1, co_vertex_2
    ])
    centers = v_and_co_v.mean(axis=0)

    d = sdistance.cdist(v_and_co_v, v_and_co_v, metric='euclidean')
    width = d[0, 1]
    height = d[2, 3]

    vector_2 = v_and_co_v[1] - v_and_co_v[0]
    vector_2 /= np.linalg.norm(vector_2)

    angle = np.degrees(np.arccos([1, 0] @ vector_2))

    ellipse_patch = mpatches.Ellipse(
        centers, width=width, height=height, angle=angle,
        **patch_kwargs
    )
    return ellipse_patch

def add_mpatch(roi, patch_kwargs={}):
    roi = roi.copy()
    points = parse_roi_points(roi['all_points'])
    roi.loc['parsed_points'] = points

    roi_type = roi['type']
    if roi_type in ['Point', 'Line']:
        roi_mpatch = mpatches.Polygon(points, closed=False, **patch_kwargs)
    elif roi_type in ['Rectangle', 'Polygon', 'Polyline']:
        roi_mpatch = mpatches.Polygon(points, closed=True, **patch_kwargs)
    elif roi_type == 'Ellipse':
        roi_mpatch = ellipse_points_to_patch(*points, patch_kwargs=patch_kwargs)
    else:
        raise ValueError

    roi.loc['mpatch'] = roi_mpatch
    return roi

import dask.dataframe as dd
import dask.diagnostics
def compute_inside_mask(roi, df, parallel=False):
    sub_df = df[['X_centroid', 'Y_centroid']][
        df.X_centroid.between(roi.x_min, roi.x_max, inclusive='both') * 
        df.Y_centroid.between(roi.y_min, roi.y_max, inclusive='both')
    ]
    if parallel & (sub_df.shape[0] > 100000):
        dd_sub_df = dd.from_pandas(sub_df, chunksize=10000)
        with dask.diagnostics.ProgressBar():
            inside = dd_sub_df.map_partitions(
                roi.mpatch.contains_points, meta=(None, bool)
            ).compute(scheduler='processes')
    else:
        inside = roi.mpatch.contains_points(sub_df)
    mask = df.X_centroid > df.X_centroid + 1
    mask.loc[sub_df[inside].index] = True
    return mask.values

def demo():
    # load single-cell quantifacation table
    ss = pd.read_csv('quantification/cellRingMask/Lung_061_CFX16a_Full_C11@20201024_195941_804220.csv', index_col=0)

    # load ROI table exported from OMERO
    _roi_table = pd.read_csv('Lung_061_CFX16a_Full_C11@20201024_195941_804220.ome.tiff-1327440-rois.csv')

    # set patch properties for visualization
    patch_kwargs = dict(edgecolor='red', facecolor='none')

    # add 'mpatch' column to the returned table
    roi_table = _roi_table.apply(add_mpatch, axis=1, patch_kwargs=patch_kwargs)

    # precompute bounding box for better performance
    roi_table.loc[:, ['x_min', 'y_min', 'x_max', 'y_max']] = np.hstack([
        [*roi_table['parsed_points'].apply(np.min, axis=0)],
        [*roi_table['parsed_points'].apply(np.max, axis=0)]
    ])

    # add 'inside_mask' column to the table
    roi_table.loc[:, 'inside_mask'] = roi_table.apply(
        compute_inside_mask, df=ss, parallel=True, axis=1
    )
    inside_mask_all = np.logical_or.reduce(roi_table['inside_mask'].to_list())

    plt.figure()
    plt.scatter(
        ss['X_centroid'], ss['Y_centroid'], 
        marker='.', linewidths=0, s=2, c=inside_mask_all
    )
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    for p in roi_table['mpatch']:
        plt.gca().add_patch(p)







