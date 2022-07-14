import pandas as pd
import numpy as np
import tabbi
import omero_roi


# load single-cell quantifacation table
ss = pd.read_csv(r'W:\cycif-production\109-coukos-ovarian\mcmicro\Z236_13\quantification\unmicst-Z236_13_cellRingMask.csv', index_col='CellID')

# load ROI table exported from OMERO
_roi_table = pd.read_csv(r'W:\cycif-production\109-coukos-ovarian\demo-roi\rois\13121 3A.ome.tif-1735-rois.csv')

# set patch properties for visualization
patch_kwargs = dict(edgecolor='red', facecolor='none')

# add 'mpatch' column to the returned table
roi_table = (
    _roi_table
    .apply(omero_roi.add_mpatch, axis=1, patch_kwargs=patch_kwargs)
    .assign(has_exclude=_roi_table.Text.map(lambda x: 'exclude' in x))
    .sort_values(['has_exclude', 'Text'])
)

# precompute bounding box for better performance
roi_table.loc[:, ['x_min', 'y_min', 'x_max', 'y_max']] = np.hstack([
    [*roi_table['parsed_points'].apply(np.min, axis=0)],
    [*roi_table['parsed_points'].apply(np.max, axis=0)]
])

# add 'inside_mask' column to the table
roi_table.loc[:, 'inside_mask'] = roi_table.apply(
    omero_roi.compute_inside_mask, df=ss, parallel=True, axis=1
)

mask_exclude = np.vstack(
    roi_table.inside_mask.iloc[3:].values
).sum(axis=0) > 0
mask_include = roi_table.inside_mask.iloc[1] & (~mask_exclude)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow([[0]])
plt.scatter(ss.X_centroid, ss.Y_centroid, s=1, linewidths=0, c=mask_include)




import datashader as ds

tabbi.scatter.datashader_tissue_scatter(
    ss.assign(var=pd.Series(roi_table.inside_mask.iloc[0]).astype('category')),
    kwargs=dict(aggregator=ds.count_cat('var'), cmap='viridis')
)

tabbi.scatter.datashader_tissue_scatter(
    ss.assign(var=pd.Series(mask_include).astype('category')),
    kwargs=dict(aggregator=ds.count_cat('var'), cmap='viridis')
)

tabbi.gmm.plot_hist_gmm(ss, ['CD3e', 'panCK', 'CD8a'], n_components=3)
tabbi.scatter.plot_pair(ss, 'CD3e', 'panCK', gates=(2603, 652))

tabbi.scatter.datashader_tissue_scatter(ss.assign(test=np.log1p(ss.panCK)), kwargs=dict(aggregator=ds.mean('test')))
