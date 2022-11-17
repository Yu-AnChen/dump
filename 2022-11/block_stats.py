import palom 
import numpy as np
import skimage

def mms(regionmask, intensity):
    intensity = intensity[regionmask]
    return (
        intensity.mean(),
        np.median(intensity),
        intensity.std()
    )

import pandas as pd
def quantify(mask, img):
    results = skimage.measure.regionprops_table(
        np.asarray(mask),
        np.asarray(img),
        properties=('label', 'centroid', 'area'),
        extra_properties=[mms]
    )
    results = pd.DataFrame(results)
    results.set_index('label', inplace=True)
    results.rename(columns={
        'mms-0': 'mean',
        'mms-1': 'median',
        'mms-2': 'std',
        'centroid-0': 'Y_centroid',
        'centroid-1': 'X_centroid'
    }, inplace=True)
    return pd.DataFrame(results)



c1r = palom.reader.OmePyramidReader(r"Z:\RareCyte-S3\P37_CRCstudy_Round1\P37_S24_Full_A24_C59kX_E15@20220104_223653_988272.ome.tiff") 
c2r = palom.reader.OmePyramidReader(r"Z:\RareCyte-S3\P37_CRCstudy_Round2\P37_S39_Full_A24_C59mX_E15@20220128_010233_084078.ome.tiff") 
 
LEVEL = 0
THUMBNAIL_LEVEL = 4 
 
c21l = palom.align.Aligner( 
    c1r.read_level_channels(LEVEL, 0),  
    c2r.read_level_channels(LEVEL, 0), 
    ref_thumbnail=c1r.read_level_channels(THUMBNAIL_LEVEL, 0), 
    moving_thumbnail=c2r.read_level_channels(THUMBNAIL_LEVEL, 0), 
    ref_thumbnail_down_factor=c1r.level_downsamples[THUMBNAIL_LEVEL] / c1r.level_downsamples[LEVEL], 
    moving_thumbnail_down_factor=c2r.level_downsamples[THUMBNAIL_LEVEL] / c2r.level_downsamples[LEVEL] 
) 
 
 
c21l.coarse_register_affine() 


c2mask = palom.img_util.block_labeled_mask(
    c2r.pyramid[LEVEL].shape[1:], (100, 100), out_chunks=1024
)
c1mask = palom.align.block_affine_transformed_moving_img(
    c21l.ref_img.astype(np.int32),
    c2mask,
    c21l.affine_matrix,
    is_mask=True
)

img1 = c1r.read_level_channels(LEVEL, 0)
img2 = c2r.read_level_channels(LEVEL, 0)


import dask.diagnostics
with dask.diagnostics.ProgressBar():
    r1 = quantify(c1mask, img1)
    r2 = quantify(c2mask, img2)
r1r2 = r1.merge(r2, on='label')

area_mask = (r1r2.filter(like='area').min(axis=1) > 9000) & (r1r2.filter(like='mean').min(axis=1) > 5000)
mean1, mean2 = r1r2[area_mask].filter(like='mean').T.values


import sklearn.linear_model
import matplotlib.pyplot as plt

regr = sklearn.linear_model.LinearRegression()

plt.figure()
plt.scatter(np.sort(mean1), np.sort(mean2), s=4, linewidths=0, c='darkblue', alpha=0.2)
plt.scatter(mean1, mean2, s=4, linewidths=0, c='darkolivegreen', alpha=0.2)

regr.fit(np.sort(mean1).reshape(-1, 1), np.sort(mean2).reshape(-1, 1))
print('ranked', regr.coef_[0][0])
plt.plot([0, 16000], [regr.predict([[0]])[0], regr.predict([[16000]])[0]], c='dodgerblue', linestyle='--')

regr.fit(mean1.reshape(-1, 1), mean2.reshape(-1, 1))
print('registered', regr.coef_[0][0])
plt.plot([0, 16000], [regr.predict([[0]])[0], regr.predict([[16000]])[0]], c='yellowgreen', linestyle='--')