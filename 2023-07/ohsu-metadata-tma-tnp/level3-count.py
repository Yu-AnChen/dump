import pathlib

import joblib
import numpy as np
import pandas as pd
import tifffile


# ---------------------------------------------------------------------------- #
#                                  for phase 1                                 #
# ---------------------------------------------------------------------------- #
curr = pathlib.Path(r'Y:\sorger\data\computation\Yu-An\temp-ohsu')
masks = sorted(curr.glob('*/dearray/*/*Mask.tif'))

def num_obj(mask_path):
    img = tifffile.imread(mask_path)
    return np.unique(img).size - 1

counts = joblib.Parallel(n_jobs=-1, verbose=10)(
    joblib.delayed(num_obj)(pp) for pp in masks
)
names = ['_'.join((pp.parts[-4], pp.parts[-2], pp.parts[-1])) for pp in masks]

pd.DataFrame({'Filename': names, 'count': counts}).to_csv('mask-object-count.csv', index=False)



curr = pathlib.Path(r'X:\cycif-production\111-TMA-TNP\168-OHSU_TMATNP-2020SEP\transfer\OHSU_TMA1_011\level3')
masks = sorted(curr.glob('*.ome.tif'))

counts = joblib.Parallel(n_jobs=-1, verbose=10)(
    joblib.delayed(num_obj)(pp) for pp in masks
)
names = [pp.name for pp in masks]

pd.DataFrame({'Filename': names, 'count': counts}).to_csv('mask-object-count-011.csv', index=False)


# ---------------------------------------------------------------------------- #
#                                  for phase 3                                 #
# ---------------------------------------------------------------------------- #
curr = pathlib.Path(r'X:\cycif-production\111-TMA-TNP\199-OMS_Atlas_TMA-2021NOV')
masks = sorted(curr.glob('LSP*/level3/*.ome.tif'))

counts = joblib.Parallel(n_jobs=-1, verbose=10)(
    joblib.delayed(num_obj)(pp) for pp in masks
)
names = [pp.name for pp in masks]
pd.DataFrame({'Filename': names, 'count': counts}).to_csv('mask-object-phase-3.csv', index=False)