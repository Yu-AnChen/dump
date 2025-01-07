import numpy as np
import skimage.exposure
import tifffile

import cellpose.denoise

img = tifffile.imread(
    r"Z:\computation\YC-20250105-mcmicro-issue-568-s3seg\crop2.ome.tif", key=0
)
in_range = np.percentile(img, [0, 99.9])
img_float = skimage.exposure.rescale_intensity(
    img, in_range=tuple(in_range), out_range="float"
)
dn_model = cellpose.denoise.CellposeDenoiseModel(
    gpu=True,
    model_type="cyto3",
    restore_type="deblur_cyto3",
)
mask = dn_model.eval(
    img_float,
    diameter=20.0,
    channels=[0, 0],
    # inputs are globally normalized already
    normalize=False,
    flow_threshold=0.6,
    tile=True,
)[0]
tifffile.imwrite(
    r"Z:\computation\YC-20250105-mcmicro-issue-568-s3seg\cellpose\crop2-mask-cellpose.tif",
    mask.astype("int32"),
    tile=(1024, 1024),
    compression="zlib",
)
