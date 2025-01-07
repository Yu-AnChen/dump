import json

import cv2
import numpy as np
import shapely
import tifffile


def shapely_polygons_to_mask(polygons, mask_shape=None, fill_value_min=None):
    coords = [np.array(pp.exterior.coords) for pp in polygons]

    if mask_shape is None:
        mask_shape = np.ceil(np.vstack(coords).max(axis=0)).astype("int")[::-1]
    h, w = mask_shape

    if fill_value_min is None:
        fill_value_min = 1
    fill_value = int(fill_value_min)

    mask = np.zeros((int(h), int(w)), "int32")
    for pp in coords:
        cv2.fillPoly(mask, [pp.round().astype("int")], fill_value)
        fill_value += 1
    return mask


stardist_json = json.load(
    open(r"Z:\computation\YC-20250105-mcmicro-issue-568-s3seg\crop2.geojson")
)
polygons = [
    shapely.from_geojson(json.dumps(ff["geometry"]))
    for ff in stardist_json["features"]
    if ff["properties"]["objectType"] == "detection"
]
img = tifffile.imread(
    r"Z:\computation\YC-20250105-mcmicro-issue-568-s3seg\crop2.ome.tif", key=0
)
mask = shapely_polygons_to_mask(polygons, img.shape)
tifffile.imwrite(
    r"Z:\computation\YC-20250105-mcmicro-issue-568-s3seg\masks\crop2-mask-stardist.tif",
    mask.astype("int32"),
    tile=(1024, 1024),
    compression="zlib",
)
