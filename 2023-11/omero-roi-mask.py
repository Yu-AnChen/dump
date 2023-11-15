import base64

import numpy as np
import ome_types


def bin_data_to_mask(data_str, width, height):
    data_b64_bytes = base64.b64decode(data_str)
    data_array = np.frombuffer(data_b64_bytes, dtype="uint8")
    data_unpacked = np.unpackbits(data_array)
    if (width is not None) & (height is not None):
        return np.resize(data_unpacked, (height, width))
    return data_unpacked


def mask_roi_to_img(roi):
    w = int(roi.width)
    h = int(roi.height)
    return bin_data_to_mask(roi.bin_data.value, w, h)


img_path = r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\p110_e4_OvarianSTIC_GeoMx\GeoMx_OME_tiff\LSP15356_Scan1.ome.tiff"
ome = ome_types.from_tiff(img_path)