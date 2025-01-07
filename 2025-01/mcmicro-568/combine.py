import tifffile
import skimage.segmentation
import numpy as np
import skimage.transform


img = tifffile.imread(
    r"Z:\computation\YC-20250105-mcmicro-issue-568-s3seg\crop2.ome.tif", key=0
)
mask_paths = [
    r"Z:\computation\YC-20250105-mcmicro-issue-568-s3seg\mcmicro\mcmicro-issue-568-crop2\segmentation\crop2\nucleiRing.ome.tif",
    r"Z:\computation\YC-20250105-mcmicro-issue-568-s3seg\masks\crop2-mask-stardist.tif",
    r"Z:\computation\YC-20250105-mcmicro-issue-568-s3seg\masks\crop2-mask-cellpose.tif",
]
masks = [tifffile.imread(pp) for pp in mask_paths]
contours = [skimage.segmentation.find_boundaries(mm, mode="inner") for mm in masks]

max_value = (img.max() * 0.8).astype(img.dtype)
data = np.array([img, *[mm.astype(img.dtype) * max_value for mm in contours]])
subresolutions = np.ceil(max(img.shape) / 1024).astype("int") - 1

pixelsize = 0.325  # micrometer
channel_names = ["Hoechst", "unmicst-s3seg", "stardist", "cellpose"]
with tifffile.TiffWriter(
    r"Z:\computation\YC-20250105-mcmicro-issue-568-s3seg\masks\all-results.ome.tif",
    bigtiff=True,
) as tif:
    metadata = {
        "PhysicalSizeX": pixelsize,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": pixelsize,
        "PhysicalSizeYUnit": "µm",
        "Channel": {"Name": channel_names},
    }
    options = dict(
        tile=(1024, 1024),
        compression="zlib",
        resolutionunit="CENTIMETER",
    )
    tif.write(
        data,
        subifds=subresolutions,
        resolution=(1e4 / pixelsize, 1e4 / pixelsize),
        metadata=metadata,
        **options,
    )
    # write pyramid levels to the two subifds
    # in production use resampling to generate sub-resolution images
    for level in range(subresolutions):
        mag = 2 ** (level + 1)
        tif.write(
            np.array(
                [
                    np.floor(skimage.transform.downscale_local_mean(cc, mag))
                    for cc in data
                ]
            ).astype(img.dtype),
            subfiletype=1,
            resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize),
            **options,
        )
