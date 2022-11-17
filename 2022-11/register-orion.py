import palom
import pathlib
import numpy as np


files = [
    r'Z:\RareCyte-S3\P54_WD76847_51_P54_A31_C100_HMS_Orion7@20221027_231411_492648\P54_WD76847_51_P54_A31_C100_HMS_Orion7@20221027_231411_492648.ome.tiff',
    r'Z:\RareCyte-S3\P54_WD76847_51_B1_P54_A31_C100_HMS_Orion7@20221110_153331_662526\P54_WD76847_51_B1_P54_A31_C100_HMS_Orion7@20221110_153331_662526.ome.tiff',
    r'Z:\RareCyte-S3\P54_WD76847_51_B2_P54_A31_C100_HMS_Orion7@20221111_125351_889500\P54_WD76847_51_B2_P54_A31_C100_HMS_Orion7@20221111_125351_889500.ome.tiff',
    r'Z:\RareCyte-S3\P54_WD76847_51_RE1_P54_A31_C100_HMS_Orion7@20221116_210604_910821\P54_WD76847_51_RE1_P54_A31_C100_HMS_Orion7@20221116_210604_910821.ome.tiff',
]

files = [pathlib.Path(ff) for ff in files]
readers = [palom.reader.OmePyramidReader(ff) for ff in files]

LEVEL = 1
THUMBNAIL_LEVEL = 5

def get_aligner(c1r, c2r):
     
    return palom.align.Aligner( 
        c1r.read_level_channels(LEVEL, 0),  
        c2r.read_level_channels(LEVEL, 0), 
        ref_thumbnail=c1r.read_level_channels(THUMBNAIL_LEVEL, 0), 
        moving_thumbnail=c2r.read_level_channels(THUMBNAIL_LEVEL, 0), 
        ref_thumbnail_down_factor=c1r.level_downsamples[THUMBNAIL_LEVEL] / c1r.level_downsamples[LEVEL], 
        moving_thumbnail_down_factor=c2r.level_downsamples[THUMBNAIL_LEVEL] / c2r.level_downsamples[LEVEL] 
    ) 
 
aligners = [
    get_aligner(readers[0], rr)
    for rr in readers[1:]
]

block_affines = []
for aa in aligners:
    aa.coarse_register_affine()
    aa.compute_shifts()
    aa.constrain_shifts()

    block_affines.append(aa.block_affine_matrices_da)
    palom.align.viz_shifts(aa.original_shifts, aa.grid_shape)


mosaics = []
m_ref = readers[0].read_level_channels(LEVEL, slice(None))
mosaics.append(m_ref)
for rr, mx in zip(readers[1:], block_affines):
    m_moving = palom.align.block_affine_transformed_moving_img(
        ref_img=readers[0].read_level_channels(LEVEL, 0),
        moving_img=rr.read_level_channels(LEVEL, slice(None)),
        mxs=mx
    )
    mosaics.append(m_moving)

palom.pyramid.write_pyramid(
    mosaics=palom.pyramid.normalize_mosaics(mosaics),
    output_path=r"Z:\RareCyte-S3\YC-analysis\P54_WD76847_51\P54_WD76847_51.ome.tif",
    pixel_size=0.325*readers[0].level_downsamples[LEVEL]
)

