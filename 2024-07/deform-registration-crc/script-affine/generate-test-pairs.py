import palom
import skimage.transform
import tifffile
import numpy as np

orion_paths = r"""
Z:\RareCyte-S3\P37_CRCstudy_Round1\P37_S29_A24_C59kX_E15@20220106_014304_946511-zlib.ome.tiff
Z:\RareCyte-S3\P37_CRCstudy_Round1\P37_S30_A24_C59kX_E15@20220106_014319_409148-zlib.ome.tiff
Z:\RareCyte-S3\P37_CRCstudy_Round1\P37_S31_A24_C59kX_E15@20220106_014409_014236-zlib.ome.tiff
Z:\RareCyte-S3\P37_CRCstudy_Round1\P37_S32_A24_C59kX_E15@20220106_014630_553652-zlib.ome.tiff
Z:\RareCyte-S3\P37_CRCstudy_Round1\P37_S34_A24_C59kX_E15@20220107_202112_212579-zlib.ome.tiff
Z:\RareCyte-S3\P37_CRCstudy_Round1\P37_S35_A24_C59kX_E15@20220108_012037_490594-zlib.ome.tiff
Z:\RareCyte-S3\P37_CRCstudy_Round3\P37_S57_Full_A24_C59nX_E15@20220224_011032_774034-zlib.ome.tiff
Z:\RareCyte-S3\P37_CRCstudy_Round1\P37_S37_A24_C59kX_E15@20220108_012113_953544-zlib.ome.tiff
Z:\RareCyte-S3\P37_CRCstudy_Round1\P37_S38_A24_C59kX_E15@20220108_012130_664519-zlib.ome.tiff
""".strip().split("\n")

cycif_paths = r"""
Z:\JL503_JERRY\192-CRCWSI_Tumor-2021JUL\TNPCRC_01.ome.tif
Z:\JL503_JERRY\192-CRCWSI_Tumor-2021JUL\TNPCRC_02.ome.tif
Z:\JL503_JERRY\192-CRCWSI_Tumor-2021JUL\TNPCRC_03.ome.tif
Z:\JL503_JERRY\192-CRCWSI_Tumor-2021JUL\TNPCRC_04.ome.tif
Z:\JL503_JERRY\192-CRCWSI_Tumor-2021JUL\TNPCRC_06.ome.tif
Z:\JL503_JERRY\192-CRCWSI_Tumor-2021JUL\TNPCRC_08.ome.tif
Z:\JL503_JERRY\192-CRCWSI_Tumor-2021JUL\TNPCRC_09.ome.tif
Z:\JL503_JERRY\192-CRCWSI_Tumor-2021JUL\TNPCRC_10.ome.tif
Z:\JL503_JERRY\192-CRCWSI_Tumor-2021JUL\TNPCRC_11.ome.tif
""".strip().split("\n")

case_number = [1, 2, 3, 4, 6, 7, 8, 9, 10]

for oo, cc, nn in zip(orion_paths[:], cycif_paths, case_number):
    r1 = palom.reader.OmePyramidReader(oo)
    r2 = palom.reader.OmePyramidReader(cc)

    aligner = palom.align.get_aligner(r1, r2, thumbnail_level2=None)
    aligner.coarse_register_affine(
        n_keypoints=30_000,
        test_flip=True,
        test_intensity_invert=False,
        auto_mask=True,
    )

    ref = aligner.ref_thumbnail.compute()
    img = aligner.moving_thumbnail.compute()
    tform = skimage.transform.AffineTransform(aligner.coarse_affine_matrix)
    moving = skimage.transform.warp(
        img,
        skimage.transform.AffineTransform(matrix=aligner.coarse_affine_matrix).inverse,
        preserve_range=True,
        output_shape=ref.shape,
    )
    tifffile.imwrite(f"C{nn:02}-ref.tif", ref, compression="zlib")
    tifffile.imwrite(
        f"C{nn:02}-moving.tif",
        np.floor(moving).astype(r2.pyramid[0].dtype),
        compression="zlib",
    )
