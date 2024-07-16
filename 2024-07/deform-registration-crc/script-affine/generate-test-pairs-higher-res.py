import palom
import skimage.transform
import tifffile
import numpy as np

# ---------------------------------------------------------------------------- #
#                                   C1 - C10                                   #
# ---------------------------------------------------------------------------- #
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


# ---------------------------------------------------------------------------- #
#                                   C17 - C40                                  #
# ---------------------------------------------------------------------------- #
# C22 needed pyramid level 6, 5 and 30_000 key points due to larger tissue
# folding (others use pyramid 7, 6 and 20_000 key points)
orion_paths = r"""
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round2\P37_S49_Full_A24_C59mX_E15@20220129_015121_911264-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round2\P37_S50_Full_A24_C59mX_E15@20220129_015242_755602-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round2\P37_S51_Full_A24_C59mX_E15@20220129_015300_669681-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round2\P37_S52_Full_A24_C59mX_E15@20220129_015324_574779-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round3\P37_S58_Full_A24_C59nX_E15@20220224_011058_014787-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round3\P37_S59_Full_A24_C59nX_E15@20220224_011113_455637-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round3\P37_S60_Full_A24_C59nX_E15@20220224_011127_971497-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round3\P37_S61_Full_A24_C59nX_E15@20220224_011149_079291-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round3\P37_S62_Full_A24_C59nX_E15@20220224_011204_784145-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round3\P37_S63_Full_A24_C59nX_E15@20220224_011246_458738-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round3\P37_S64_Full_A24_C59nX_E15@20220224_011259_841605-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round3\P37_S65_Full_A24_C59nX_E15@20220224_011333_386280-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round3\P37_S66_Full_A24_C59nX_E15@20220224_011348_519133-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round3\P37_S67_Full_A24_C59nX_E15@20220224_011408_506939-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round4\P37_S74_Full_A24_C59qX_E15@20220302_234837_137590-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round4\P37_S75_Full_A24_C59qX_E15@20220302_235001_586560-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round4\P37_S76_01_A24_C59qX_E15@20220302_235136_561323-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round4\P37_S77_Full_A24_C59qX_E15@20220302_235222_359806-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round4\P37_S78_Full_A24_C59qX_E15@20220302_235239_498836-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round4\P37_S79_Full_A24_C59qX_E15@20220302_235254_496641-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round4\P37_S80_Full_A24_C59qX_E15@20220307_235159_333000-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round4\P37_S81_Full_A24_C59qX_E15@20220302_235331_704703-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round4\P37_S82_Full_A24_C59qX_E15@20220304_200614_832683-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round4\P37_S83_Full_A24_C59qX_E15@20220304_200429_490805-zlib.ome.tiff
\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3\P37_CRCstudy_Round4\P37_S76_02_A24_C59qX_E15@20220302_235158_533766-zlib.ome.tiff
""".strip().split("\n")

cycif_paths = r"""
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10532.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10543.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10554.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10566.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10576.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10587.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10598.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10609.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10621.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10631.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10642.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10653.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10664.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10675.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10687.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10697.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10708.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10719.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10730.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10741.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10752.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10763.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10774.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10790.ome.tif
V:\hits\lsp\collaborations\lsp-analysis\cycif-production\221-CRC_ORION-2022APR\LSP10708.ome.tif
""".strip().split("\n")

case_number = list(range(17, 41)) + [332]


for oo, cc, nn in zip(orion_paths[5:6], cycif_paths[5:6], case_number[5:6]):
    r1 = palom.reader.OmePyramidReader(oo)
    r2 = palom.reader.OmePyramidReader(cc)

    aligner = palom.align.Aligner(
        ref_img=r1.pyramid[5][0],
        moving_img=r2.pyramid[4][0],
        # do feature detection and matching at resolution of ~40 µm/pixel
        ref_thumbnail=palom.img_util.cv2_downscale_local_mean(r1.pyramid[6][0], 2),
        moving_thumbnail=palom.img_util.cv2_downscale_local_mean(r2.pyramid[5][0], 2),
        ref_thumbnail_down_factor=2 ** (7 - 5),
        moving_thumbnail_down_factor=2 ** (6 - 4),
    )
    aligner.coarse_register_affine(
        n_keypoints=20_000,
        test_flip=True,
        test_intensity_invert=False,
        auto_mask=True,
    )

    ref = aligner.ref_img.compute()
    img = aligner.moving_img.compute()
    moving = skimage.transform.warp(
        img,
        skimage.transform.AffineTransform(matrix=aligner.affine_matrix).inverse,
        preserve_range=True,
        output_shape=ref.shape,
    )
    np.savetxt(f"C{nn:02}-affine-matrix.csv", aligner.affine_matrix, delimiter=",")
    tifffile.imwrite(f"C{nn:02}-ref.tif", ref, compression="zlib")
    tifffile.imwrite(
        f"C{nn:02}-moving.tif",
        np.floor(moving).astype(r2.pyramid[0].dtype),
        compression="zlib",
    )


# ---------------------------------------------------------------------------- #
#                         warp with displacement field                         #
# ---------------------------------------------------------------------------- #

# scikit-image use "float64" internally
my, mx = np.mgrid[: ref.shape[0], : ref.shape[1]].astype("float32")

tmy = skimage.transform.warp(
    my,
    skimage.transform.AffineTransform(matrix=aligner.affine_matrix).inverse,
    preserve_range=True,
)

tmx = skimage.transform.warp(
    mx,
    skimage.transform.AffineTransform(matrix=aligner.affine_matrix).inverse,
    preserve_range=True,
)

warpped_moving_image = skimage.transform.warp(
    img,
    np.array([tmy, tmx]),
    preserve_range=True,
)
