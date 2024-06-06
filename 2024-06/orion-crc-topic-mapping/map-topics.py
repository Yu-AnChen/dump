import napari
import palom
import numpy as np
import tifffile
import ome_types
import skimage.transform
import matplotlib.pyplot as plt


def run(topic_img_path, orion_path, cycif_path, topic_img_factor=100):
    timgs = tifffile.imread(topic_img_path)
    r1 = palom.reader.OmePyramidReader(orion_path)
    r2 = palom.reader.OmePyramidReader(cycif_path)

    aligner = palom.align.get_aligner(r1, r2, thumbnail_channel2=None)
    aligner.coarse_register_affine(
        n_keypoints=30_000,
        test_flip=True,
        test_intensity_invert=False,
        auto_mask=True,
    )
    plt.close(plt.gcf())
    timgs_rescaled = skimage.transform.rescale(
        timgs,
        topic_img_factor / (aligner.ref_thumbnail_down_factor * r1.pixel_size),
        channel_axis=0,
    )
    ref_level = {vv: kk for kk, vv in r1.level_downsamples.items()}[
        aligner.ref_thumbnail_down_factor
    ]
    ref_imgs = r1.pyramid[ref_level].compute()
    h, w = np.array([ref_imgs.shape[1:], timgs_rescaled.shape[1:]]).min(axis=0)
    _weighted_sum = [
        (ref_imgs[:, :h, :w] * tt[:h, :w]).sum(axis=(1, 2)) for tt in timgs_rescaled
    ]

    _topic_sum = [
        (ref_imgs[:, :h, :w] * (tt[:h, :w] > 0)).sum(axis=(1, 2))
        for tt in timgs_rescaled
    ]
    moving_level = {vv: kk for kk, vv in r2.level_downsamples.items()}[
        aligner.moving_thumbnail_down_factor
    ]
    moving_imgs = r2.pyramid[moving_level].compute()
    moving_imgs = np.array(
        [
            skimage.transform.warp(
                cc,
                skimage.transform.AffineTransform(
                    matrix=aligner.coarse_affine_matrix
                ).inverse,
            )
            for cc in moving_imgs
        ]
    )
    h2, w2 = np.array([moving_imgs.shape[1:], timgs_rescaled.shape[1:]]).min(axis=0)

    _, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    axs[0].imshow(timgs_rescaled.sum(axis=0)[:h, :w])
    axs[1].imshow(ref_imgs[0][:h, :w])
    axs[2].imshow(moving_imgs[0][:h2, :w2])
    plt.close(plt.gcf())
    weighted_sum2 = [
        (moving_imgs[:, :h2, :w2] * tt[:h2, :w2]).sum(axis=(1, 2))
        for tt in timgs_rescaled
    ]

    topic_sum2 = [
        (moving_imgs[:, :h2, :w2] * (tt[:h2, :w2] > 0)).sum(axis=(1, 2))
        for tt in timgs_rescaled
    ]
    return weighted_sum2


timg_paths = r"""
U:\YC-20230126-squidpy_demo\C01-topics.ome.tif
U:\YC-20230126-squidpy_demo\C02-topics.ome.tif
U:\YC-20230126-squidpy_demo\C03-topics.ome.tif
U:\YC-20230126-squidpy_demo\C04-topics.ome.tif
U:\YC-20230126-squidpy_demo\C06-topics.ome.tif
U:\YC-20230126-squidpy_demo\C07-topics.ome.tif
U:\YC-20230126-squidpy_demo\C08-topics.ome.tif
U:\YC-20230126-squidpy_demo\C09-topics.ome.tif
U:\YC-20230126-squidpy_demo\C10-topics.ome.tif
""".strip().split("\n")
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
timg_factor = [
    100*0.325/0.320,
    100*0.325/0.320,
    100*0.325/0.320,
    100*0.325/0.320,
    100,
    100,
    100,
    100,
    100
]



jerry_paths = r"""
U:\YC-20230126-squidpy_demo\dataC01.csv
U:\YC-20230126-squidpy_demo\dataC02.csv
U:\YC-20230126-squidpy_demo\dataC03.csv
U:\YC-20230126-squidpy_demo\dataC04.csv
U:\YC-20230126-squidpy_demo\dataC05.csv
U:\YC-20230126-squidpy_demo\dataC06.csv
U:\YC-20230126-squidpy_demo\dataC07.csv
U:\YC-20230126-squidpy_demo\dataC08.csv
U:\YC-20230126-squidpy_demo\dataC09.csv
U:\YC-20230126-squidpy_demo\dataC10.csv
""".strip().split("\n")

mcmicro_paths = r"""
W:\crc-scans\C1-C40-sc-tables\P37_S29-CRC01
W:\crc-scans\C1-C40-sc-tables\P37_S30-CRC02
W:\crc-scans\C1-C40-sc-tables\P37_S31-CRC03
W:\crc-scans\C1-C40-sc-tables\P37_S32-CRC04
W:\crc-scans\C1-C40-sc-tables\P37_S33-CRC05
W:\crc-scans\C1-C40-sc-tables\P37_S34-CRC06
W:\crc-scans\C1-C40-sc-tables\P37_S35-CRC07
W:\crc-scans\C1-C40-sc-tables\P37_S57-CRC08
W:\crc-scans\C1-C40-sc-tables\P37_S37-CRC09
W:\crc-scans\C1-C40-sc-tables\P37_S38-CRC10
""".strip().split("\n")

run(
    r"U:\YC-20230126-squidpy_demo\C09-topics.ome.tif",
    r"Z:\RareCyte-S3\P37_CRCstudy_Round1\P37_S37_A24_C59kX_E15@20220108_012113_953544-zlib.ome.tiff",
    r"Z:\JL503_JERRY\192-CRCWSI_Tumor-2021JUL\TNPCRC_10.ome.tif",
)


timgs = tifffile.imread(r"U:\YC-20230126-squidpy_demo\C09-topics.ome.tif")
bin_size = 100

r1 = palom.reader.OmePyramidReader(
    r"Z:\RareCyte-S3\P37_CRCstudy_Round1\P37_S37_A24_C59kX_E15@20220108_012113_953544-zlib.ome.tiff"
)


v = napari.Viewer()

v.add_image([pp[0] for pp in r1.pyramid], blending="additive", visible=False)
r2 = palom.reader.OmePyramidReader(
    r"Z:\JL503_JERRY\192-CRCWSI_Tumor-2021JUL\TNPCRC_10.ome.tif"
)
aligner = palom.align.get_aligner(r1, r2, thumbnail_channel2=None)
aligner.coarse_register_affine(
    n_keypoints=10_000,
    test_flip=True,
    test_intensity_invert=False,
    auto_mask=True,
)
v.add_image(
    [pp[0] for pp in r2.pyramid],
    blending="additive",
    visible=False,
    affine=palom.img_util.to_napari_affine(aligner.affine_matrix),
)

v.add_image(
    r2.pyramid,
    visible=False,
    channel_axis=0,
    name=[cc.name for cc in ome_types.from_tiff(r2.path).images[0].pixels.channels],
    affine=palom.img_util.to_napari_affine(aligner.affine_matrix),
)

v.add_image(
    timgs[7],
    scale=(bin_size / 0.325,) * 2,
    translate=(bin_size / 0.325) * np.array((0.5, 0.5)),
    name="topic 7",
)
v.add_image(
    timgs[8],
    scale=(bin_size / 0.325,) * 2,
    translate=(bin_size / 0.325) * np.array((0.5, 0.5)),
    name="topic 8",
)


import skimage.transform

timgs_rescaled = skimage.transform.rescale(
    timgs,
    100 / (aligner.ref_thumbnail_down_factor * r1.pixel_size),
    channel_axis=0,
)
# FIXME
ref_imgs = r1.pyramid[7].compute()
h, w = np.array([ref_imgs.shape[1:], timgs_rescaled.shape[1:]]).min(axis=0)

weighted_sum = [
    (ref_imgs[:, :h, :w] * tt[:h, :w]).sum(axis=(1, 2)) for tt in timgs_rescaled
]

topic_sum = [
    (ref_imgs[:, :h, :w] * (tt[:h, :w] > 0)).sum(axis=(1, 2)) for tt in timgs_rescaled
]
# FIXME
moving_imgs = r2.pyramid[6].compute()
moving_imgs = np.array(
    [
        skimage.transform.warp(
            cc,
            skimage.transform.AffineTransform(
                matrix=aligner.coarse_affine_matrix
            ).inverse,
        )
        for cc in moving_imgs
    ]
)
h2, w2 = np.array([moving_imgs.shape[1:], timgs_rescaled.shape[1:]]).min(axis=0)

weighted_sum2 = [
    (moving_imgs[:, :h2, :w2] * tt[:h2, :w2]).sum(axis=(1, 2)) for tt in timgs_rescaled
]

topic_sum2 = [
    (moving_imgs[:, :h2, :w2] * (tt[:h2, :w2] > 0)).sum(axis=(1, 2))
    for tt in timgs_rescaled
]
