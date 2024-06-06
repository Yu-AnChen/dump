import pathlib

import KDEpy
import matplotlib.cm
import numpy as np
import ome_types
import pandas as pd
import skimage.util
import tifffile


def hist_2d(
    df,
    bin_size,
    query=None,
    spatial_x_name="Xt",
    spatial_y_name="Yt",
    kde=False,
    kde_kwargs=None,
):
    # snap coordinates to grid
    df_coords = df[[spatial_y_name, spatial_x_name]] / bin_size
    df_coords = df_coords.apply(lambda x: np.floor(x).astype("int"))

    # set output image size
    h, w = df_coords.max() + 1
    # query by condition string
    if query is not None:
        df_coords = df_coords[df.eval(query)]
    counts = df_coords.groupby([spatial_y_name, spatial_x_name]).size()
    y, x = counts.index.to_frame().values.T
    img = np.zeros((h, w))
    img[y, x] = counts

    if kde:
        img_kde = _kde_2d(img, kde_kwargs)
        return img, img_kde

    return img


def _kde_2d(img, kde_kwargs=None):
    assert img.ndim == 2
    rs, cs = img.nonzero()
    values = img[rs, cs]

    h, w = img.shape
    if kde_kwargs is None:
        kde_kwargs = {}
    kde = KDEpy.FFTKDE(**kde_kwargs)
    # 1. FIXME does it need to be shifted by half pixel?
    # 2. ValueError: Every data point must be inside of the grid.
    grid_pts = np.mgrid[-1 : h + 1, -1 : w + 1].reshape(2, -1).T
    return (
        kde.fit(np.vstack([rs, cs]).T, weights=values)
        .evaluate(grid_pts)
        .reshape(h + 2, w + 2)[1:-1, 1:-1]
    )


out_dir = pathlib.Path(r"U:\YC-20230126-squidpy_demo")
bin_size = 100
for ii in range(1, 21):
    case_number = f"C{ii:02}"

    df = pd.read_csv(rf"U:\YC-20230126-squidpy_demo\data{case_number}.csv")
    jdf = df.set_index("CellID")
    timgs = [
        hist_2d(jdf, bin_size=bin_size, kde=False, query=f"topics == {i}")
        for i in range(1, 13)
    ]
    timgs = [np.zeros_like(timgs[0])] + timgs
    print(
        f"data{case_number}.csv",
        bin_size,
        np.all((np.array(timgs) > 0).sum(axis=0) <= 1),
    )
    assert np.all((np.array(timgs) > 0).sum(axis=0) <= 1)

    tifffile.imwrite(
        out_dir / f"{case_number}-topics.ome.tif", np.array(timgs).astype("uint16")
    )
    ome = ome_types.from_tiff(out_dir / f"{case_number}-topics.ome.tif")

    colors = skimage.util.img_as_ubyte(matplotlib.cm.tab20(np.arange(12)))
    colors = np.vstack([[[0, 0, 0, 255]], colors])
    for idx, (cc, co) in enumerate(zip(ome.images[0].pixels.channels, colors)):
        cc.name = f"Topic {idx}"
        if idx == 0:
            cc.name = "Background"
        cc.color = ome_types.model.Color(tuple(co[:3]))

    ome.images[0].pixels.physical_size_x = bin_size
    ome.images[0].pixels.physical_size_y = bin_size

    tifffile.tiffcomment(
        out_dir / f"{case_number}-topics.ome.tif", ome.to_xml().encode()
    )
    # NOTE: pixel size calculation is wrong for C1, C2, C3, C4 - it was using
    # 0.320 μm/pixel instead of 0.325 μm/pixel. Need to fix the topic image
    # metadata afterwards

    # 0.325 / 0.320 μm/pixel instead of 1 μm/pixel


r"""
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


r"""
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
