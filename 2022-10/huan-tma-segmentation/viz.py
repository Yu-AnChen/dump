import tifffile
import napari
import palom

codex = tifffile.imread(r"Y:\sorger\data\computation\Yu-An\YC-20221013-huan_tma_segmentation\2022_tma_1-pmap.tif", key=[0, 1])

reader = palom.reader.OmePyramidReader(r"Y:\sorger\data\computation\Yu-An\YC-20221013-huan_tma_segmentation\2022_tma_1-f9_nucleiRing.ome.tif")

v = napari.Viewer()

v.add_image([codex[:, ::4**i, ::4**i] for i in range(4)], channel_axis=0)
v.add_labels(reader.pyramid, blending='additive')