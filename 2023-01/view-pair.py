import palom.reader
import napari
import numpy as np
import pathlib
import csv
import fire


CHANNEL_NAMES = 'Hoechst,AF1,CD31,Ki-67,CD68,CD163,CD20,CD4,CD8a,CD45RO,PD-L1,CD3e,E-Cadherin,PD-1,FOXP3,CD45,Pan-CK,Blank,SMA'.split(',')
CONTRAST_LIMITS = (200, 2000)


def view_if_pyramid(path, viewer=None):
    reader = palom.reader.OmePyramidReader(path)
    v = napari.Viewer() if viewer is None else viewer
    v.add_image(
        reader.pyramid, channel_axis=0, visible=False,
        name=CHANNEL_NAMES, contrast_limits=CONTRAST_LIMITS
    )
    return viewer


def view_he_pyramid(path, viewer=None):
    reader = palom.reader.OmePyramidReader(path)
    name = pathlib.Path(path).stem
    v = napari.Viewer() if viewer is None else viewer
    v.add_image(
        [np.moveaxis(pp, 0, 2) for pp in reader.pyramid],
        visible=True,
        name=name
    )
    return viewer


def view_pair(if_path, he_path):
    v = napari.Viewer()
    v = view_if_pyramid(if_path, viewer=v)
    v = view_he_pyramid(he_path, viewer=v)
    return v


if __name__ == '__main__':
    fire.Fire(view_pair)
    napari.run()