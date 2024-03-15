import copy
import itertools
import pathlib

import ome_types


def write_ori_omexml(img_path):
    ome = ome_types.from_tiff(img_path)
    img_path = pathlib.Path(img_path)
    with open(img_path.parent / f"{img_path.name.split('.')[0]}.ome.xml", "w") as f:
        f.write(ome.to_xml())


def update_xml_with_z(img_path, channel_names):
    img_path = pathlib.Path(img_path)
    ome = ome_types.from_tiff(img_path)
    n_channels = len(channel_names)
    n_z = len(ome.images[0].pixels.channels) / n_channels
    assert n_z == int(n_z)
    n_z = int(n_z)

    ome_planes = [
        ome_types.model.Plane(the_c=cc, the_z=zz, the_t=0)
        for cc, zz in itertools.product(range(n_channels), range(n_z))
    ]
    ome_channels = [
        ome_types.model.Channel(id=f"Channel:0:{c}", name=n)
        for (c, n) in zip(range(n_channels), channel_names)
    ]
    out_ome = copy.deepcopy(ome)
    pixel = out_ome.images[0].pixels
    pixel.channels = ome_channels
    pixel.size_c = n_channels
    pixel.size_z = n_z
    # NOTE Z here needs to come before C!
    pixel.dimension_order = "XYZCT"
    pixel.planes = ome_planes
    with open(
        img_path.parent / f"{img_path.name.split('.')[0]}-multi-z.ome.xml", "w"
    ) as f:
        f.write(out_ome.to_xml())
    return out_ome


def process_file(img_path, channel_names):
    import tifffile

    write_ori_omexml(img_path)
    ome = update_xml_with_z(img_path, channel_names)

    tifffile.tiffcomment(img_path, ome.to_xml().encode())


channel_names = "Hoechst1, Goat, Rabbit, Mouse, Hoechst2, CytoC, PDL1, PD1, Hoechst3, E-cad, panCK, PDL1, Hoechst4, LaminB1, CD3d, PD1, Hoechst5, p62, 28-8, CAL10, Hoechst6, PDI, COXIV, HLAA, Hoechst7, EGFR, Nestin, NUP98, Hoechst8, Catenin, Ki67, H2ax, Hoechst9, PCNA, Actin, tubulin, Hoechst10, CD45, CD68, CD163, Hoechst11, PCNA, CD11c, PDL1, Hoechst12, CD4, CD44, CD8a, Hoechst13, LaminAC, Desmin, HER2, Hoechst14, Bax, Vinculin, KAP1, Hoechst15, EpCAM, SOX2, pKAP1, Hoechst16, gTUB, CK7, S100A4, Hoechst17, CK17, CK20, GLUT1, Hoechst18, p53, SLFN11, CK5"
channel_names = channel_names.split(", ")

process_file(
    r"X:\cycif-production\149-B417-PDL1\Highres_InCell\mcmicro\LSP18900\registration\LSP18900.ome.tif",
    channel_names=channel_names,
)

process_file(
    r"X:\cycif-production\149-B417-PDL1\Highres_InCell\mcmicro\LSP18906\registration\LSP18906.ome.tif",
    channel_names=channel_names,
)
