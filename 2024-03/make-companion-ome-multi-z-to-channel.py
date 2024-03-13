import copy
import pathlib
import uuid
import xml.etree.ElementTree as ET

import ome_types
import tifffile


def make_ome_pixel(img_path, sample_ome, position_x, position_y, pixel_size=1):
    img_path = pathlib.Path(img_path)
    sample_ome = copy.deepcopy(sample_ome)
    pixel = sample_ome.images[0].pixels

    pixel.physical_size_x = pixel_size
    pixel.physical_size_y = pixel_size

    UUID = ome_types.model.tiff_data.UUID(
        file_name=str(img_path.name), value=uuid.uuid4().urn
    )

    tiff_block = pixel.tiff_data_blocks[0]

    num_planes = tiff_block.plane_count
    tiff_block.uuid = UUID

    for i in range(num_planes):
        plane = ome_types.model.Plane(
            the_c=i, the_z=0, the_t=0, position_x=position_x, position_y=position_y
        )
        pixel.planes.append(plane)

    return pixel


# custom func to parse stage position from image file header
def stage_positions(img_path):
    with tifffile.TiffFile(img_path) as tif:
        root = ET.fromstring(tif.pages[0].tags["ImageDescription"].value)

        pos_y = root.findall("log/y_pos")[0].text
        pos_x = root.findall("log/x_pos")[0].text
        return float(pos_x), float(pos_y)


def make_companion(cycle_dir, filename_pattern=None, channels=None):
    from ashlar import reg

    if filename_pattern is None:
        filename_pattern = "A - 1(fld * wv {channel}).tif"
    if channels is None:
        channels = [
            "UV - DAPI z 1",
            "UV - DAPI z 2",
            "UV - DAPI z 3",
            "Blue - FITC z 1",
            "Blue - FITC z 2",
            "Blue - FITC z 3",
            "Green - dsRed z 1",
            "Green - dsRed z 2",
            "Green - dsRed z 3",
            "Red - Cy5 z 1",
            "Red - Cy5 z 2",
            "Red - Cy5 z 3",
        ]

    cycle_dir = pathlib.Path(cycle_dir)
    xdce_filepath = sorted(cycle_dir.glob("*.xdce"))[0]
    pixel_size = reg.BioformatsMetadata(str(xdce_filepath)).pixel_size
    yx_positions = (
        reg.BioformatsMetadata(str(xdce_filepath)).positions * pixel_size * [-1, 1]
    )
    all_pixels = []
    FACTOR = 1
    PIXEL_SIZE = pixel_size
    for channel in channels:
        img_paths = sorted(cycle_dir.glob(filename_pattern.format(channel=channel)))
        print(f"{channel}: {len(img_paths)} files")
        assert len(img_paths) > 0
        if channel == channels[0]:
            tifffile.imwrite("sample.ome.tif", tifffile.imread(img_paths[0]))
            sample_ome = ome_types.from_tiff("sample.ome.tif")

        pixels = [
            make_ome_pixel(path, sample_ome, pos_x * FACTOR, pos_y * FACTOR, PIXEL_SIZE)
            for path, (pos_y, pos_x) in zip(img_paths, yx_positions)
        ]
        all_pixels.append(pixels)

    num_files = [len(pp) for pp in all_pixels]
    assert (
        len(set(num_files)) == 1
    ), f"number of files for each channel does not match: {num_files}"

    for p0, *p_others in zip(*all_pixels):
        p0.size_c = len(all_pixels)
        for idx, pi in enumerate(p_others):
            p0.channels.extend(pi.channels)
            p0.tiff_data_blocks.extend(pi.tiff_data_blocks)
            p0.tiff_data_blocks[-1].first_c = idx + 1
            p0.planes.extend(pi.planes)
            p0.planes[-1].the_c = idx + 1

    omexml = ome_types.model.OME()
    omexml.images = [ome_types.model.Image(pixels=p) for p in all_pixels[0]]

    out_path = cycle_dir / f"{cycle_dir.name}.companion.ome"
    print(f"\nWriting to {out_path}\n")
    with open(out_path, "w") as f:
        f.write(omexml.to_xml())


curr = pathlib.Path(
    r"X:\cycif-production\149-B417-PDL1\Highres_InCell\Jerry_PDL1_LSP18906_20240222"
)
cycles = curr.glob("*")
for pp in cycles:
    make_companion(pp)

"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate companion-ome",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("cycle_dir", type=pathlib.Path, help="cycle directory")
    parser.add_argument(
        "--filename_pattern",
        default="*_P*_Z01_{channel}.tif",
        help="image filename pattern",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["dapi", "fitc", "cy3", "cy5"],
        help="channel identifier(s)",
    )

    args = parser.parse_args()
    make_companion(args.cycle_dir, args.filename_pattern, args.channels)
"""
