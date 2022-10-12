import pathlib
import re
import uuid

import zarr
import ome_types
import tifffile
import numpy as np
import fire


def parse_info_file(info_file, regex_pattern):
    with open(info_file) as f:
        return np.array(
            re.findall(regex_pattern, f.read())[0]
        ).astype(float)


def construct_ome_pixel(
    img, info_file, pixel_size, regex_pattern
):
    zimg = zarr.open(tifffile.imread(img, aszarr=True))
    size_c, size_y, size_x = zimg.shape
    dtype = np.dtype(zimg.dtype).name
    pixel = ome_types.model.Pixels(
        dimension_order='XYCZT',
        size_x=size_x,
        size_y=size_y,
        size_c=size_c,
        size_z=1,
        size_t=1,
        type=dtype,
        physical_size_x=pixel_size,
        physical_size_y=pixel_size
    )
    position_x, position_y = parse_info_file(info_file, regex_pattern)

    UUID = ome_types.model.tiff_data.UUID(
        file_name=str(img.name),
        value=uuid.uuid4().urn
    )

    for c in range(size_c):
        channel = ome_types.model.Channel(color=-1, samples_per_pixel=1)

        tiff_data_block = ome_types.model.TiffData(
            first_c=c,
            ifd=c,
            plane_count=1,
            uuid=UUID
        )

        plane = ome_types.model.Plane(
            the_c=c, the_z=0, the_t=0,
            position_x=position_x, position_y=position_y
        )

        pixel.channels.append(channel)
        pixel.planes.append(plane)
        pixel.tiff_data_blocks.append(tiff_data_block)

    return pixel


def main(
    img_dir,
    img_pattern='Img-*.tif',
    info_dir=None,
    info_pattern='Info-*.TXT',
    position_regex_pattern='ImageCoordX=(.*)\n.*ImageCoordY=(.*)',
    pixel_size=0.172,
    verbose=True
):

    data_dir = pathlib.Path(img_dir)
    if info_dir is None:
        info_dir = img_dir

    imgs = sorted(pathlib.Path(img_dir).glob(img_pattern))
    info_files = sorted(pathlib.Path(info_dir).glob(info_pattern))

    assert len(imgs) == len(info_files), (
        f"Num images ({len(imgs)}) must equal to num info files ({len(info_files)})"
    )
    assert len(imgs) > 0, (
        f'No image file found at {img_dir} with specified pattern ({img_pattern})'
    )

    print('\nConstructing ome-xml')

    if verbose:
        print('FOVs')
        for idx, (img, info_file) in enumerate(zip(imgs, info_files)):
            print(f"  {idx:04}: {img.name} - {info_file.name}")

    pixels = [
        construct_ome_pixel(img, info_file, pixel_size, regex_pattern=position_regex_pattern)
        for img, info_file in zip(imgs, info_files)
    ]

    omexml = ome_types.model.OME()
    omexml.images = [ome_types.model.Image(pixels=p) for p in pixels]

    out_path = data_dir / f'{data_dir.name}.companion.ome'
    print(f"Writing to {out_path}\n")
    with open(out_path, 'w') as f:
        f.write(omexml.to_xml())

    return 0


if __name__ == '__main__':
  fire.Fire(main)