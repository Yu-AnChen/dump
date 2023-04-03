import copy
import pathlib
import uuid
import xml.etree.ElementTree as ET

import ome_types
import tifffile


def make_ome_pixel(
    img_path, sample_ome, position_x, position_y, pixel_size=1
):
    img_path = pathlib.Path(img_path)
    sample_ome = copy.deepcopy(sample_ome)
    pixel = sample_ome.images[0].pixels
    
    pixel.physical_size_x = pixel_size
    pixel.physical_size_y = pixel_size

    UUID = ome_types.model.tiff_data.UUID(
        file_name=str(img_path.name),
        value=uuid.uuid4().urn
    )

    tiff_block = pixel.tiff_data_blocks[0]

    num_planes = tiff_block.plane_count
    tiff_block.uuid = UUID

    for i in range(num_planes):
        plane = ome_types.model.Plane(
            the_c=i, the_z=0, the_t=0,
            position_x=position_x, position_y=position_y
        )
        pixel.planes.append(plane)
    
    return pixel


# custom func to parse stage position from image file header
def stage_positions(img_path):
    with tifffile.TiffFile(img_path) as tif:
        root = ET.fromstring(tif.pages[0].tags['ImageDescription'].value)
        pos_y = root.findall(".//*[@id='Olympus Stage Y']")[0].attrib['value']
        pos_x = root.findall(".//*[@id='Olympus Stage X']")[0].attrib['value']
        return float(pos_x), float(pos_y)


# pixel size is found in image file header
PIXEL_SIZE = 0.325


all_pixels = []
for channel in range(1, 5):
    img_paths = sorted(
        pathlib.Path('.').glob(f'cycle2/PIO51_cycle2_w{channel}_s*_t1.TIF'),
        # key=lambda x: int(re.search(r'_s(?P<t>\d*)_', x.name).group('t'))
    )

    # use positions in the first channel
    if channel == 1:
        # generate an ome-xml template
        tifffile.imwrite('sample.ome.tif', tifffile.imread(img_paths[0]))
        sample_ome = ome_types.from_tiff('sample.ome.tif')
        
        xy_positions = [stage_positions(p) for p in img_paths]

    pixels = [
        make_ome_pixel(path, sample_ome, pos_x, pos_y, PIXEL_SIZE)
        for path, (pos_x, pos_y) in zip(img_paths, xy_positions)
    ]
    all_pixels.append(pixels)

for p0, *p_others in zip(*all_pixels):
    p0.size_c = len(all_pixels)
    for idx, pi in enumerate(p_others):
        p0.channels.extend(pi.channels)
        p0.tiff_data_blocks.extend(pi.tiff_data_blocks)
        p0.tiff_data_blocks[-1].first_c = idx+1
        p0.planes.extend(pi.planes)
        p0.planes[-1].the_c = idx+1


omexml = ome_types.model.OME()
omexml.images = [ome_types.model.Image(pixels=p) for p in all_pixels[0]]

out_path = 'cycle2/PIO51_cycle2.companion.ome'
print(f"Writing to {out_path}\n")
with open(out_path, 'w') as f:
    f.write(omexml.to_xml())