import palom
import tifffile
import pathlib


def compress_pyramid(img_path, out_path=None):
    img_path = pathlib.Path(img_path)
    reader = palom.reader.OmePyramidReader(img_path)
    if out_path is None:
        out_path = img_path.parent / img_path.name.replace('.ome', '-zlib.ome')
    palom.pyramid.write_pyramid(
        [reader.pyramid[0]],
        out_path,
        downscale_factor=2,
        compression='zlib',
        tile_size=1024
    )
    ome_xml = tifffile.tiffcomment(img_path)
    tifffile.tiffcomment(out_path, ome_xml.encode())
    return out_path


def fix_stitcher_ome_xml(
    img_path,
    replace_from,
    replace_to
):
    ori = tifffile.tiffcomment(img_path)
    n_to_replace = ori.count(replace_from)
    if n_to_replace == 0:
        print(f"Substring to be replaced not found in the file ({img_path})")
        return 0
    fixed = ori.replace(replace_from, replace_to)
    tifffile.tiffcomment(img_path, fixed.encode())
    print(f"{n_to_replace} instance(s) of {replace_from} replaced with {replace_to}")
    return 0


import csv
import pathlib


curr = pathlib.Path(r'Z:\RareCyte-S3\P37_CRCstudy_Round1')
with open(r'Z:\RareCyte-S3\YC-analysis\P37_CRCstudy_Round1\scripts-processing\file_list.csv') as ff:
    csv_reader = csv.DictReader(ff)
    file_config = [dict(row) for row in csv_reader]


def wrap(in_path):
    out_path = compress_pyramid(in_path)
    fix_stitcher_ome_xml(
        out_path,
        '</Channel><Plane',
        '</Channel><MetadataOnly></MetadataOnly><Plane'
    )
    return

from joblib import Parallel, delayed

Parallel(n_jobs=3, verbose=1)(delayed(wrap)(curr / row['path']) for row in file_config[1:-2])
