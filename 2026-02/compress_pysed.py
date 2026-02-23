import ome_types
import tifffile
import tqdm
import zarr
from xsdata.formats.dataclass.parsers.config import ParserConfig


def compress_pysed(pysed_path, out_path):
    zimg = zarr.open(tifffile.imread(pysed_path, aszarr=True), mode="r")
    # need to supress this message
    # <tifffile.TiffFile 'LSP67203_P215_0…93.pysed.ome.tif'> OME series contains invalid TiffData index, raised ValueError('invalid entry in coordinates array')

    with tifffile.TiffWriter(out_path, bigtiff=True) as tif:
        for ii in tqdm.trange(len(zimg)):
            tif.write(zimg[ii], compression="zlib")

    parser_config = ParserConfig(
        fail_on_unknown_properties=False, fail_on_unknown_attributes=False
    )
    ome = ome_types.from_tiff(pysed_path, parser_kwargs={"config": parser_config})

    tifffile.tiffcomment(out_path, ome.to_xml().encode())
    return out_path
