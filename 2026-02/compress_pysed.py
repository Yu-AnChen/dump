import logging

import ome_types
import tifffile
import tqdm
import zarr
from xsdata.formats.dataclass.parsers.config import ParserConfig


class MessageFilter(logging.Filter):
    def __init__(self, *substrings):
        self.substrings = substrings

    def filter(self, record):
        return not any(s in record.getMessage() for s in self.substrings)


logging.getLogger("tifffile").addFilter(
    MessageFilter("OME series contains invalid TiffData index")
)


def compress_pysed(pysed_path, out_path):
    zimg = zarr.open(tifffile.imread(pysed_path, aszarr=True), mode="r")
    # WARNING:tifffile:<tifffile.TiffFile 'LSP67131_P215_0…87.pysed.ome.tif'> OME series contains invalid TiffDataindex, raised ValueError('invalid entry in coordinates array')

    with tifffile.TiffWriter(out_path, bigtiff=True) as tif:
        for ii in tqdm.trange(len(zimg)):
            tif.write(zimg[ii], compression="zlib")

    parser_config = ParserConfig(
        fail_on_unknown_properties=False, fail_on_unknown_attributes=False
    )
    ome = ome_types.from_tiff(pysed_path, parser_kwargs={"config": parser_config})

    tifffile.tiffcomment(out_path, ome.to_xml().encode())
    return out_path
