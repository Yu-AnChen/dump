from ashlar import reg, fileseries
import numpy as np
import re
import pathlib


class TileReader(fileseries.FileSeriesReader):
    def __init__(
        self, path, pattern, position_file_dir, position_file_pattern, pixel_size=1.0
    ):
        
        super().__init__(
            path, pattern, overlap=1, width=1, height=1, pixel_size=pixel_size
        )
        self.position_file_dir = position_file_dir
        self.position_file_pattern = position_file_pattern

        self.regex_pattern = 'ImageCoordX=(.*)\n.*ImageCoordY=(.*)'
    
    def parse(self):
        self.validate_position_files()
        self.parse_positions()
    
    def validate_position_files(self):
        dir = pathlib.Path(self.position_file_dir)
        assert dir.exists()
        p_files = sorted(dir.glob(self.position_file_pattern))
        assert len(p_files) == self.metadata.num_images
        self.position_files = p_files
    
    def parse_positions(self):
        parsed_positions = []
        for p in self.position_files:
            with open(p) as f:
                parsed_positions.append(
                    re.findall(self.regex_pattern, f.read())[0]
                )
        parsed_positions = np.array(parsed_positions).astype(float)

        _ = self.metadata.positions
        self.metadata._positions = np.fliplr(parsed_positions)
        self.metadata._positions /= self.metadata.pixel_size



c1r = TileReader(
    # dir contains tif images
    '/Users/yuanchen/Dropbox (HMS)/ASHLAR Shared Folder/tile images',
    pattern='Img-R1-{series:06}.tif',
    # dir contains metadata TXT files
    position_file_dir='/Users/yuanchen/Dropbox (HMS)/ASHLAR Shared Folder/tile images/metadata R1',
    position_file_pattern='Info-*.TXT',
    # µm/pixel, required for microscope stage positions
    pixel_size=0.172
)
c1r.parse()

c1e = reg.EdgeAligner(c1r, verbose=True, filter_sigma=1)
c1e.run()

c2r = TileReader(
    '/Users/yuanchen/Dropbox (HMS)/ASHLAR Shared Folder/tile images',
    pattern='Img-R2-{series:06}.tif',
    position_file_dir='/Users/yuanchen/Dropbox (HMS)/ASHLAR Shared Folder/tile images/metadata R2',
    position_file_pattern='Info-*.TXT',
    pixel_size=0.172
)
c2r.parse()

c21l = reg.LayerAligner(c2r, c1e, verbose=True, filter_sigma=1)
c21l.run()

mosaic_shape = c1e.mosaic_shape

mosaics = [
    reg.Mosaic(aligner, c1e.mosaic_shape, verbose=True)
    for aligner in (c1e, c21l)
]

writer = reg.PyramidWriter(
    mosaics, 'out.ome.tif', verbose=True
)
writer.run()

# Reference for processing many cycles
# https://github.com/labsyspharm/ashlar/blob/bdc277ff4de4502d002777a48bc9405c38c7e86d/ashlar/scripts/ashlar.py#L222-L279