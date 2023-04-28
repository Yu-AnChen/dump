from ashlar import reg
import networkx as nx
import pathlib


class SubsetBioformatsMetadata(reg.BioformatsMetadata):
    
    def __init__(self, path, series):
        super().__init__(path)
        self.series = series
    
    @property
    def active_series(self):
        return self.series


def detect_sub_series(file_path):
    c1r = reg.BioformatsReader(file_path)
    c1e = reg.EdgeAligner(c1r)
    sub_graphs = nx.connected_components(c1e.neighbors_graph)
    sub_series = [
        sorted(map(int, g))
        for g in sub_graphs
    ]
    return sub_series


def make_sub_series_reader(file_path, series):
    reader = reg.BioformatsReader(file_path)
    reader.metadata = SubsetBioformatsMetadata(file_path, series)
    return reader


def split_slide(file_path):
    sub_series = detect_sub_series(file_path)
    num_regions = len(sub_series)
    plural = 's' if num_regions > 1 else ''
    print(f"{num_regions} region{plural} detected in {file_path}")
    if num_regions == 1:
        return reg.BioformatsReader(file_path)
    return [make_sub_series_reader(file_path, ss) for ss in sub_series]


def stitch_orion(file_path):
    from ashlar.scripts.ashlar import process_axis_flip
    readers = split_slide(file_path)
    file_path = pathlib.Path(file_path)
    output_base = (
        f"{file_path.stem.split('.')[0]}-{{}}ashlar.ome.tif"
    )

    def run_region(reader, output_path):
        process_axis_flip(reader, False, True)
        ea = reg.EdgeAligner(
            reader, filter_sigma=1, do_make_thumbnail=True,
            verbose=True
        )
        ea.run()
        reg.plot_edge_quality(ea, img=ea.reader.thumbnail)
        mshape = ea.mosaic_shape
        writer = reg.PyramidWriter(
            [reg.Mosaic(ea, mshape, verbose=True)],
            output_path,
            verbose=True
        )
        writer.run()
        return ea
    
    from joblib import Parallel, delayed
    return Parallel(n_jobs=len(readers))(
        delayed(run_region)(
            r,
            file_path.parent / output_base.format(f"ROI-{i+1:02}-")
        )
        for i, r in enumerate(readers)
    )

file_path = 'LSP16791.pysed.ome.tif'
eas = stitch_orion(file_path)
