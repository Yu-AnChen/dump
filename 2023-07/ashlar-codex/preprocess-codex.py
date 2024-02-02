import cv2
import numpy as np
from joblib import Parallel, delayed


def whiten(img, sigma=1):
    border_mode = cv2.BORDER_REFLECT
    g_img = img if sigma == 0 else cv2.GaussianBlur(
        img, (0, 0), sigma,
        borderType=border_mode
    ).astype(np.float32)
    log_img = cv2.Laplacian(
        g_img, cv2.CV_32F, ksize=1,
        borderType=border_mode
    )
    return log_img


def compute_best_z(
    reader,
    n_threads=4
):
    def wrap(i):
        return [
            np.linalg.norm(whiten(reader.read(i, 0, z=z), 1))
            for z in reader.metadata.z_map.keys()
        ]
    edgy_scores = np.array(
        Parallel(verbose=1, n_jobs=n_threads, backend='threading')(
            delayed(wrap)(i) 
            for i in range(reader.metadata.num_images)
        )
    )
    
    tile_best_z = np.argmax(edgy_scores, axis=1)
    print(tile_best_z)
    return tile_best_z


import json
import pathlib

import fileseries_reader as fileseries
import numpy as np
import tifffile


def read_experiment_json(json_path):
    with open(json_path) as f:
        return json.load(f)


def make_cycle_reader(cycle_path, imaging_cfg):
    series_pattern = '1_{series}_Z{z}_CH{channel}.tif'
    
    cycle_path = pathlib.Path(cycle_path)
    img = tifffile.imread(
        next(cycle_path.glob(series_pattern.format(series='*', z='*', channel='*')))
    )
    assert img.ndim == 2

    overlap_y = imaging_cfg['tile_overlap_Y'] / img.shape[0]
    overlap_x = imaging_cfg['tile_overlap_X'] / img.shape[1]
    overlap = np.array([overlap_y, overlap_x])

    width = imaging_cfg['region_width']
    height = imaging_cfg['region_height']
    layout = imaging_cfg['tiling_mode']
    pixel_size = imaging_cfg['per_pixel_XY_resolution'] / 1000
    
    reader = fileseries.FileSeriesReader(
        path=str(cycle_path),
        pattern=series_pattern,
        overlap=overlap,
        width=width,
        height=height,
        layout=layout,
        direction="horizontal",
        pixel_size=pixel_size,
    )
    return reader


import copy
import pathlib
import uuid

import joblib
import ome_types


def make_companion_ome(reader, best_zs):
    metadata = reader.metadata
    pixel_base = ome_types.model.Pixels(
        dimension_order='XYCZT',
        size_c=metadata.num_channels,
        size_x=metadata.size[1],
        size_y=metadata.size[0],
        size_t=1,
        size_z=1,
        type=metadata.pixel_dtype.name,
        physical_size_x=metadata.pixel_size,
        physical_size_y=metadata.pixel_size
    )

    ome_images = []
    for ss in range(metadata.num_images):
        pixel = copy.deepcopy(pixel_base)
        tiff_data_blocks = []
        planes = []
        for cc in range(metadata.num_channels):
            UUID = ome_types.model.TiffData.UUID(
                file_name=str(metadata.filename(ss, cc, best_zs[ss])),
                value=uuid.uuid4().urn
            )
            tiffdata = ome_types.model.TiffData(
                first_c=cc,
                uuid=UUID
            )
            plane = ome_types.model.Plane(
                the_c=cc,
                the_t=1,
                the_z=1,
                position_x=metadata.positions[ss][1]*metadata.pixel_size,
                # Flip y position to match ashlar's default configuration
                position_y=-metadata.positions[ss][0]*metadata.pixel_size
            )
            tiff_data_blocks.append(tiffdata)
            planes.append(plane)
        pixel.tiff_data_blocks = tiff_data_blocks
        pixel.planes = planes
        ome_images.append(ome_types.model.Image(pixels=pixel))

    companion_ome = ome_types.model.OME(images=ome_images)

    path = pathlib.Path(metadata.path)
    out_path = path / f"{path.name}.companion.ome"
    print(f"    Writing to {out_path}\n")
    with open(out_path, 'w') as f:
        f.write(companion_ome.to_xml())
    return


def process_cycle(
        cycle_path: str | pathlib.Path,
        experiment_cfg_path: str | pathlib.Path,
        n_threads:int = 4
    ):
    print('Processing', cycle_path)
    cfg = read_experiment_json(experiment_cfg_path)
    reader = make_cycle_reader(cycle_path, cfg)
    best_zs = compute_best_z(reader, n_threads=n_threads)
    make_companion_ome(reader, best_zs)
    print()


def process_slide(
        slide_path: str | pathlib.Path,
        experiment_cfg_path: str | pathlib.Path = None,
        n_threads:int = 4
    ):
    slide_path = pathlib.Path(slide_path)
    cycle_paths = sorted(filter(
        lambda x: x.is_dir(), 
        pathlib.Path(slide_path).glob('*')
    ))
    if experiment_cfg_path is None:
        experiment_cfg_path = slide_path / 'Experiment.json'
    
    n_processes = min(joblib.cpu_count(), len(cycle_paths))
    _ = joblib.Parallel(verbose=1, n_jobs=n_processes)(
        joblib.delayed(process_cycle)(cycle_path, experiment_cfg_path, n_threads=n_threads) 
        for cycle_path in cycle_paths
    )


if __name__ == '__main__':
    import fire
    fire.Fire({
        'cycle': process_cycle,
        'slide': process_slide
    })
    import threadpoolctl
    threadpoolctl.threadpool_limits(1)