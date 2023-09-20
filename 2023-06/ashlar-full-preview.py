# ---------------------------------------------------------------------------- #
#        Hide warnings: Stage coordinates' measurement unit is undefined       #
# ---------------------------------------------------------------------------- #
import warnings
import functools
from ashlar import reg


def ignore_warnings_stage_position_unit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=reg.DataWarning,
                message="Stage coordinates' measurement unit is undefined"
            )
            return func(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------- #
#                                  Main block                                  #
# ---------------------------------------------------------------------------- #
import joblib
import numpy as np
import tqdm
import zarr
from ashlar import reg


def _paste(target, img, pos):
    rs, cs = pos
    h, w = img.shape
    target[rs:rs+h, cs:cs+w] = img


@ignore_warnings_stage_position_unit
def _pyramid_shapes(img_path):
    metadata = reg.BioformatsMetadata(str(img_path))
    base_shape = np.ceil(
        metadata.positions.max(axis=0) - metadata.origin + metadata.size
    ).astype(int)
    shapes = [
        np.ceil(base_shape / 8**i).astype(int)
        for i in range(3)
    ]
    return shapes


def _make_ngff(path, shape, tile_shape=None, dtype='uint16', pixel_size=1):
    import ome_zarr.scale
    import ome_zarr.writer
    import ome_zarr.io
    import dask.array as da

    store = ome_zarr.io.parse_url(path, mode="w").store
    root = zarr.group(store=store, overwrite=True)
    # Total 3 levels, 8x downsizing each level
    n_levels = 3
    scaler = ome_zarr.scale.Scaler(
        downscale=8,
        max_layer=n_levels-1,
        method='nearest',
        copy_metadata=False,
        in_place=False,
        labeled=False,
    )
    # Is this a point of optimization?
    # FIXME add 1 px to highest level pyramid
    data = da.zeros([shape[0], shape[1]+8**2, shape[2]+8**2], dtype=dtype)

    if tile_shape is None: tile_shape = (1024, 1024)
    chunks = [
        (1, *np.ceil(np.divide(tile_shape, 8**i)).astype(int))
        for i in range(n_levels)
    ]

    ome_zarr.writer.write_image(
        image=data,
        group=root,
        scaler=scaler,
        axes='cyx',
        # Is using FOV-sized chunk better? Yes
        storage_options=[dict(chunks=cc) for cc in chunks],
        compute=False
    )
    root.attrs['multiscales'] = _update_pixel_size(
        root.attrs['multiscales'], pixel_size
    )
    return root


def _update_pixel_size(multiscale_metadata, pixel_size, downscale_factor=8):
    ori = {**multiscale_metadata[0]}
    axes = [
        ax if ax['type'] != 'space'
        else {**ax, **{'unit': 'micrometer'}}
        for ax in ori['axes']
    ]
    datasets = []
    for idx, dd in enumerate(ori['datasets']):
        factor = downscale_factor**idx
        micron_scale = [
            ss*ff
            for ss, ff in zip(
                dd['coordinateTransformations'][0]['scale'],
                [1, pixel_size*factor, pixel_size*factor]
            )
        ]
        datasets.append({**dd, **{'coordinateTransformations': [{'scale': micron_scale}]}})
    updated = {
        **ori,
        **{'axes': axes, 'datasets': datasets}
    }
    return [updated]

    ...


def _rcjob_channel_names(rcjob_path):
    import json
    with open(rcjob_path) as f:
        markers = json.load(f)['scanner']['assay']['biomarkers']
    return [
        '-'.join(mm.split('-')[:-1]) for mm in markers
    ]


def add_channel_metadata(
    path,
    channel_names,
    channel_colors=None,
    channel_contrasts=None,
):
    import ome_zarr.io
    store = ome_zarr.io.parse_url(path, mode="a").store
    root = zarr.group(store=store)
    n_channels, _, _ = root[0].shape

    # channel names
    # list[str]
    if len(channel_names) != n_channels:
        print(
            f"Adding channel names aborted. {n_channels} channels in {path}"
            f" but {len(channel_names)} channel names were provided."
        )
    channel_names = [
        {'label': n} for n in channel_names
    ]

    # channel colors
    # list[str]
    default_colors = [
        '0000ff',
        '00ff00',
        'ffffff',
        'ff0000',
        'ff9900',
    ]
    if channel_colors is None:
        channel_colors = [
            default_colors[i%len(default_colors)] 
            for i in range(n_channels)
        ]
    assert len(channel_colors) == n_channels
    channel_colors = [
        {'color': c} for c in channel_colors
    ]

    # channel contrast
    # list[tuple(float, float)]
    dtype = root[0].dtype
    if channel_contrasts is None:
        if np.issubdtype(dtype, np.integer):
            dmin, dmax = np.iinfo(dtype).min, np.iinfo(dtype).max
            channel_contrasts = [{'window': {'start': dmin, 'end': dmax}}] * n_channels
        else:
            channel_contrasts = [None] * n_channels
    else:
        channel_contrasts = [
            {'window': {'start': dmin, 'end': dmax}}
            for dmin, dmax in channel_contrasts
        ]
    assert len(channel_contrasts) == n_channels

    channels = []
    for name, color, contrast in zip(
        channel_names, channel_colors, channel_contrasts
    ):
        if contrast is None:
            contrast = {}
        channels.append(
            {**name, **color, **contrast, **{'active': False}}
        )
    root.attrs["omero"] = {"channels": channels}
    return channels


@ignore_warnings_stage_position_unit
def make_reader_pyramid(img_path, parallelize):
    assert parallelize in ['channel', 'tile']
    shapes = _pyramid_shapes(img_path)

    metadata = reg.BioformatsMetadata(str(img_path))
    num_channels = metadata.num_channels
    tile_shape = metadata.size
    pixel_size = metadata.pixel_size

    cache_dir = get_path(CACHE_DIR)
    cache_path = cache_dir / f"{pathlib.Path(img_path).stem}.ome.zarr"
    if cache_path.exists():
        print(f"{cache_path} already exists.")
        return None

    root = _make_ngff(
        cache_path,
        (num_channels, *shapes[0]),
        tile_shape=tile_shape,
        dtype=metadata.pixel_dtype,
        pixel_size=pixel_size
    )

    for channel in range(num_channels):
        if parallelize == 'tile':
            paste_tile_parallel(img_path, channel, root)
    
    if parallelize == 'channel':
        n_jobs = min(num_channels, joblib.cpu_count())
        _ = joblib.Parallel(n_jobs=4, verbose=0)(
            joblib.delayed(paste_tile)(img_path, channel, root, verbose)
            for channel, verbose in zip(range(num_channels), [True]+(num_channels-1)*[False]) 
        )
    return root


# ---------------------------------------------------------------------------- #
#                 Parallel channel mosaic assembly and writing                 #
# ---------------------------------------------------------------------------- #
@ignore_warnings_stage_position_unit
def paste_tile(img_path, channel, zgroup, verbose=False):
    reader = reg.BioformatsReader(str(img_path))
    positions = reader.metadata.positions
    positions -= reader.metadata.origin
    positions = np.floor(positions).astype(int)
    
    enum = enumerate(positions)
    if verbose:
        enum = enumerate(tqdm.tqdm(positions))
    for idx, pp in enum:
        img = reader.read(idx, channel)
        for i, (_, aa) in enumerate(zgroup.arrays()):
            rs, cs = np.floor(pp / 8**i).astype(int)
            _img = img[::8**i, ::8**i]
            h, w = _img.shape
            aa[channel, rs:rs+h, cs:cs+w] = _img


# ---------------------------------------------------------------------------- #
#                           Parallelize tile reading                           #
# ---------------------------------------------------------------------------- #
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Callable


@ignore_warnings_stage_position_unit
def paste_tile_parallel(img_path, channel, zgroup):
    reader = ThreadSafeBioformatsReader(str(img_path))
    positions = reader.metadata.positions
    positions -= reader.metadata.origin
    positions = np.floor(positions).astype(int)
    
    def _paste_tile(tile, pos):
        img = reader.read(tile, channel)
        for i, (_, aa) in enumerate(zgroup.arrays()):
            _paste(
                aa[channel],
                img[::8**i, ::8**i],
                np.floor(pos / 8**i).astype(int)
            )

    execute_parallel(
        _paste_tile,
        args=[(idx, pp) for idx, pp in enumerate(positions)]
    )


def execute_parallel(func: Callable, *, args: list, tqdm_args: dict = None):
    if tqdm_args is None:
        tqdm_args = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        with tqdm.tqdm(total=len(args), **tqdm_args) as pbar:
            futures = {executor.submit(func, *arg): i for i, arg in enumerate(args)}
            for _ in as_completed(futures):
                pbar.update(1)


class ThreadSafeBioformatsReader(reg.BioformatsReader):
    lock = Lock()

    def read(self, *args, **kwargs):
        with self.lock:
            return super().read(*args, **kwargs)


# ---------------------------------------------------------------------------- #
#                       Helper to handle Windows shortcut                      #
# ---------------------------------------------------------------------------- #
import os
import pathlib
import platform


# handle processing on window shortcuts 
def get_path(path):
    path = str(path)
    system = platform.system()
    if system in ["Linux", "Darwin"]:
        path = os.path.realpath(path)
    elif system == "Windows":
        import win32com.client
        shell = win32com.client.Dispatch("WScript.Shell")
        if path.endswith('lnk'):
            path = shell.CreateShortCut(path).Targetpath
        else: path = ''
    else:
        print('Unknown os.')
        path = ''
    return pathlib.Path(path)


# ---------------------------------------------------------------------------- #
#                         Process folder with shortcuts                        #
# ---------------------------------------------------------------------------- #
import datetime
import time
import warnings


TARGET_DIR = r"/Users/yuanchen/Dropbox (HMS)/ashlar-dev-data/ashlar-rotation-data/2"
CACHE_DIR = r"/Users/yuanchen/projects/napari-wsi-reader/src/.dev"


def _process_path(filepath):
    filepath = get_path(filepath)
    rcpnl, rcjob = None, None
    if not filepath.exists(): return
    if filepath.is_dir():
        rcpnl = next(filepath.glob('*.rcpnl'), None)
        rcjob = next(filepath.glob('*.rcjob'), None)
    if filepath.suffix == '.rcpnl':
        rcpnl = filepath
    return rcpnl, rcjob


def _to_log(log_path, img_path, img_shape, time_diff):
    pathlib.Path(log_path).parent.mkdir(exist_ok=True, parents=True)
    with open(log_path, 'a') as log_file:
        log_file.write(
            f"{datetime.timedelta(seconds=time_diff)} | {img_path.name} | {img_shape} \n"
        )


def process_dir(target_dir):
    cache_dir = get_path(CACHE_DIR)
    cache_dir.mkdir(exist_ok=True, parents=True)
    target_dir = pathlib.Path(target_dir)
    slides = []
    for filepath in target_dir.iterdir():
        slides.append(_process_path(filepath))
    slides = filter(lambda x: x[0], slides)

    for rcpnl, rcjob in slides:
        img_path = rcpnl
        print('Processing', rcpnl)
        start_time = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter(
                action='ignore',
                category=reg.DataWarning,
            )
        root = make_reader_pyramid(str(img_path), 'channel')
        if root is None:
            continue
        if rcjob is not None:
            print('Adding channel names from rcjob')
            channel_names = _rcjob_channel_names(rcjob)
            channel_names = [
                f"{nn} ({rcpnl.name.split('@')[0]})"
                for nn in channel_names
            ]
            add_channel_metadata(root.store.dir_path(), channel_names=channel_names)

        end_time = time.perf_counter()
        img_shape = root['0'].shape
        time_diff = int(end_time - start_time)
        _to_log(cache_dir / '000-process.log', rcpnl, img_shape, time_diff)

        print('elapsed', datetime.timedelta(seconds=time_diff))
        print()


if __name__ == '__main__':
    process_dir(TARGET_DIR)

# ---------------------------------------------------------------------------- #
#                                  Next steps                                  #
# ---------------------------------------------------------------------------- #
'''
[ ] Option to write out half resolution for 0.325 µm/px data
[ ] Performance improvement by optimize stage positions and chunking
[ ] Import into omero or upload to AWS S3 bucket
'''
