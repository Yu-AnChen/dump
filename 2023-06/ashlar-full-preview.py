import joblib
import numpy as np
import tqdm
import zarr
from ashlar import reg


def _paste(target, img, pos):
    rs, cs = pos
    h, w = img.shape
    target[rs:rs+h, cs:cs+w] = img


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


def make_reader_pyramid(img_path, parallelize):
    assert parallelize in ['channel', 'tile']
    shapes = _pyramid_shapes(img_path)

    metadata = reg.BioformatsMetadata(str(img_path))
    num_channels = metadata.num_channels

    cache_dir = pathlib.Path(CACHE_DIR)
    cache_path = cache_dir / f"{pathlib.Path(img_path).stem}-ashlar-lt.zarr"
    if cache_path.exists():
        print(f"{cache_path} already exists.")
        return None
    root = zarr.open(cache_path, mode='w')

    root.create_groups(*range(3))
    for channel in range(num_channels):
        for idx, (_, group) in enumerate(root.groups()):
            group.zeros(
                channel,
                shape=shapes[idx],
                dtype=metadata.pixel_dtype,
                chunks=np.ceil(np.divide(metadata.size, 8**idx)).astype(int)
            )
        if parallelize == 'tile':
            paste_tile_parallel(img_path, channel, root)
    
    if parallelize == 'channel':
        n_jobs = min(num_channels, joblib.cpu_count())
        _ = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
            joblib.delayed(paste_tile)(img_path, channel, root, verbose)
            for channel, verbose in zip(range(num_channels), [True]+(num_channels-1)*[False]) 
        )
    return root


# ---------------------------------------------------------------------------- #
#                 Parallel channel mosaic assembly and writing                 #
# ---------------------------------------------------------------------------- #
def paste_tile(img_path, channel, zgroup, verbose=False):
    reader = reg.BioformatsReader(str(img_path))
    positions = reader.metadata.positions
    positions -= reader.metadata.origin
    positions = np.ceil(positions).astype(int)
    
    enum = enumerate(positions)
    if verbose:
        enum = enumerate(tqdm.tqdm(positions))
    for idx, pp in enum:
        img = reader.read(idx, channel)
        for i, (_, group) in enumerate(zgroup.groups()):
            _paste(
                group[channel],
                img[::8**i, ::8**i],
                np.floor(pp / 8**i).astype(int)
            )


# ---------------------------------------------------------------------------- #
#                           Parallelize tile reading                           #
# ---------------------------------------------------------------------------- #
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Callable


def paste_tile_parallel(img_path, channel, zgroup):
    reader = ThreadSafeBioformatsReader(str(img_path))
    positions = reader.metadata.positions
    positions -= reader.metadata.origin
    positions = np.ceil(positions).astype(int)
    
    def _paste_tile(tile, pos):
        img = reader.read(tile, channel)
        for i, (_, group) in enumerate(zgroup.groups()):
            _paste(
                group[channel],
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

TARGET_DIR = r'D:\20230915-ashlar-lt-preview'
CACHE_DIR = r'D:\000-ASHLAR-LT'


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
    cache_dir = pathlib.Path(CACHE_DIR)
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
            store = make_reader_pyramid(str(img_path), 'channel')
        if store is None:
            continue
        end_time = time.perf_counter()
        img_shape = (len(store[0]), *store['/0/0'].shape)
        time_diff = int(end_time - start_time)
        _to_log(cache_dir / '000-process.log', rcpnl, img_shape, time_diff)

        print()
        print('elapsed', datetime.timedelta(seconds=time_diff))

