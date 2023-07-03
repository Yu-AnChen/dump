import zarr
import numpy as np
import joblib
from ashlar import reg
import tqdm


def _paste(target, img, pos):
    rs, cs = pos
    h, w = img.shape
    target[rs:rs+h, cs:cs+w] = img


def _pyramid_shapes(img_path):
    metadata = reg.BioformatsMetadata(img_path)
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

    metadata = reg.BioformatsMetadata(img_path)
    num_channels = metadata.num_channels

    cache_path = r"C:\rcpnl\scans\000-dev\ttt.zarr"
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


def paste_tile(img_path, channel, zgroup, verbose=False):
    reader = reg.BioformatsReader(img_path)
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


from threading import Lock
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


def paste_tile_parallel(img_path, channel, zgroup):
    reader = ThreadSafeBioformatsReader(img_path)
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


import platform
import os
import pathlib


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


import time, datetime

# img_path = r"C:\rcpnl\scans\LSP15513@20230610_164824_823377\LSP15513@20230610_164824_823377.rcpnl"
img_path = r"C:\rcpnl\YX_GBM-PDX\LSP17019@20230531_225954_474845\LSP17019@20230531_225954_474845.rcpnl"
start_time = time.perf_counter()

zzz = make_reader_pyramid(img_path, 'channel')

end_time = time.perf_counter()

print()
print('elapsed', datetime.timedelta(seconds=int(end_time)-int(start_time)))
print()



import dask.array as da
import napari
pyramid = [
    da.array([da.from_zarr(aa) for aa in group.values()])
    for _, group in zzz.groups()
]
pyramid[1] = pyramid[1].persist()
pyramid[2] = pyramid[2].persist()

v = napari.Viewer()
v.add_image(pyramid, channel_axis=0, contrast_limits=(0, 65535))






# could use watchdog to monitor directory file changes















# ---------------------------------------------------------------------------- #
#                                   version 1                                  #
# ---------------------------------------------------------------------------- #

c1r = reg.BioformatsReader(r"C:\rcpnl\scans\LSP12317@20230601_202742_555378\LSP12317@20230601_202742_555378.rcpnl")

zzz = make_reader_pyramid(c1r)

import dask.array as da
import napari
pyramid = [
    da.array([da.from_zarr(aa) for aa in group.values()])
    for _, group in zzz.groups()
]
pyramid[1] = pyramid[1].persist()
pyramid[2] = pyramid[2].persist()

v = napari.Viewer()
v.add_image(pyramid, channel_axis=0, contrast_limits=(0, 65535))