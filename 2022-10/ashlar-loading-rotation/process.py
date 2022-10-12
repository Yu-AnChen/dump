import pathlib
import numpy as np
from ashlar import viewer
from ashlar import reg
from ashlar import utils
import find_rotation
import tqdm
import preproc_reader

slide_id = 'LSP12961'
mcmicro_dir = pathlib.Path(r'X:\cycif-production\45-OMS-Atlas\JRL-216-OMS_2022MAR-2022MAR\mcmicro')
target_file = mcmicro_dir / slide_id / 'registration' / f"{slide_id}.ome.tif"
offset = 16


rcpnls = sorted(pathlib.Path(r'X:\cycif-production\45-OMS-Atlas\JRL-216-OMS_2022MAR-2022MAR\00_RCPNL').rglob(f'{slide_id}@*.rcpnl'))[1:]


c1r = reg.BioformatsReader(str(rcpnls[0]))
c2r = reg.BioformatsReader(str(rcpnls[4]))
ffp_path = (mcmicro_dir / slide_id / 'illumination' / f"{rcpnls[4].stem}-ffp.tif")
dfp_path = (mcmicro_dir / slide_id / 'illumination' / f"{rcpnls[4].stem}-dfp.tif")
assert ffp_path.exists() & dfp_path.exists()


c1e = reg.EdgeAligner(c1r, verbose=True, max_shift=30, filter_sigma=1)
c1e.run()

c21l = reg.LayerAligner(c2r, c1e, max_shift=30)
c21l.make_thumbnail()
c21l.coarse_align()

edgy_scores = [
    utils.whiten(c21l.reader.read(i, 0), sigma=1).var()
    for i in tqdm.tqdm(range(c21l.reader.metadata.num_images))
]
edgy_tiles = np.argsort(edgy_scores)[::-1][:20]

found_rotations = [
    find_rotation.register(c21l, i)
    for i in tqdm.tqdm(edgy_tiles)
]

rotation_angle = np.median(found_rotations)
print('Detected rotation angle:', rotation_angle)

c2rr = preproc_reader.HeBioformatsReader(c2r.path, angle=rotation_angle)
c2rr.metadata._positions *= [-1, 1]
c21lr = reg.LayerAligner(c2rr, c1e, max_shift=50, verbose=True, filter_sigma=1)
c21lr.run()


c2m = reg.Mosaic(
    c21lr, c1e.mosaic_shape, verbose=True,
    ffp_path=str(ffp_path), dfp_path=str(dfp_path),
)


import tifffile, zarr
import skimage.transform

store = tifffile.imread(target_file, aszarr=True, mode='r+')
z = zarr.open(store, mode='r+')

assert c1e.mosaic_shape == z[0].shape[1:]

for channel in c2m.channels:
    img = c2m.assemble_channel(channel)
    for i, gp in enumerate(z):
        if i > 0:
            img = skimage.transform.downscale_local_mean(
                img, (2, 2)
            )
            img = np.around(img).astype(np.uint16)
        z[gp][channel+offset] = img

store.close()
