import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import palom

# ---------------------------------------------------------------------------- #
#                          matplotlib helper functions                         #
# ---------------------------------------------------------------------------- #
def set_matplotlib_font(font_size=12):
    font_families = matplotlib.rcParams['font.sans-serif']
    if font_families[0] != 'Arial':
        font_families.insert(0, 'Arial')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams.update({'font.size': font_size})


def save_all_figs(dpi=300, format='pdf', out_dir=None):
    figs = [plt.figure(i) for i in plt.get_fignums()]
    names = [f._suptitle.get_text() if f._suptitle else '' for f in figs]
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for f, n, nm in zip(figs, plt.get_fignums(), names):
        f.savefig(out_dir / f"{n}-{nm}.{format}", dpi=dpi, bbox_inches='tight')
        plt.close(f)

# ---------------------------------------------------------------------------- #
#                                  main script                                 #
# ---------------------------------------------------------------------------- #
files = [
    'small-ome-tiff/S3_T1-4_I0.ome.tif',
    'small-ome-tiff/S3_T5-8_I0.ome.tif',
    'small-ome-tiff/S3_T9-12_I0.ome.tif'
]
output_path = pathlib.Path('small-ome-tiff/registered.ome.tif')

out_dir = output_path.parent
out_dir.mkdir(parents=True, exist_ok=True)

readers = [palom.reader.OmePyramidReader(f) for f in files]
ref_reader = readers[0]

# workaround for file-is-closed issue
for reader in readers:
    _ = reader.pyramid[0].blocks.ravel()[0].persist()


THUMBNAIL_LEVEL = ref_reader.get_thumbnail_level_of_size(500)
CHANNEL = 2

aligners = []
for reader in readers[1:]:
    aligner = palom.align.get_aligner(
        ref_reader,
        reader,
        channel1=CHANNEL,
        channel2=CHANNEL,
        thumbnail_level1=THUMBNAIL_LEVEL,
        thumbnail_level2=THUMBNAIL_LEVEL,
        thumbnail_channel1=CHANNEL,
        thumbnail_channel2=CHANNEL
    )
    aligners.append(aligner)

N_KEYPOINTS = 5_000

mosaics = [ref_reader.pyramid[0]]
for aligner, reader in zip(aligners, readers[1:]):
    p1 = ref_reader.path
    p2 = reader.path
    aligner.coarse_register_affine(n_keypoints=N_KEYPOINTS, detect_flip_rotate=True)
    plt.gcf().suptitle(f"{p2.name} (coarse alignment)", fontsize=8)
    plt.gca().set_title(f"{p1.name} - {p2.name}", fontsize=6)
    save_all_figs(out_dir=out_dir / 'qc', format='png')

    aligner.compute_shifts()

    aligner.ref_thumbnail = np.array(aligner.ref_thumbnail)
    fig = aligner.plot_shifts()
    fig.suptitle(f"{p2.name} (block shift distance)", fontsize=8)
    fig.axes[0].set_title(p1.name, fontsize=6)
    save_all_figs(out_dir=out_dir / 'qc', format='png')

    aligner.constrain_shifts()
    
    mosaic = palom.align.block_affine_transformed_moving_img(
        ref_img=aligner.ref_img,
        moving_img=reader.pyramid[0],
        mxs=aligner.block_affine_matrices_da
    )
    mosaics.append(mosaic)

palom.pyramid.write_pyramid(
    mosaics=mosaics,
    output_path=output_path,
    pixel_size=ref_reader.pixel_size,
    compression='zlib',
    downscale_factor=2,
    save_RAM=True,
    tile_size=1024,
)
