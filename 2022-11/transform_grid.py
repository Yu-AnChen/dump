# 
# quantification functions
# 
import skimage.measure
import pandas as pd

def quantify_channel(mask, img, channel_name=None, extra_properties=None):
    channel_name = '' if channel_name is None else f"{channel_name}_"
    results = skimage.measure.regionprops_table(
        label_image=np.asarray(mask),
        intensity_image=np.asarray(img),
        properties=('label',),
        extra_properties=extra_properties
    )
    results = pd.DataFrame(results)
    results.set_index('label', inplace=True)
    results.rename(columns=lambda x: f"{channel_name}{x}", inplace=True)
    return results


def quantify_mask(mask):
    results = skimage.measure.regionprops_table(
        label_image=np.asarray(mask),
        intensity_image=None,
        properties=('label', 'centroid', 'area'),
    )
    results = pd.DataFrame(results)
    results.set_index('label', inplace=True)
    results.rename(columns={
        'centroid-0': 'Y_centroid',
        'centroid-1': 'X_centroid'
    }, inplace=True)
    return results


def quantify(mask, img, extra_properties=None):
    r1 = quantify_mask(mask)
    r2 = quantify_channel(mask, img, extra_properties=extra_properties)
    return r1.join(r2)


import tqdm.contrib
def quantify_all(mask, imgs, channel_names, extra_properties):
    assert len(imgs) == len(channel_names)
    assert len(set(channel_names)) == len(channel_names)
    print('Quantify mask')
    r1 = quantify_mask(mask)
    print('Quantify channels')
    for img, name in tqdm.contrib.tzip(imgs, channel_names):
        r2 = quantify_channel(mask, img, name, extra_properties)
        r1 = r1.join(r2)
    return r1


# 
# custom properties to quantify
# 
import functools
def mms(regionmask, intensity, cutoff=None):
    intensity = intensity[regionmask]
    if cutoff is None:
        return (
            intensity.mean(),
            np.median(intensity),
            intensity.std()
        )
    cintensity = intensity[intensity > cutoff]
    run = cintensity.size > 0
    return (
            intensity.mean(),
            np.median(intensity),
            intensity.std(),
            cintensity.size,
            cintensity.mean() if run else np.nan,
            np.median(cintensity) if run else np.nan,
            cintensity.std() if run else np.nan,
        )

CUTOFF = 1000
mmsc = functools.partial(mms, cutoff=CUTOFF)
# skimage.measure.regionprops_table uses __name__ in func of extra_properties
mmsc.__name__ = 'mmsc'

def rename_mmsc(name):
    mapping = {
        'mmsc-0': 'mean',
        'mmsc-1': 'median',
        'mmsc-2': 'std',
        'mmsc-3': 'ccount',
        'mmsc-4': 'cmean',
        'mmsc-5': 'cmedian',
        'mmsc-6': 'cstd',
    }
    for kk, vv in mapping.items():
        if kk in name:
            return name.replace(kk, vv)
    return name


# 
# process files
# 

# 
# glob files
# 
import pathlib

_files = r'''
\\rc-lab-store-2\orion\rcpnl\tissue\P54_S14_Full_Or6_A31_C90b_HMS@20221005_042252_499647
\\rc-lab-store-2\orion\rcpnl\tissue\P54_S15_Full_Or6_A31_C90b_HMS@20221005_042408_601175
\\rc-lab-store-2\orion\rcpnl\tissue\P54_S16_Full_Or6_A31_C90b_HMS@20221005_042457_495214
\\rc-lab-store-2\orion\rcpnl\tissue\P54_S17_Full_Or6_A31_C90b_HMS@20221005_042540_626251
\\rc-lab-store-2\orion\rcpnl\tissue\P54_S18_Full_Or6_A31_C90b_HMS@20221005_042625_736292
\\rc-lab-store-2\orion\rcpnl\tissue\P54_S19_Full_Or6_A31_C90b_HMS@20221005_042705_464252
\\rc-lab-store-2\orion\rcpnl\tissue\P54_S20_Full_Or6_A31_C90b_HMS@20221010_234316_637182
\\rc-lab-store-2\orion\rcpnl\tissue\P54_S21_Full_Or6_A31_C90b_HMS@20221010_234222_955778
'''
files = _files.replace('\n', '').split(r'\\rc-lab-store-2\orion\rcpnl\tissue''\\')[1:]

files = [
    pathlib.Path(r'\\rc-lab-store-2\orion\rcpnl\tissue') / ff
    for ff in files
]

files = [
    next(ff.glob('*.ome.tiff'))
    for ff in files
]


# 
# align images
# 
import palom
readers = [
    palom.reader.OmePyramidReader(pp)
    for pp in files
]


def get_aligner(c1r, c2r, thumbnail_size=1000):
    LEVEL = 0
    THUMBNAIL_LEVEL = c1r.get_thumbnail_level_of_size(thumbnail_size)

    return palom.align.Aligner(
        ref_img=c1r.read_level_channels(LEVEL, 0),
        moving_img=c2r.read_level_channels(LEVEL, 0),
        ref_thumbnail=c1r.read_level_channels(THUMBNAIL_LEVEL, 0),
        moving_thumbnail=c2r.read_level_channels(THUMBNAIL_LEVEL, 0),
        ref_thumbnail_down_factor=c1r.level_downsamples[THUMBNAIL_LEVEL] / c1r.level_downsamples[LEVEL],
        moving_thumbnail_down_factor=c2r.level_downsamples[THUMBNAIL_LEVEL] / c2r.level_downsamples[LEVEL]
    )


# align to reference section
# tried to align the the middle section 
# ref_id = 4
# but align the first seems sufficient
ref_id = 0
aligners = [
    get_aligner(readers[ref_id], r, 500)
    for r in readers
]

for aa in aligners:
    aa.coarse_register_affine(n_keypoints=5000)


# viz the alignment
import numpy as np
def to_napari_affine(mx):
    out = np.eye(3)
    out[:2] = np.flipud(mx[:2])
    out[:, :2] = np.fliplr(out[:, :2])
    return out

import napari
v = napari.Viewer()

for rr, aa in zip(readers, aligners):
    v.add_image(
        [pp[0] for pp in rr.pyramid],
        affine=to_napari_affine(aa.affine_matrix),
        blending='additive',
        colormap='green',
        contrast_limits=(1000, 10000),
        visible=False,
        name=rr.path.name[:7]
    )


channel_names = [
    'Hoechst', 'AF1', 'CD31', 'Ki-67', 'CD68',
    'CD163', 'CD20', 'CD4', 'CD8a', 'CD45RO',
    'PD-L1', 'CD3e', 'E-Cadherin', 'PD-1','FOXP3',
    'CD45', 'Pan-CK', 'Blank', 'SMA'
]


out_dir = pathlib.Path(r'C:\Users\ychen\Downloads\YC-projects\20221130-compare_stain')
tile_sizes = (100, 300, 600)

for idx, (aa, rr) in enumerate(zip(aligners, readers)):
    for ts in tile_sizes:
        LEVEL = 0
        ref_mask = palom.img_util.block_labeled_mask(
            readers[ref_id].pyramid[LEVEL].shape[1:], (ts, ts), out_chunks=1024
        )
        mask =  palom.align.block_affine_transformed_moving_img(
            aa.moving_img.astype(np.int32),
            ref_mask,
            np.linalg.inv(aa.affine_matrix),
            is_mask=True
        )
        if idx == ref_id:
            mask = ref_mask
        print('Processing', rr.path.name[:7], '@ tile-size', ts)
        df1 = quantify_all(mask, rr.pyramid[0], channel_names=channel_names, extra_properties=[mmsc])
        df1.rename(columns=rename_mmsc, inplace=True)
        df1.to_csv(out_dir / f"{rr.path.name[:7]}-tile_{ts}-cutoff_{CUTOFF}.csv")

