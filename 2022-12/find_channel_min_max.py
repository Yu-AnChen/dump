import tabbi
import palom, napari
import pandas as pd
import numpy as np
import skimage.feature
import dask.array as da
import dask.delayed
import matplotlib.pyplot as plt


def level2df(
    img_path,
    thumbnail_size=2000,
    channel_names=None,
    n_samples=10000,
    sample_random_state=None
):
    reader = palom.reader.OmePyramidReader(img_path)

    n_channels = len(reader.pyramid[0])
    if channel_names is None:
        channel_names = range(n_channels)

    LEVEL = reader.get_thumbnail_level_of_size(thumbnail_size)
    img = reader.pyramid[LEVEL].compute()
    mask = palom.img_util.entropy_mask(img[0]) & img.min(axis=0) > 0

    df = pd.DataFrame(img[:, mask].T, columns=channel_names)
    if n_samples is None:
        n_samples = len(df)
    n_samples = np.clip(n_samples, 1, len(df))
    return df.sample(n_samples, random_state=sample_random_state)


def biased_patch2df(
    img_path,
    thumbnail_size=2000,
    crop_size=200,
    channel_names=None,
    n_samples=10000,
    sample_random_state=None,
    kwargs_peak_local_max=None
):
    reader = palom.reader.OmePyramidReader(img_path)

    n_channels = len(reader.pyramid[0])
    if channel_names is None:
        channel_names = range(n_channels)

    LEVEL = reader.get_thumbnail_level_of_size(thumbnail_size)
    img = reader.pyramid[LEVEL].compute()
    non_blank_channels = list(range(n_channels))
    non_blank_channels.pop(5)
    mask = palom.img_util.entropy_mask(
        palom.img_util.whiten(img[0], 1)
    ) & img[non_blank_channels, ...].min(axis=0) > 0
    _, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs[0].imshow(np.log1p(img[0]))
    axs[1].imshow(mask)


    if kwargs_peak_local_max is None:
        kwargs_peak_local_max = dict(min_distance=100, num_peaks=20)
    if 'min_distance' not in kwargs_peak_local_max:
        kwargs_peak_local_max['min_distance'] = 100
    if 'num_peaks' not in kwargs_peak_local_max:
        kwargs_peak_local_max['num_peaks'] = 20

    # reduce min_distance if num_peaks isn't satisfied
    _coor_local_max = np.empty((0))
    while _coor_local_max.ndim != 3:
        _coor_local_max = np.array([
            skimage.feature.peak_local_max(c*mask, **kwargs_peak_local_max)
            for c in img
        ])
        d = kwargs_peak_local_max['min_distance']
        kwargs_peak_local_max['min_distance'] = int(0.9*d)
        assert kwargs_peak_local_max['min_distance'] >= 1
    
    rng = np.random.default_rng()
    random_coors = rng.choice(
        np.vstack(np.nonzero(mask)).T,
        1000,
        axis=0,
        replace=False
    )
    coor_local_max = np.array([
        skimage.feature.peak.ensure_spacing(
            np.vstack([c, random_coors]),
            # spacing=kwargs_peak_local_max['min_distance']
        )[:1*kwargs_peak_local_max['num_peaks']]
        # )[1*kwargs_peak_local_max['num_peaks']:2*kwargs_peak_local_max['num_peaks']]
        for c in _coor_local_max
    ])

    size = crop_size // 2
    coor_local_max *= reader.level_downsamples[LEVEL]
    coor_local_max -= size

    # return coor_local_max

    patches = dask.delayed(np.array)([
        da.array([
            reader.pyramid[0][idx, r : r + size, c : c + size] 
            for r, c in coor
        ])
        for idx, coor in enumerate(coor_local_max)
    ]).compute()
    patches = patches.reshape(19, -1)

    df_patch = pd.DataFrame(
        patches[:, patches.min(axis=0) > 0].T,
        columns=channel_names
    )
    if n_samples is None:
        n_samples = len(df_patch)
    n_samples = np.clip(n_samples, 1, len(df_patch))
    return df_patch.sample(n_samples, random_state=sample_random_state)


def plot(df, n_components):
    n_channels = df.shape[1]
    grid_shape = (1, n_channels)
    if n_channels > 4:
        grid_shape = (4, int(np.ceil(n_channels / 4)))
    tabbi.gmm.plot_hist_gmm(
        df,
        df.columns,
        n_components,
        grid_shape
    )


def get_min_max(df, n_components, max_percentile=99.9):
    mins = [
        np.expm1(tabbi.gmm.get_gmm_and_pos_label(v, n_components=n_components)[2])
        for v in df.transform(np.log1p).T.values
    ]
    maxs = [
        [np.percentile(v[v>mm], max_percentile) for mm in m]
        for (m, v) in zip(mins, df.T.values)
    ]
    return mins, maxs


def napari_check(img_path, mins, maxs, channel_names=None):
    reader = palom.reader.OmePyramidReader(img_path)
    v = napari.Viewer()
    v.add_image(
        reader.pyramid,
        channel_axis=0,
        contrast_limits=[ms for ms in zip(mins, maxs)],
        visible=False,
        colormap='cyan',
        name=channel_names
    )
    return v


import pathlib
import json

def replace_json_min_max(json_path, mins, maxs, out_path, img_path=None):
    
    def replace_min_max(channels):
        out_channels = list(channels)
        for c in out_channels:
            c.update(dict(min=mins[c['id']], max=maxs[c['id']]))
        return out_channels

    with open(json_path) as f:
        in_json = json.load(f)
    
    for group in in_json['groups']:
        group['channels'] = replace_min_max(group['channels'])
        group['render'] = replace_min_max(group['render'])
    
    if img_path is not None:
        in_json['in_file'] = str(img_path)

    with open(out_path, "w") as wf:
        json.dump(in_json, wf)

json_template_path = pathlib.Path(
    r'Z:\RareCyte-S3\YC-analysis\minerva-P37_CRC\scripts\orion-template.json'
)
channel_names = [
    'Hoechst', 'AF1', 'CD31', 'CD45', 'CD68', 'Argo550', 'CD4', 'FOXP3', 'CD8a',
    'CD45RO', 'CD20', 'PD-L1', 'CD3e', 'CD163', 'E-cadherin', 'PD-1', 'Ki67', 'Pan-CK', 'SMA'
]

files = pd.read_csv(r'Z:\RareCyte-S3\YC-analysis\minerva-P37_CRC\scripts\file_list.csv')
plot_out_dir = pathlib.Path(r'Z:\RareCyte-S3\YC-analysis\minerva-P37_CRC\scripts\plots-gmm')
json_out_dir = pathlib.Path(r'Z:\RareCyte-S3\YC-analysis\minerva-P37_CRC\scripts\channel-json')

for n, p in zip(files["name"], files["path"]):
    img_path = pathlib.Path(p)
    print('Processing', img_path.name)

    # df1 = level2df(img_path, channel_names=channel_names)
    # plot(df1, n_components=3)
    # mins1, maxs1 = get_min_max(df1, n_components=3)

    df2 = biased_patch2df(
        img_path,
        channel_names=channel_names,
        sample_random_state=1001,
        kwargs_peak_local_max=dict(min_distance=100, num_peaks=20)
    )
    
    plot(df2, n_components=2)
    fig = plt.gcf()
    fig.set_size_inches(18, 12)
    fig.suptitle(f"{img_path.name} (biased GMM)")
    fig.tight_layout()
    fig.savefig(plot_out_dir / f"{img_path.stem}-gmm.png", dpi=144)

    mins2, maxs2 = get_min_max(df2, n_components=2, max_percentile=95)
    mins2 = np.array([m[0] for m in mins2]) / 65535
    maxs2 = np.array([m[0] for m in maxs2]) / 65535

    replace_json_min_max(
        json_template_path,
        mins2, maxs2,
        json_out_dir / f"{n}.story.json",
        img_path=img_path.resolve()
    )


    command_dir = pathlib.Path(r'Z:\RareCyte-S3\YC-analysis\minerva-P37_CRC\scripts\pyramid-command')
    pyramid_cmd = '''
    python save_exhibit_pyramid.py --force
    "{}"
    "{}"
    "{}"
    '''
    minerva_dir = pathlib.Path(r'Z:\RareCyte-S3\YC-analysis\minerva-P37_CRC\orion-crc')
    pyramid_cmd = pyramid_cmd.format(
        img_path,
        json_out_dir / f"{n}.story.json",
        minerva_dir / n
    )
    with open(command_dir / f"{n}-command.txt", 'w') as f:
        f.write(
            pyramid_cmd
            .replace('\n', ' ')
            .replace('\t', '')
            .strip()
        )

    # v2 = napari_check(img_path, [m[0] for m in mins2], [m[0] for m in maxs2], channel_names)
