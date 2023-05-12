import functools

import cv2
import numpy as np
import skimage.measure
import skimage.util
import tifffile
from joblib import Parallel, delayed


def shannon_entropy(img, block_size):
    assert img.ndim == 2
    wimg = skimage.util.view_as_windows(img, block_size, block_size)
    h, w, *_ = wimg.shape
    out = np.zeros((h, w))
    out.flat = Parallel(n_jobs=4)(
        delayed(skimage.measure.shannon_entropy)(wimg[i, j])
        for i, j in np.mgrid[:h, :w].reshape(2, -1).T
    )
    return out


def var_of_laplacian(img, block_size, sigma=0):
    assert img.ndim == 2
    wimg = skimage.util.view_as_windows(img, block_size, block_size)
    h, w, *_ = wimg.shape
    out = np.zeros((h, w))

    func = lambda x: x
    if sigma != 0:
        func = functools.partial(cv2.GaussianBlur, ksize=(0, 0), sigmaX=sigma)
    for i, j in np.mgrid[:h, :w].reshape(2, -1).T:
        out[i, j] = np.var(
            cv2.Laplacian(func(wimg[i, j]), cv2.CV_32F, ksize=1)
        )
    return out


def var_block(img, block_size):
    assert img.ndim == 2
    wimg = skimage.util.view_as_windows(img, block_size, block_size)
    return np.var(wimg, axis=(2, 3))


def mean_block(img, block_size):
    assert img.ndim == 2
    wimg = skimage.util.view_as_windows(img, block_size, block_size)
    return np.mean(wimg, axis=(2, 3))


import sklearn.mixture


def gmm_cutoffs(var_img, plot=False):
    gmm = sklearn.mixture.GaussianMixture(n_components=3)
    limg = np.sort(np.log1p(var_img.flat))
    labels = gmm.fit_predict(limg.reshape(-1, 1))
    diff_idxs = np.where(np.diff(labels))
    diffs = np.mean(
        (limg[diff_idxs[0]], limg[diff_idxs[0]+1]),
        axis=0
    )
    filtered_diffs = diffs[
        (diffs > gmm.means_.min()) & (diffs < gmm.means_.max())
    ]
    if plot:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        h, *_ = ax.hist(limg, bins=200)
        ax.vlines(filtered_diffs, 0, h.max(), colors='salmon')
    return np.expm1(filtered_diffs)


import logging
import pathlib

import skimage.filters

logging.basicConfig( 
    format="%(asctime)s | %(levelname)-8s | %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S", 
    level=logging.INFO 
)


def process_file(img_path, plot=False, out_dir=None):
    img_path = pathlib.Path(img_path)
    
    logging.info(f"Reading {img_path}")
    img = tifffile.imread(img_path, key=0)

    logging.info("Computing entropy")
    entropy_img = shannon_entropy(img, 128)
    tissue_mask = entropy_img > skimage.filters.threshold_triangle(entropy_img)

    logging.info("Computing variance")
    lvar_img = var_of_laplacian(img, 128, 1)
    # qc_img = lvar_img / (mean_block(img, 128)+1)
    # qc_img = np.nan_to_num(qc_img)
    # qc_mask = qc_img > skimage.filters.threshold_triangle(qc_img)
    # qc_mask = lvar_img > skimage.filters.threshold_triangle(lvar_img)
    qc_mask = lvar_img >= gmm_cutoffs(lvar_img)[1]
    iou = (tissue_mask & qc_mask).sum() / (tissue_mask | qc_mask).sum()

    text = f"""

    Tissue area fraction {100*tissue_mask.sum() / tissue_mask.size:.1f}%
    Good quality area fraction {100*qc_mask.sum() / qc_mask.size:.1f}%
    IoU {iou*100:.1f}%

    """
    logging.info(text)
    logging.info('Done')

    if plot:
        if out_dir is None:
            out_dir = img_path.stem
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        fig = plot_tissue_quality(
            mean_block(img, 32),
            entropy_img,
            tissue_mask,
            lvar_img,
            qc_mask,
            thumbnail_extent_factor=32/128
        )
        fig.suptitle(img_path.name, y=.90, va='top')
        fig.set_size_inches(fig.get_size_inches()*2)
        fig.savefig(out_dir / f"{img_path.stem}-qc.png", dpi=144, bbox_inches='tight')


# ---------------------------------------------------------------------------- #
#                                 plotting code                                #
# ---------------------------------------------------------------------------- #
from mpl_img_grid import make_img_axes


def plot_tissue_quality(
    thumbnail,
    entropy_img,
    entropy_mask,
    focus_img,
    qc_mask,
    thumbnail_extent_factor=1,
    cmap='cividis',
    color_contour_tissue='magenta',
    color_contour_qc='cyan',
    stripe_direction='auto',
    plot_colorbar=True
):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    assert stripe_direction in ['auto', 'horizontal', 'vertical']
    if stripe_direction == 'auto':
        stripe_direction = 'vertical'
        if np.divide(*thumbnail.shape) < 1:
            stripe_direction = 'horizontal'

    fig, ax = plt.subplots()
    axs = make_img_axes(ax, [1, 2], stripe_direction=stripe_direction)

    im0 = axs[0].imshow(
        X=thumbnail.astype(float)+1,
        norm=mcolors.LogNorm(),
        extent=(
            0, thumbnail.shape[1]*thumbnail_extent_factor,
            thumbnail.shape[0]*thumbnail_extent_factor, 0
        ),
        cmap=cmap
    )
    im1 = axs[1].imshow(entropy_img, cmap=cmap)
    im2 = axs[2].imshow(focus_img+1, norm=mcolors.LogNorm(), cmap=cmap)

    contour_kwargs = dict(levels=[.5], linewidths=0.5)
    
    import skimage.morphology

    entropy_mask = skimage.morphology.remove_small_objects(entropy_mask, 4)
    qc_mask = skimage.morphology.remove_small_objects(qc_mask, 4)

    axs[0].contour(entropy_mask, colors=color_contour_tissue, **contour_kwargs)
    axs[0].contour(qc_mask, colors=color_contour_qc, **contour_kwargs)
    axs[1].contour(entropy_mask, colors=color_contour_tissue, **contour_kwargs)
    axs[2].contour(qc_mask, colors=color_contour_qc, **contour_kwargs)
    
    for pax in axs:
        pax.tick_params(direction="in")

    if plot_colorbar:

        cbar_kwargs = dict(height='50%', loc='lower left', borderpad=0)
        cax0 = inset_axes(
            axs[0], width='4%',
            bbox_to_anchor=(-0.10, 0., 1, 1),
            bbox_transform=axs[0].transAxes,
            **cbar_kwargs
        )

        bbox = (1.10, 0., 1, 1)
        if stripe_direction == 'horizontal':
            bbox = (-0.20, 0., 1, 1)
        cax1 = inset_axes(
            axs[1], width='8%',
            bbox_to_anchor=bbox,
            bbox_transform=axs[1].transAxes,
            **cbar_kwargs
        )
        
        cax2 = inset_axes(
            axs[2], width='8%',
            bbox_to_anchor=(1.10, 0., 1, 1),
            bbox_transform=axs[2].transAxes,
            **cbar_kwargs
        )

        fig.colorbar(im0, cax=cax0)
        fig.colorbar(im1, cax=cax1)
        fig.colorbar(im2, cax=cax2)

        cax0.yaxis.set_ticks_position('left')
        if stripe_direction == 'horizontal':
            cax1.yaxis.set_ticks_position('left')
        cax1.axhline(entropy_img[entropy_mask].min(), c=color_contour_tissue)
        cax2.axhline(focus_img[qc_mask].min(), c=color_contour_qc)

    iou = (entropy_mask & qc_mask).sum() / (entropy_mask | qc_mask).sum()

    annot_kwargs = dict(
        xy=(0, 0), xytext=(5, -5),
        textcoords='offset points', va='top',
        bbox=dict(facecolor='black', alpha=0.5),
    )
    axs[0].annotate(
        f"{iou*100:.1f}% IoU ",
        color='white',
        **annot_kwargs
    )
    axs[1].annotate(
        f"{100*entropy_mask.sum() / entropy_mask.size:.1f}% tissue area",
        color=color_contour_tissue,
        **annot_kwargs
    )
    axs[2].annotate(
        f"{100*qc_mask.sum() / qc_mask.size:.1f}% quality area",
        color=color_contour_qc,
        **annot_kwargs
    )

    return fig


def set_matplotlib_font():
    import matplotlib
    font_families = matplotlib.rcParams['font.sans-serif']
    if font_families[0] != 'Arial':
        font_families.insert(0, 'Arial')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 10


# ---------------------------------------------------------------------------- #
#                                test one image                                #
# ---------------------------------------------------------------------------- #
set_matplotlib_font()
process_file(r"X:\cycif-production\149-Orion-Awad_Batch2\LSP16096new_P54_A31_C100_HMS_Orion7@20230406_173306_581461.ome.tiff", out_dir=r'X:\cycif-production\149-Orion-Awad_Batch2\qc', plot=True)

# curr = pathlib.Path(r'X:\cycif-production\149-Orion-Awad_Batch2')
# ometiffs = sorted(curr.glob('*.ome.tiff'))
# for p in ometiffs:
#     process_file(p, out_dir=r'X:\cycif-production\149-Orion-Awad_Batch2\qc', plot=True)


# ---------------------------------------------------------------------------- #
#                                  dev section                                 #
# ---------------------------------------------------------------------------- #
def plot_tissue_quality_basic(
    thumbnail,
    entropy_img,
    entropy_mask,
    focus_img,
    qc_mask,
    thumbnail_extent_factor=1
):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, sharex=True, sharey=True)
    imshow_kwargs = dict(
        X=thumbnail.astype(float)+1,
        norm=mcolors.LogNorm(),
        extent=(
            0, thumbnail.shape[1]*thumbnail_extent_factor,
            thumbnail.shape[0]*thumbnail_extent_factor, 0
        )
    )
    p1 = axs[0].imshow(**imshow_kwargs)
    fig.colorbar(p1, ax=axs[0], location='bottom')
    
    p2 = axs[1].imshow(entropy_img)
    axs[1].contour(entropy_mask, levels=[.5], colors='magenta')
    fig.colorbar(p2, ax=axs[1], location='bottom')
    
    p3 = axs[2].imshow(focus_img)
    axs[2].contour(qc_mask, levels=[.5], colors='cyan')
    fig.colorbar(p3, ax=axs[2], location='bottom')

    p4 = axs[3].imshow(**imshow_kwargs)
    plt.contour(entropy_mask, levels=[.5], colors='magenta')
    plt.contour(qc_mask, levels=[.5], colors='cyan')
    fig.colorbar(p4, ax=axs[3], location='bottom')

    return fig