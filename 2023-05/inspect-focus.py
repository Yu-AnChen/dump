import skimage.measure
import skimage.util
import tifffile
from joblib import Parallel, delayed
import cv2
import functools
import numpy as np


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



import skimage.filters
import pathlib
import logging

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
    qc_mask = lvar_img > skimage.filters.threshold_triangle(lvar_img)
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
        fig.suptitle(img_path.stem)
        fig.set_size_inches(fig.get_size_inches()*2)
        fig.savefig(out_dir / f"{img_path.stem}-qc.png", dpi=144, bbox_inches='tight')


# ---------------------------------------------------------------------------- #
#                                 plotting code                                #
# ---------------------------------------------------------------------------- #
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable, Size
import numpy as np


def make_img_axes(
    ax,
    num_panels_per_stripe=None,
    stripe_direction='vertical',
    axes_class=None, **kwargs
):  
    assert stripe_direction in ['vertical', 'horizontal']
    divider = make_axes_locatable(ax)

    num_stripes = 1

    total_stripe_size = Size.AxesX(ax) if stripe_direction == 'vertical' else Size.AxesY(ax)
    stripe_sizes = [1/n*total_stripe_size for n in num_panels_per_stripe]

    total_panel_size = Size.AxesY(ax) if stripe_direction == 'vertical' else Size.AxesX(ax)
    total_panel_grid_size = np.prod(num_panels_per_stripe)
    unit_panel_size = 1/total_panel_grid_size * total_panel_size
    panel_sizes = [unit_panel_size]*total_panel_grid_size
    panel_spans = [int(total_panel_grid_size / n) for n in num_panels_per_stripe]

    if stripe_direction == 'vertical':
        divider.set_horizontal(stripe_sizes)
        divider.set_vertical(panel_sizes)
    else:
        # start at upper left, flip the vertical sizes
        divider.set_vertical(stripe_sizes[::-1])
        divider.set_horizontal(panel_sizes)

    axs = []
    if axes_class is None:
        try:
            axes_class = ax._axes_class
        except AttributeError:
            axes_class = type(ax)
    for stripe_pos, (ns, ps) in enumerate(
        zip(num_panels_per_stripe, panel_spans)
    ):
        for _panel_pos in range(ns):
            
            ax1 = axes_class(ax.get_figure(), ax.get_position(original=True),
                            sharex=ax, sharey=ax, **kwargs)
            if stripe_pos == _panel_pos == 0:
                ax1 = ax
            
            if stripe_direction == 'vertical':
                panel_pos = ns - 1 - _panel_pos
                locator = divider.new_locator(
                    nx=stripe_pos,
                    ny=panel_pos*ps, ny1=(panel_pos+1)*ps
                )
            else:
                panel_pos = _panel_pos
                locator = divider.new_locator(
                    ny=len(num_panels_per_stripe) - 1 - stripe_pos,
                    nx=panel_pos*ps, nx1=(panel_pos+1)*ps
                )
            
            ax1.set_axes_locator(locator)
            
            for t in ax1.yaxis.get_ticklabels() + ax1.xaxis.get_ticklabels():
                t.set_visible(False)
            try:
                for axis in ax1.axis.values():
                    axis.major_ticklabels.set_visible(False)
            except AttributeError:
                pass
            
            axs.append(ax1)
    
    fig = ax.get_figure()
    for ax1 in axs:
        fig.add_axes(ax1)

    return axs


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
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
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

    contour_tissue = dict(levels=[.5], linewidths=1, colors=color_contour_tissue)
    contour_qc = dict(levels=[.5], linewidths=1, colors=color_contour_qc)
    
    import skimage.morphology

    entropy_mask = skimage.morphology.remove_small_objects(entropy_mask, 4)
    qc_mask = skimage.morphology.remove_small_objects(qc_mask, 4)

    axs[0].contour(entropy_mask, **contour_tissue)
    axs[0].contour(qc_mask, **contour_qc)
    axs[1].contour(entropy_mask, **contour_tissue)
    axs[2].contour(qc_mask, **contour_qc)
    
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
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
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
