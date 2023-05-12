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