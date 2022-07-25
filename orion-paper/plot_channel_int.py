import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches


def set_matplotlib_font():
    font_families = matplotlib.rcParams['font.sans-serif']
    if font_families[0] != 'Arial':
        font_families.insert(0, 'Arial')
    matplotlib.rcParams['pdf.fonttype'] = 42


def plot_corr(
    corr, name='', 
    grouping=None, grouping_colors=None,
    highlighted_fluors=None,
    heatmap_kwargs=None
):
    set_matplotlib_font()

    plt.figure(figsize=(6.5, 7))
    plt.gcf().patch.set_alpha(0)
    _heatmap_kwargs = dict(
        cmap='coolwarm',
        center=0,
        linewidths=1,
        square=True,
        cbar=False
    )
    if heatmap_kwargs is None:
        heatmap_kwargs = {}
    heatmap_kwargs = {**_heatmap_kwargs, **heatmap_kwargs}
    ax = sns.heatmap(corr, **heatmap_kwargs)
    ax.set_xlim(np.array([-.2, .2])+ax.get_xlim())
    ax.set_ylim(np.array([.2, -.2])+ax.get_ylim())
    plt.suptitle(name)

    if grouping is not None:
        orion_ori = np.cumsum([0] + grouping)

        ax = plt.gca()
        for o, w, c in zip(orion_ori, grouping, grouping_colors):
            ax.add_patch(mpatches.Rectangle(
                (0, o), 0, w, edgecolor=c,        
                fill=False, lw=5
            ))
    if highlighted_fluors is not None:
        for o in highlighted_fluors:
            ax.add_patch(mpatches.Rectangle(
                (o, o), 1, 1, edgecolor='black',        
                fill=False, lw=3
            ))


df_index = [
    'Hoechst',
    'AF',
    'ArgoFluor 515',
    'ArgoFluor 555L',
    'ArgoFluor 535',
    'ArgoFluor 572',
    'ArgoFluor 584',
    'ArgoFluor 602',
    'ArgoFluor 624',
    'ArgoFluor 660L',
    'ArgoFluor 662',
    'ArgoFluor 686',
    'ArgoFluor 706',
    'ArgoFluor 730',
    'ArgoFluor 760',
    'ArgoFluor 795',
    'ArgoFluor 845',
    'ArgoFluor 875'
]

colors = [
    '#9933ff',
    '#00e6e6',
    '#00e600',
    '#ffff00',
    '#ffbf00',
    '#ff4000',
    '#cc0000'
]

orion_ex_group = [1, 1, 2, 1, 5, 4, 4]

# Plot extraction heatmaps
for i, name in zip(range(2), ['before', 'after']):
    df = pd.read_excel('Extraction_P37_A28_Stringent_220721.xlsx', sheet_name=i)
    ddf = df.iloc[-18:, -18:]
    ddf.index = df_index
    ddf.columns = np.arange(18) + 1
    plot_corr(
        ddf.T,
        grouping=orion_ex_group,
        grouping_colors=colors,
        heatmap_kwargs=dict(cmap='viridis', center=None, vmin=-.1, vmax=1)
    )
    fig = plt.gcf()
    fig.set_figwidth(4)
    fig.savefig(f"extraction-{name}.pdf")

# Plot colorbar for extraction heatmaps
fig, ax = plt.subplots()
plt.colorbar(
    mappable=matplotlib.cm.ScalarMappable(
        matplotlib.colors.Normalize(vmin=-.1, vmax=1),
        cmap='viridis'
    ),
    cax=ax
)
fig.set_figwidth(.3)
fig.set_figheight(1.5)
plt.tight_layout()
fig.savefig('extraction-cbar.pdf')



# Plot spectral scans
df_spectra = pd.read_excel('Bead Crosstalk Calculator modular -DEC for CRC Paper.xlsx', sheet_name='paper')
df_spectra = df_spectra.set_index('Laser').iloc[2:]
excitations = np.floor(df_spectra.columns.astype(float))

ex_unique = np.unique(excitations)

# colors = matplotlib.cm.rainbow(
#     (ex_unique - min(ex_unique)) / 
#     (max(ex_unique) - min(ex_unique))
# )


fig, axs = plt.subplots(len(ex_unique), 1, sharex=True, sharey=False)
plt.subplots_adjust(hspace=0.05)

fluor_id = 1
for idx, ex in enumerate(ex_unique):
    ax = axs[idx]
    df = df_spectra.loc[:, excitations == ex]
    ax.plot(df_spectra.index, df, c='k')
    ax.axvline(ex, c=colors[idx])
    ax.vlines(df.astype(float).idxmax(), 0, 1, colors='#333333', alpha=0.5)
    ax.set_ylim(-.1, 1.5)
    for xx in df.astype(float).idxmax().values:
        ax.annotate(fluor_id, (xx, 1.1), fontsize=8)
        fluor_id += 1
    ax.set_yticks([1.4/2])
    ax.set_yticklabels([int(ex)])
    ax.yaxis.set_ticks_position('none')

for ax in axs[:-1]:
    ax.xaxis.set_ticks_position('none')

fig.set_figwidth(4)
fig.savefig("spectra.pdf")

