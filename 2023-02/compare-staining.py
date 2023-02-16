import numpy as np


def get_lognormals(
    ln_means,
    ln_stds,
    sizes,
    plot=False
):
    samples = [
        np.random.lognormal(mm, st, ss)
        for mm, st, ss in
        zip(ln_means, ln_stds, sizes)
    ]
    idxs = [
        np.ones(aa.size, dtype=np.uint8) * idx
        for idx, aa in enumerate(samples)
    ]
    out = np.array([np.concatenate(samples), np.concatenate(idxs)])
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        samples, idxs = out
        _, axs = plt.subplots(1, 2)
        for ax, func in zip(axs, [np.asarray, np.log1p]):
            sns.histplot(
                data=func(samples),
                element='step',
                fill=False,
                stat='percent',
                ax=ax,
                color='k',
                linestyle='--',
                linewidth=2
            )
            sns.histplot(
                data=pd.DataFrame({'value': func(samples), 'id': idxs}),
                element='step',
                fill=False,
                stat='percent',
                hue='id',
                x='value',
                ax=ax
            )

    return out


def pad_size(dfs):
    num_rows = [df.shape[0] for df in dfs]
    max_row = max(num_rows)
    return max_row - np.array(num_rows, dtype=int)


def hist_iou(lines):
    assert len(lines) == 2
    l1, l2 = lines
    data1, data2 = l1.get_xydata(), l2.get_xydata()
    # make sure the x values matches
    assert np.all(data1.T[0] == data2.T[0])
    heights = [data1.T[1], data2.T[1]]
    iou = np.min(heights, axis=0).sum() / np.max(heights, axis=0).sum()
    return iou


def emd(data1, data2):
    return scipy.stats.wasserstein_distance(data1, data2)


def compare(data1, data2):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    pad1, pad2 = pad_size([data1, data2])
    samples = pd.DataFrame({
        'data1': np.pad(data1, (0, pad1), constant_values=np.nan),
        'data2': np.pad(data2, (0, pad2), constant_values=np.nan),
    })

    _, axs = plt.subplots(1, 2)
    ious = []
    emds = []
    pvals = []
    for ax, func in zip(axs, [np.asarray, np.log1p]):
        sns.histplot(
            data=samples.transform(func),
            element='step',
            fill=False,
            stat='percent',
            ax=ax,
            log_scale=(False, False)
        )
        ious.append(hist_iou(ax.lines[::-1]))
        emds.append(emd(func(data1), func(data2)))
        pvals.append(
            scipy.stats.cramervonmises_2samp(func(data1), func(data2)).pvalue
        )

    print('IoU:', ious)
    print('EMD:', emds)
    print('CVM', pvals)


for i in np.linspace(6.5, 7.5, 10):
    data1 = get_lognormals([5.5, 6.6], [.5, .5], [100_000, 10_000], plot=False)
    data2 = get_lognormals([5.5, i], [.5, .5], [100_000, 10_000], plot=False)

    compare(data1[0], data2[0])

'''
## -- End pasted text --
IoU: [0.9557904468960857, 0.9600071272662482]
EMD: [6.582837607010487, 0.008336728323483013]
CVM [0.10448520746215983, 0.10448520746215983]
IoU: [0.9598849018280297, 0.9634356397647457]
EMD: [1.4972420544610945, 0.0036917595013314067]
CVM [0.31009693330938237, 0.31009693330938237]
IoU: [0.9531165383830046, 0.957827197408584]
EMD: [9.610690347352318, 0.011730104503188064]
CVM [0.02328727974414002, 0.02328727974414002]
IoU: [0.9421163675526796, 0.9466963977595498]
EMD: [20.38777340554629, 0.02222357608390352]
CVM [1.4590941444181382e-05, 1.4590941444181382e-05]
IoU: [0.9352656579873329, 0.9347720936777213]
EMD: [31.85946580217753, 0.03265431494008045]
CVM [1.419789397161253e-09, 1.419789397161253e-09]
IoU: [0.9233210357910935, 0.9190516481886933]
EMD: [44.48016780404211, 0.04359435773516431]
CVM [1.2334910870492877e-11, 1.2334910870492877e-11]
IoU: [0.9087698901594684, 0.9056945358788677]
EMD: [56.42739516674571, 0.0500476598832799]
CVM [3.9891256964352806e-10, 3.9891256964352806e-10]
IoU: [0.8958248955146708, 0.8941926040724956]
EMD: [72.86565504531734, 0.060944508829995366]
CVM [6.537377306159442e-10, 6.537377306159442e-10]
IoU: [0.8856204948874203, 0.8844575784830185]
EMD: [91.2225793651294, 0.07247510903038262]
CVM [1.0339124001390587e-09, 1.0339124001390587e-09]
IoU: [0.8790014007037683, 0.8785040344960082]
EMD: [110.0742350097251, 0.08305246303917856]
CVM [6.017297771165886e-10, 6.017297771165886e-10]
'''

import scipy.stats

print(scipy.stats.wasserstein_distance(data1[0], data2[0]))
print(scipy.stats.wasserstein_distance(np.log1p(data1[0]), np.log1p(data2[0])))



np.random.lognormal(5.5, 2.5/3, 10000)


for i in range(5, 15):
    dd = np.random.normal(i, 0.1, 100000)
    ori_dd = np.expm1(dd)
    
    print(ori_dd.mean(), 'linear mean')
    print(np.expm1(dd).mean())
    print()
    print(ori_dd.std(), 'linear std')
    print(np.expm1(dd).mean()*dd.std())
    print()


def test(mean, std):
    data = np.random.normal(mean, std, 100000)
    print('linear mean', np.expm1(data).mean())
    print(np.expm1(data).mean())
    print('linear std', np.expm1(data).std())
    print(np.expm1(data).mean()*data.std())


