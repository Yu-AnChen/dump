import pathlib
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np

from ashlar import reg


files = sorted(
    pathlib.Path(
        r"\\files.med.harvard.edu\HITS\lsp-data\cycif-production\183-Gray_Breast\Orion_plus_FISH_20241206"
    ).glob("*/*402.rcpnl")
)
readers = [reg.BioformatsReader(str(ff)) for ff in files]
aligners = [
    reg.EdgeAligner(
        rr, verbose=True, max_shift=30, filter_sigma=1, do_make_thumbnail=True
    )
    for rr in readers
]
for aa in aligners:
    aa.run()

plt.figure()
plt.gca().set_aspect("equal")
plt.gca().invert_yaxis()
for ii, aa in enumerate(aligners):
    p_nominal = aa.metadata.positions
    p_stitched = aa.positions + aa.origin

    l1, l2 = None, None
    if ii == 0:
        l1, l2 = "Nominal", "Stitched"
    plt.scatter(*p_nominal.T[::-1], color="royalblue", marker="x", label=l1)
    plt.scatter(*p_stitched.T[::-1], color="orange", marker=".", label=l2)

    affine = skimage.transform.AffineTransform()
    affine.estimate(np.fliplr(p_nominal), np.fliplr(p_stitched))
    np.rad2deg(affine.rotation)
    plt.text(
        *np.subtract(p_nominal.min(axis=0)[::-1], [100, 300]),
        f"Detected rotation: {np.rad2deg(affine.rotation):.3f} degrees",
    )
