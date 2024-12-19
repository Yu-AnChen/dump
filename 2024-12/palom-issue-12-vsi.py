import pathlib

import matplotlib.pyplot as plt

import palom
from palom.cli.align_he import save_all_figs, set_subplot_size


# ------------------- configure file path and channel names ------------------ #
# Only test the first 3 cycles (slice the first 3 file paths and channel names)
files = r"""
C:\Users\me\Downloads\CLL14_CyCIF\CLL14_NoPrimary_JustSecondary_488-550-647.vsi
C:\Users\me\Downloads\CLL14_CyCIF\CLL14_Round1_CD4-488_FOXP3-550_LAG3-647.vsi
C:\Users\me\Downloads\CLL14_CyCIF\CLL14_Round2_CCL3-488_CD68-550_KI67-647.vsi
C:\Users\me\Downloads\CLL14_CyCIF\CLL14_Round3-2_CD20-488_CD57-550_CD8-647.vsi
C:\Users\me\Downloads\CLL14_CyCIF\CLL14_Round4-2_GranzymeB-488_CD45PL-550_CD3-647.vsi
C:\Users\me\Downloads\CLL14_CyCIF\CLL14_Round5_no-488_no-550_PD1-647.vsi
""".strip().split("\n")[:3]
channel_names = [
    ["DAPI", "Background-488", "Background-550", "Background-647"],
    ["DAPI", "CD4", "FOXP3", "LAG3"],
    ["DAPI", "CCL3", "CD68", "KI67"],
    ["DAPI", "CD20", "CD57", "CD8"],
    ["DAPI", "Granzyme B", "CD45PL", "CD3"],
    ["DAPI", "Empty-488", "Empty-550", "Empty-647"],
][:3]


# ----------------------- some essential settings here ----------------------- #
# There are actually 2 "scenes" in the VSI, see the following `print` block
scene = 1
# Specify which pyramid level (resolution) one wants to align; one could use a
# lower resolution, such as `full_res_level = 2` for a quick test of the
# alignment
full_res_level = 0
slide_id = "CLL14_CyCIF"
out_dir = (
    pathlib.Path(r"C:\Users\me\Downloads\CLL14_CyCIF") / f"registered-scene-{scene}"
)

# --------------------- initiate the readers and aligners -------------------- #
readers = [palom.reader.VsiReader(pp, scene=scene) for pp in files]

c1r = readers[0]
print("Number of scenes in vsi file:", c1r.store._slide.num_scenes)
print(f"Processing scene {c1r.scene}")
# Use thumbnails of size ~1000 pixels for coarse alignment
thumbnail_level = c1r.get_thumbnail_level_of_size(1000)

aligners = [
    palom.align.get_aligner(
        c1r,
        rr,
        level1=full_res_level,
        level2=full_res_level,
        thumbnail_level1=thumbnail_level,
        thumbnail_level2=thumbnail_level,
    )
    for rr in readers[1:]
]

for idx, aa in enumerate(aligners):
    # Run coarse alignment, use higher number of keypoints than default for
    # exhaustive feature detection and matching
    aa.coarse_register_affine(n_keypoints=10_000)

    # Save coarse alignment plots
    p1 = c1r.path
    p2 = readers[idx + 1].path
    fig, ax = plt.gcf(), plt.gca()
    fig.suptitle(f"{p2.name} (coarse alignment)", fontsize=8)
    ax.set_title(f"{p1.name} - {p2.name}", fontsize=6)
    im_h, im_w = ax.images[0].get_array().shape
    set_subplot_size(im_w / 288, im_h / 288, ax=ax)
    ax.set_anchor("N")
    fig.subplots_adjust(top=1 - 0.5 / fig.get_size_inches()[1])
    save_all_figs(out_dir=out_dir / "qc", format="jpg", dpi=144)

    # Run "full resolution" block-wise phase correlation (translation only)
    aa.compute_shifts()

    # Save the refinement visulization plot
    fig = aa.plot_shifts()
    fig.suptitle(f"{p2.name} (block shift distance)", fontsize=8)
    fig.axes[0].set_title(p1.name, fontsize=6)
    save_all_figs(out_dir=out_dir / "qc", format="png")

    # Throw away outliers in the phase correlation result
    aa.constrain_shifts()

# ------------------------ final mosaics construction ------------------------ #
# `mosaics` has all the channels from all the cycles
mosaics = [c1r.pyramid[full_res_level]]
# `mosaics_skip_dapi` contain DAPI from the first cycle but not from the other
# cycles
mosaics_skip_dapi = [c1r.pyramid[full_res_level]]

for rr, aa in zip(readers[1:], aligners):
    mm = palom.align.block_affine_transformed_moving_img(
        # `ref_img` is just for specifying the "canvas" of the output image 
        ref_img=c1r.pyramid[full_res_level][0],
        moving_img=rr.pyramid[full_res_level],
        # local affine matrix from global affine + translation
        mxs=aa.block_affine_matrices_da,
    )
    mosaics.append(mm)
    # slice out DAPI
    mosaics_skip_dapi.append(mm[1:])

pyramid_kwargs = dict(
    # set the pixel size based on `full_res_level`
    pixel_size=c1r.level_downsamples[full_res_level] * c1r.pixel_size,
    downscale_factor=2,
    tile_size=1024,
    compression="zlib",
    save_RAM=True,
)
palom.pyramid.write_pyramid(
    mosaics=mosaics,
    output_path=out_dir / f"{slide_id}.ome.tif",
    channel_names=channel_names,
    **pyramid_kwargs,
)
palom.pyramid.write_pyramid(
    mosaics=mosaics_skip_dapi,
    output_path=out_dir / f"{slide_id}-skip-dapi.ome.tif",
    # set channel names for skip DAPI image
    channel_names=[channel_names[0]] + [nn[1:] for nn in channel_names[1:]],
    **pyramid_kwargs,
)
