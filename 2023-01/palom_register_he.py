import palom
import numpy as np
import pathlib


orion_dir = pathlib.Path(r'Z:\RareCyte-S3\P54_CRCstudy_Bridge')
orion_files = '''
P54_S32_Full_Or6_A31_C90c_HMS@20221025_001512_517216.ome.tiff
P54_S33_Full_Or6_A31_C90c_HMS@20221025_001610_632297.ome.tiff
P54_S34_Full_Or6_A31_C90c_HMS@20221028_061306_306312.ome.tiff
'''

he_dir = pathlib.Path(r'X:\crc-scans\histowiz scans\20230105-orion_2_cycles')
he_files = '''
22199$P54_32_HE$US$SCAN$OR$001 _105745.svs
22199$P54_33_HE$US$SCAN$OR$001 _104050.svs
22199$P54_34_HE$US$SCAN$OR$001 _103804.svs
'''

orion_files = [orion_dir / pp for pp in orion_files.strip().split('\n')][:1]
he_files = [he_dir / pp for pp in he_files.strip().split('\n')][:1]

# Specify pixel size to enable auto-detect thumbnail level
c1rs = [palom.reader.OmePyramidReader(pp, pixel_size=0.325) for pp in orion_files]
c2rs = [palom.reader.SvsReader(pp) for pp in he_files]
for reader in c2rs:
    # Flip horizontally 
    reader.pyramid = [np.flip(pp, 2) for pp in reader.pyramid]

aligners = [
    palom.align.get_aligner(
        r1, r2,
        thumbnail_level1=None,
        # Use autofluoroscence channel for coarse alignment
        channel1=1, channel2=2
    )
    for r1, r2 in zip(c1rs, c2rs)
]
for aa, r1 in zip(aligners, c1rs):
    aa.coarse_register_affine(n_keypoints=10000)
    # Switch back to Hoechst channel for block alignment
    aa.ref_img = r1.read_level_channels(0, 0)
    aa.compute_shifts()
    palom.align.viz_shifts(aa.shifts, aa.grid_shape)
    aa.constrain_shifts()

mosaics = [
    palom.align.block_affine_transformed_moving_img(
        ref_img=aa.ref_img,
        moving_img=r2.pyramid[0],
        mxs=aa.block_affine_matrices_da
    )
    for aa, r2 in zip(aligners, c2rs)
]

out_dir = pathlib.Path(r'X:\crc-scans\histowiz scans\20230105-orion_2_cycles\registered')
for mm, r2 in zip(mosaics, c2rs):
    palom.pyramid.write_pyramid(
        mosaics=[mm],
        output_path=out_dir / f"{r2.path.stem}-registered.ome.tif",
        pixel_size=0.65,
        channel_names=[list('RBG')]
    )
