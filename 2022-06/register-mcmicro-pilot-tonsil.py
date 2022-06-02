import palom
import numpy as np
import skimage

c1r = palom.reader.OmePyramidReader(r"Y:\sorger\data\Broad\mcmicro_paper\TNP_pilot_cycif\registration\TNP_pilot_cycif.ome.tif")
c2r = palom.reader.OmePyramidReader(r"Y:\sorger\data\Broad\mcmicro_paper\TNP_pilot_mIHC\registration\ashlar-75684-rotation-deconvolved-v1.ome.tif")
c3r = palom.reader.OmePyramidReader(r"Y:\sorger\data\Broad\mcmicro_paper\TNP_pilot_codex\registration\PilotTonsil_5_z08.ome.tif")

LEVEL = 0
THUMBNAIL_LEVEL = 5

c21l = palom.align.Aligner(
    ref_img=c1r.read_level_channels(LEVEL, 0),
    moving_img=c2r.read_level_channels(LEVEL, 1),
    ref_thumbnail=c1r.read_level_channels(THUMBNAIL_LEVEL, 10).compute(),
    moving_thumbnail=c2r.read_level_channels(THUMBNAIL_LEVEL, -3).compute(),
    ref_thumbnail_down_factor=c1r.level_downsamples[THUMBNAIL_LEVEL] / c1r.level_downsamples[LEVEL],
    moving_thumbnail_down_factor=c2r.level_downsamples[THUMBNAIL_LEVEL] / c2r.level_downsamples[LEVEL]
)


def to_napari_affine(mx):
    out = np.eye(3)
    out[:2, -1] = mx[:2, -1][::-1]
    out[:2, :2] = np.fliplr(np.flipud(mx[:2, :2]))
    return out


c31l = palom.align.Aligner(
    ref_img=c1r.read_level_channels(LEVEL, 0),
    moving_img=c2r.read_level_channels(LEVEL, 1),
    ref_thumbnail=c1r.read_level_channels(3, 10).compute()[2000:2600, 1000:1500],
    moving_thumbnail=np.rot90(c3r.read_level_channels(4, 13).compute()),
    ref_thumbnail_down_factor=8,
    moving_thumbnail_down_factor=16
)

c31l.coarse_register_affine(n_keypoints=20000)
affine = skimage.transform.AffineTransform
mx_ref = affine(scale=1/c31l.ref_thumbnail_down_factor).params
mx_ref2 = affine(translation=(-1000, -2000)).params
mx_moving = affine(scale=1/c31l.moving_thumbnail_down_factor, rotation=np.deg2rad(-90)).params
affine_matrix = (
    np.linalg.inv(mx_ref) @
    np.linalg.inv(mx_ref2) @
    c31l.coarse_affine_matrix.copy() @
    mx_moving
)
affine_matrix = affine(translation=(-300, 3600)).params @ affine_matrix

tform = affine(affine_matrix)

r_s, c_s = 0, 0
r_e, c_e = c3r.pyramid[0].shape[1:]

coords = tform([
    [c_s, r_s],
    [c_e, r_s],
    [c_e, r_e],
    [c_s, r_e]

])

c_ss, r_ss = coords.min(axis=0)
c_ee, r_ee = coords.max(axis=0)

# [17035:20694, 8739:11883]
