import palom
import numpy as np
import napari
import ome_types
import re
import skimage.transform
import itertools


# ---------------------------------------------------------------------------- #
#                  Coarse to fine approach gives better result                 #
# ---------------------------------------------------------------------------- #
def crop_around_center(aligner, center_yx, center_in="moving", crop_size=1000):
    assert len(center_yx) == 2
    y, x = center_yx
    assert center_in in ["ref", "moving"]
    tform = aligner.tform
    if center_in == "ref":
        tform = aligner.tform.inverse
    half = int(crop_size / 2)

    o_vertices = list(itertools.product([x - half, x + half], [y - half, y + half]))
    t_vertices = tform(o_vertices)

    o_cs, o_rs = np.floor(o_vertices).min(axis=0).astype(int)
    o_ce, o_re = np.ceil(o_vertices).max(axis=0).astype(int)
    t_cs, t_rs = np.floor(t_vertices).min(axis=0).astype(int)
    t_ce, t_re = np.ceil(t_vertices).max(axis=0).astype(int)

    if center_in == "moving":
        o_img = aligner.moving_img
        t_img = aligner.ref_img
    else:
        o_img = aligner.ref_img
        t_img = aligner.moving_img

    o_h, o_w = o_img.shape
    t_h, t_w = t_img.shape

    o_rs, o_re = np.clip([o_rs, o_re], 0, o_h)
    o_cs, o_ce = np.clip([o_cs, o_ce], 0, o_w)
    t_rs, t_re = np.clip([t_rs, t_re], 0, t_h)
    t_cs, t_ce = np.clip([t_cs, t_ce], 0, t_w)

    o_img = o_img[o_rs:o_re, o_cs:o_ce]
    t_img = t_img[t_rs:t_re, t_cs:t_ce]

    if center_in == "moving":
        return t_img, o_img, (t_rs, t_cs), (o_rs, o_cs)
    else:
        return o_img, t_img, (o_rs, o_cs), (t_rs, t_cs)


def align_around_center(
    aligner, center_yx, center_in="moving", crop_size=1000, downscale_factor=4
):
    ref, moving, ref_ori, moving_ori = crop_around_center(
        aligner, center_yx, center_in=center_in, crop_size=crop_size
    )
    if (0 in ref.shape) or (0 in moving.shape):
        return aligner.affine_matrix
    refined_mx = palom.register.feature_based_registration(
        palom.img_util.cv2_downscale_local_mean(ref, downscale_factor),
        palom.img_util.cv2_downscale_local_mean(moving, downscale_factor),
        plot_match_result=True,
        n_keypoints=10_000,
        plot_individual_result=False,
    )
    refined_mx = np.vstack([refined_mx, [0, 0, 1]])
    Affine = skimage.transform.AffineTransform
    # affine_matrix = (
    #     np.linalg.inv(Affine(translation=-1 * np.array(ref_ori)[::-1]).params)
    #     @ np.linalg.inv(Affine(scale=1 / downscale_factor).params)
    #     @ refined_mx
    #     @ Affine(scale=1 / downscale_factor).params
    #     @ Affine(translation=-1 * np.array(moving_ori)[::-1]).params
    # )
    affine_matrix = (
        Affine()
        + Affine(translation=-1 * np.array(moving_ori)[::-1])
        + Affine(scale=1 / downscale_factor)
        + Affine(matrix=refined_mx)
        + Affine(scale=1 / downscale_factor).inverse
        + Affine(translation=-1 * np.array(ref_ori)[::-1]).inverse
    ).params
    return affine_matrix


def view_coarse_align(reader1, reader2, affine_mx):
    v = napari.Viewer()
    kwargs = dict(visible=False, contrast_limits=(0, 50000), blending="additive")
    v.add_image([p[0] for p in reader1.pyramid], colormap="bop blue", **kwargs)
    v.add_image(
        [p[0] for p in reader2.pyramid],
        colormap="bop orange",
        affine=palom.img_util.to_napari_affine(affine_mx),
        **kwargs,
    )
    return v


def parse_roi_points(all_points):
    return np.array(re.findall(r"-?\d+\.?\d+", all_points), dtype=float).reshape(-1, 2)


cycif_path = r"V:\hits\lsp\collaborations\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\STIC_Batch2_2023\LSP15331\registration\LSP15331.ome.tif"
r1 = palom.reader.OmePyramidReader(cycif_path)

geomx_path = r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\p110_e4_OvarianSTIC_GeoMx\GeoMx_OME_tiff\LSP15332_Scan1.ome.tiff"
r2 = palom.reader.OmePyramidReader(geomx_path)

c21l = palom.align.get_aligner(r1, r2, thumbnail_level2=None)
c21l.coarse_register_affine(n_keypoints=10000)


ome = ome_types.from_tiff(geomx_path)
shapes = [rr.union[1] for rr in ome.rois]
polygons = filter(lambda x: isinstance(x, ome_types.model.polygon.Polygon), shapes)
polygons = [np.fliplr(parse_roi_points(pp.points)) for pp in polygons]
polygon_centers = [0.5 * (pp.min(axis=0) + pp.max(axis=0)) for pp in polygons]

v = view_coarse_align(r1, r2, c21l.affine_matrix)
v.add_shapes(
    polygons,
    shape_type="polygon",
    affine=palom.img_util.to_napari_affine(c21l.affine_matrix),
    face_color="#ffffff00",
    edge_color="salmon",
    edge_width=20,
)

polygon_affine_mxs = []
for pp, cc in zip(polygons, polygon_centers):
    mx = align_around_center(c21l, cc, crop_size=2000, downscale_factor=4)
    polygon_affine_mxs.append(mx)

t_polygons = []
for pp, mm in zip(polygons, polygon_affine_mxs):
    tform = skimage.transform.AffineTransform(matrix=mm)
    t_polygons.append(np.fliplr(tform(np.fliplr(pp))))

v.add_shapes(
    t_polygons,
    shape_type="polygon",
    face_color="#ffffff00",
    edge_color="lightgreen",
    edge_width=20,
)


# ---------------------------------------------------------------------------- #
#                    Use single affine from wsi to map ROIs                    #
# ---------------------------------------------------------------------------- #
def view_coarse_align(reader1, reader2, affine_mx):
    v = napari.Viewer()
    kwargs = dict(visible=False, contrast_limits=(0, 50000), blending="additive")
    v.add_image([p[0] for p in reader1.pyramid], colormap="bop blue", **kwargs)
    v.add_image(
        [p[0] for p in reader2.pyramid],
        colormap="bop orange",
        affine=palom.img_util.to_napari_affine(affine_mx),
        **kwargs,
    )
    return v


def parse_roi_points(all_points):
    return np.array(re.findall(r"-?\d+\.?\d+", all_points), dtype=float).reshape(-1, 2)


cycif_path = r"V:\hits\lsp\collaborations\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\STIC_Batch2_2023\LSP15331\registration\LSP15331.ome.tif"
r1 = palom.reader.OmePyramidReader(cycif_path)

geomx_path = r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\p110_e4_OvarianSTIC_GeoMx\GeoMx_OME_tiff\LSP15332_Scan1.ome.tiff"
r2 = palom.reader.OmePyramidReader(geomx_path)

c21l = palom.align.get_aligner(r1, r2, thumbnail_level2=None)
c21l.coarse_register_affine(n_keypoints=10000)


ome = ome_types.from_tiff(geomx_path)
shapes = [rr.union[1] for rr in ome.rois]
polygons = filter(lambda x: isinstance(x, ome_types.model.polygon.Polygon), shapes)
polygons = [np.fliplr(parse_roi_points(pp.points)) for pp in polygons]


v = view_coarse_align(r1, r2, c21l.affine_matrix)
v.add_shapes(
    polygons,
    shape_type="polygon",
    affine=palom.img_util.to_napari_affine(c21l.affine_matrix),
    face_color="#ffffff00",
    edge_color="salmon",
    edge_width=20,
)

tform = skimage.transform.AffineTransform(matrix=c21l.affine_matrix)
v.add_shapes([np.fliplr(tform(np.fliplr(pp))) for pp in polygons], shape_type="polygon")
# The result isn't satisfying, it will require manual adjustment afterward

# ---------------------------------------------------------------------------- #
#        Tested multi-object aligner, didn't seem to improve the result        #
# ---------------------------------------------------------------------------- #
moa = palom.align_multi_obj.MultiObjAligner(r1, r2)
moa.segment_objects(plot_segmentation=True, downscale_factor=2)


c21l = moa.make_aligner()
masked_t1 = np.full_like(c21l.ref_thumbnail, moa.fill_value_ref_thumbnail)
masked_t2 = np.full_like(c21l.moving_thumbnail, moa.fill_value_moving_thumbnail)

rs, re, cs, ce = moa.bbox_ref_thumbnail[1]
rsm, rem, csm, cem = palom.align_multi_obj.transform_bbox(
    moa.bbox_ref_thumbnail, moa.baseline_coarse_affine_matrix
)[1]

masked_t1[rs:re, cs:ce] = c21l.ref_thumbnail[rs:re, cs:ce]
masked_t2[rsm:rem, csm:cem] = c21l.moving_thumbnail[rsm:rem, csm:cem]

c21l.ref_thumbnail = masked_t1
c21l.moving_thumbnail = masked_t2
c21l.coarse_register_affine(n_keypoints=10000)

_c21l = palom.align.get_aligner(r1, r2, thumbnail_level2=None)
_c21l.coarse_register_affine(n_keypoints=10000)

v = napari.Viewer()
kwargs = dict(visible=False, contrast_limits=(0, 50000), blending="additive")
v.add_image([p[0] for p in r1.pyramid], colormap="bop blue", **kwargs)
v.add_image(
    [p[0] for p in r2.pyramid],
    colormap="bop orange",
    affine=palom.img_util.to_napari_affine(c21l.affine_matrix),
    **kwargs,
)

v.add_image(
    [p[0] for p in r2.pyramid],
    colormap="bop orange",
    affine=palom.img_util.to_napari_affine(_c21l.affine_matrix),
    **kwargs,
)
