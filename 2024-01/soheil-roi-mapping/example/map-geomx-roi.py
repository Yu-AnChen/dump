import copy
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import ome_types
import palom
import skimage.exposure
import skimage.transform
import tqdm


def align_around_center(
    aligner,
    center_yx,
    center_in="moving",
    crop_size=2000,
    downscale_factor=8,
    valid_cutoff=0.1,
    max_crop_size=10_000,
    plot_match=False,
):
    assert valid_cutoff > 0
    crop_size = int(crop_size)
    downscale_factor = int(downscale_factor)
    _center_yx = center_yx

    if crop_size > max_crop_size:
        return aligner.affine_matrix

    if center_in == "moving":
        center_yx = aligner.tform([center_yx[::-1]])[0][::-1]

    row_s, col_s = np.clip(
        np.floor(np.subtract(center_yx, crop_size / 2)).astype(int), 0, None
    )
    img1 = aligner.ref_img[row_s : row_s + crop_size, col_s : col_s + crop_size]
    img2 = palom.block_affine.block_affine(
        (row_s, col_s),
        block_shape=(crop_size, crop_size),
        src_img=aligner.moving_img,
        transformation=aligner.tform,
    )
    if 0 in [*img1.shape, *img2.shape]:
        return align_around_center(
            aligner,
            center_yx=_center_yx,
            center_in=center_in,
            crop_size=2 * crop_size,
            downscale_factor=2 * downscale_factor,
            plot_match=plot_match,
        )
    simg1 = palom.img_util.cv2_downscale_local_mean(img1, downscale_factor)
    simg2 = palom.img_util.cv2_downscale_local_mean(img2, downscale_factor)
    refined_mx = palom.register.feature_based_registration(
        simg1,
        simg2,
        plot_match_result=plot_match,
        n_keypoints=20_000,
        plot_individual_result=False,
    )
    if plot_match:
        plt.gca().set_title(
            f"Center ({center_in}): ({_center_yx[0]:.2f}, {_center_yx[1]:.2f})"
        )
    if np.linalg.norm(np.diag(refined_mx) - 1) > valid_cutoff:
        return align_around_center(
            aligner,
            center_yx=_center_yx,
            center_in=center_in,
            crop_size=2 * crop_size,
            downscale_factor=2 * downscale_factor,
            plot_match=plot_match,
        )
    refined_mx = np.vstack([refined_mx, [0, 0, 1]])
    Affine = skimage.transform.AffineTransform
    affine_matrix = (
        Affine()
        + Affine(matrix=aligner.affine_matrix)
        + Affine(translation=-1 * np.array([col_s, row_s]))
        + Affine(scale=1 / downscale_factor)
        + Affine(matrix=refined_mx)
        + Affine(scale=1 / downscale_factor).inverse
        + Affine(translation=-1 * np.array([col_s, row_s])).inverse
    ).params
    return affine_matrix


def view_coarse_align(reader1, reader2, affine_mx):
    import napari

    v = napari.Viewer()
    kwargs = dict(visible=False, contrast_limits=(0, 50000), blending="additive")
    v.add_image(
        [p[0] for p in reader1.pyramid], colormap="bop blue", name="cycif", **kwargs
    )
    v.add_image(
        [p[0] for p in reader2.pyramid],
        colormap="bop orange",
        affine=palom.img_util.to_napari_affine(affine_mx),
        name="geomx",
        **kwargs,
    )
    return v


def parse_roi_points(all_points):
    return np.array(re.findall(r"-?\d+\.?\d+", all_points), dtype=float).reshape(-1, 2)


def roi_center_yx(roi: ome_types.model.ROI):
    shapes = roi.union
    model = ome_types.model
    geometries = (
        model.Point,
        model.Polygon,
        model.Polyline,
        model.Rectangle,
        model.Ellipse,
        model.Line,
    )
    # Prioritize geometries
    is_geometry = [True if isinstance(ss, geometries) else False for ss in shapes]
    index = 0
    if True in is_geometry:
        index = is_geometry.index(True)

    shape = shapes[index]
    if hasattr(shape, "y") & hasattr(shape, "x"):
        h, w = 0, 0
        if hasattr(shape, "height"):
            h = shape.height
        if hasattr(shape, "width"):
            w = shape.width
        yx = shape.y + 0.5 * h, shape.x + 0.5 * w
    elif isinstance(shape, model.Line):
        yx = np.mean([(shape.y1, shape.y2), (shape.x1, shape.x2)], axis=0)
    else:
        points = np.fliplr(parse_roi_points(shape.points))
        yx = 0.5 * (points.min(axis=0) + points.max(axis=0))
    if shape.transform is not None:
        tt = shape.transform
        mx = [[tt.a00, tt.a01, tt.a02], [tt.a10, tt.a11, tt.a12], [0, 0, 1]]
        tform = skimage.transform.AffineTransform(matrix=np.array(mx))
        yx = np.fliplr(tform(np.fliplr([yx])))
    return np.array(yx).flatten()


def to_napari_shape(roi):
    # FIXME need to add transformation
    # FIXME support other types of ROI
    assert isinstance(roi, (ome_types.model.Polygon, ome_types.model.Ellipse))
    if isinstance(roi, ome_types.model.Polygon):
        return np.fliplr(parse_roi_points(roi.points)), "polygon"
    if isinstance(roi, ome_types.model.Ellipse):
        p1 = roi.y - roi.radius_y, roi.x - roi.radius_x
        p2 = roi.y - roi.radius_y, roi.x + roi.radius_x
        p3 = roi.y + roi.radius_y, roi.x + roi.radius_x
        p4 = roi.y + roi.radius_y, roi.x - roi.radius_x
        return [p1, p2, p3, p4], "ellipse"


def make_roi_ome_xml(ref_ome, affine_mxs, geometry_only=False):
    assert isinstance(affine_mxs, (list, tuple))
    assert len(affine_mxs) == len(ref_ome.rois)

    out_ome = ome_types.OME()

    out_ome.creator = "yu-anchen"
    out_ome.rois = copy.deepcopy(ref_ome.rois)
    out_ome.structured_annotations = copy.deepcopy(ref_ome.structured_annotations)

    ome_tforms = [
        ome_types.model.AffineTransform(
            **dict(zip(["a00", "a01", "a02", "a10", "a11", "a12"], mx.flatten()[:6]))
        )
        for mx in affine_mxs
    ]

    for roi, ot in zip(out_ome.rois, ome_tforms):
        for ss in roi.union:
            ss.transform = ot

    if not geometry_only:
        return out_ome

    texts = [
        " | ".join(list(filter(lambda x: x, [ss.text for ss in roi.union])))
        for roi in out_ome.rois
    ]
    geometries = (
        ome_types.model.Point,
        ome_types.model.Polygon,
        ome_types.model.Polyline,
        ome_types.model.Rectangle,
        ome_types.model.Ellipse,
        ome_types.model.Line,
    )
    for roi, text in zip(out_ome.rois, texts):
        if roi.name is not None:
            roi.name = f"{roi.name} || {text}"
        else:
            roi.name = text
        for ss in roi.union:
            ss.text = text
        roi.union = list(filter(lambda x: isinstance(x, geometries), roi.union))
    return out_ome


def save_all_figs(dpi=300, format="pdf", out_dir=None, prefix=None):
    figs = [plt.figure(i) for i in plt.get_fignums()]
    if prefix is not None:
        for f in figs:
            if f._suptitle:
                f.suptitle(f"{prefix} {f._suptitle.get_text()}")
            else:
                f.suptitle(prefix)
    names = [f._suptitle.get_text() if f._suptitle else "" for f in figs]
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for f, n, nm in zip(figs, plt.get_fignums(), names):
        f.savefig(out_dir / f"{n}-{nm}.{format}", dpi=dpi, bbox_inches="tight")
        plt.close(f)


def set_subplot_size(w, h, ax=None):
    """w, h: width, height in inches"""
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left  # noqa: E741
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def get_reader(path):
    path = pathlib.Path(path)
    if path.suffix in [".svs", ".ndpi"]:
        return palom.reader.SvsReader
    else:
        return palom.reader.OmePyramidReader


def run_pair(
    cycif_path: str | pathlib.Path,
    geomx_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    geomx_roi_path: str | pathlib.Path | None = None,
    crop_size: int = 2000,
    downscale_factor: int = 5,
    refine: bool = False,
    viz_napari: bool = False,
    num_processes: int = 1,
):
    from loguru import logger

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "qc").mkdir(exist_ok=True)

    (out_dir / "log").mkdir(exist_ok=True)
    logger.remove()
    logger.add(out_dir / "log" / f"{pathlib.Path(cycif_path).stem}.log", level="INFO")

    r1 = get_reader(cycif_path)(cycif_path)
    r2 = get_reader(geomx_path)(geomx_path)

    p1 = r1.path
    p2 = r2.path

    try:
        ome = ome_types.from_tiff(p2, validate=False)
    except ValueError:
        ome = ome_types.OME()
    if (ome.rois is None) or (len(ome.rois) == 0):
        print(
            f"No ROI detected in {p2}, which is expected to be an OME-TIFF exported"
            f" from the NanoString software"
        )
        if geomx_roi_path is None:
            print("`geomx_roi_path` is not specified")
            return
        ome = ome_types.from_xml(geomx_roi_path, validate=False)
        if (ome.rois is None) or (len(ome.rois) == 0):
            print(f"No ROI detected in {geomx_roi_path}, either")
            return

    c21l = palom.align.get_aligner(r1, r2, thumbnail_level2=None)
    c21l.coarse_register_affine(n_keypoints=10000)

    fig, ax = plt.gcf(), plt.gca()
    fig.suptitle(f"{p2.name} (coarse alignment)", fontsize=8)
    ax.set_title(f"{p1.name} - {p2.name}", fontsize=6)
    im_h, im_w = ax.images[0].get_array().shape
    set_subplot_size(im_w / 288, im_h / 288, ax=ax)
    ax.set_anchor("N")
    # use 0.5 inch on the top for figure title
    fig.subplots_adjust(top=1 - 0.5 / fig.get_size_inches()[1])
    save_all_figs(out_dir=out_dir / "qc", format="jpg", dpi=144)
    plt.close("all")

    centers = [roi_center_yx(rr) for rr in ome.rois]

    affine_mxs = [c21l.affine_matrix] * len(ome.rois)

    if refine:
        import joblib  # type: ignore

        num_processes = np.clip(
            num_processes, 1, np.ceil(joblib.cpu_count() / 4).astype("int")
        )
        num_processes = int(num_processes)
        _mx = c21l.coarse_affine_matrix

        def _wrap(*args, **kwargs):
            from loguru import logger

            logger.remove()
            _r1 = get_reader(cycif_path)(cycif_path)
            _r2 = get_reader(geomx_path)(geomx_path)
            _aligner = palom.align.get_aligner(_r1, _r2, thumbnail_level2=None)
            _aligner.coarse_affine_matrix = _mx
            mx = align_around_center(_aligner, *args, **kwargs)
            return mx

        if num_processes > 1:
            # FIXME make aligner pickle-able for process based parallelization
            affine_mxs = joblib.Parallel(verbose=5, n_jobs=num_processes)(
                joblib.delayed(_wrap)(
                    cc,
                    crop_size=crop_size,
                    downscale_factor=downscale_factor,
                    plot_match=False,
                )
                for cc in centers[:]
            )
        else:
            affine_mxs = [
                _wrap(
                    cc,
                    crop_size=crop_size,
                    downscale_factor=downscale_factor,
                    plot_match=False,
                )
                for cc in tqdm.tqdm(centers)
            ]

    out_ome = make_roi_ome_xml(ome, affine_mxs, geometry_only=True)
    method = "refine" if refine else "coarse"
    out_name = (
        f"{p1.name.split('.')[0]}-rois-from-{p2.name.split('.')[0]}-{method}.ome.xml"
    )
    with open(out_dir / out_name, "w") as f:
        f.write(out_ome.to_xml())

    if viz_napari:
        ome_shapes = [rr.union[1] for rr in ome.rois]
        shapes = [to_napari_shape(ss) for ss in ome_shapes]
        v = view_coarse_align(r1, r2, c21l.affine_matrix)
        for ss, mm in zip(shapes, affine_mxs):
            v.add_shapes(
                [ss[0]],
                shape_type=ss[1],
                affine=palom.img_util.to_napari_affine(mm),
                face_color="#ffffff00",
                edge_color="salmon",
                edge_width=30,
                name="roi-to-cycif",
            )
        v.add_shapes(
            [ss[0] for ss in shapes],
            shape_type=[ss[1] for ss in shapes],
            affine=palom.img_util.to_napari_affine(c21l.affine_matrix),
            face_color="#ffffff00",
            edge_color="lightgreen",
            edge_width=30,
            name="roi-geomx",
        )
    return


def run_batch(csv_path, print_args=True, dryrun=False, **kwargs):
    import csv
    import inspect
    import types
    import pprint

    if print_args:
        _args = [str(vv) for vv in inspect.signature(run_pair).parameters.values()]
        print(f"\nFunction args\n{pprint.pformat(_args, indent=4)}\n")
    _arg_types = inspect.get_annotations(run_pair)
    arg_types = {}
    for k, v in _arg_types.items():
        if isinstance(v, types.UnionType):
            v = v.__args__[0]
        arg_types[k] = v

    with open(csv_path) as f:
        files = [
            {kk: arg_types[kk](vv) for kk, vv in rr.items() if kk in arg_types}
            for rr in csv.DictReader(f)
        ]

    if dryrun:
        for ff in files:
            pprint.pprint({**ff, **kwargs})
            print()
        return

    for ff in files:
        run_pair(**{**ff, **kwargs})


if __name__ == "__main__":
    import fire
    import sys

    fire.Fire({"run-pair": run_pair, "run-batch": run_batch})

    if ("--viz_napari" in sys.argv) or ("-v" in sys.argv):
        import napari

        napari.run()

    """
    NAME
        map-geomx-roi.py run-pair

    SYNOPSIS
        map-geomx-roi.py run-pair CYCIF_PATH GEOMX_PATH OUT_DIR <flags>

    POSITIONAL ARGUMENTS
        CYCIF_PATH
            Type: str | pathlib.Path
        GEOMX_PATH
            Type: str | pathlib.Path
        OUT_DIR
            Type: str | pathlib.Path

    FLAGS
        -g, --geomx_roi_path=GEOMX_ROI_PATH
            Type: Optional[str | pathlib...
            Default: None
        -c, --crop_size=CROP_SIZE
            Type: int
            Default: 2000
        -d, --downscale_factor=DOWNSCALE_FACTOR
            Type: int
            Default: 5
        -r, --refine=REFINE
            Type: bool
            Default: False
        -v, --viz_napari=VIZ_NAPARI
            Type: bool
            Default: False
        -n, --num_processes=NUM_PROCESSES
            Type: int
            Default: 1

    
    NAME
        map-geomx-roi.py run-batch

    SYNOPSIS
        map-geomx-roi.py run-batch CSV_PATH <flags>

    POSITIONAL ARGUMENTS
        CSV_PATH

    FLAGS
        -p, --print_args=PRINT_ARGS
            Default: True
        -d, --dryrun=DRYRUN
            Default: False
        Additional flags are accepted.
    """
