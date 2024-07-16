import itk
import napari
import tifffile
import numpy as np

ref = tifffile.imread(
    "/Users/yuanchen/projects/STalign/docs/img_data/test-elastix-img-pair/C02-ref.tif"
)
moving = tifffile.imread(
    "/Users/yuanchen/projects/STalign/docs/img_data/test-elastix-img-pair/C02-moving.tif"
)
registered_image, params = itk.elastix_registration_method(ref, moving)


v = napari.Viewer()
v.add_image(ref, colormap="bop blue")
v.add_image(moving, blending="additive", colormap="bop purple")
# v.add_image(registered_image, blending="additive", colormap="bop orange")


# ---------------------------------------------------------------------------- #
#                           read parameters from file                          #
# ---------------------------------------------------------------------------- #
# parameter_object = itk.ParameterObject.New()
# parameter_object.AddParameterFile(
#     "/Users/yuanchen/projects/ITKElastix/examples/exampleoutput/parameters-crc.0.txt"
# )


# ---------------------------------------------------------------------------- #
#                            use and modify defaults                           #
# ---------------------------------------------------------------------------- #
parameter_object = itk.ParameterObject.New()

# affine
default_affine_parameter_map = parameter_object.GetDefaultParameterMap("rigid", 4)
default_affine_parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
parameter_object.AddParameterMap(default_affine_parameter_map)

# deformation
default_bspline_parameter_map = parameter_object.GetDefaultParameterMap("bspline", 4)
default_bspline_parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
del default_bspline_parameter_map["FinalGridSpacingInPhysicalUnits"]
default_bspline_parameter_map["FinalGridSpacingInVoxels"] = ["96.0", "96.0"]
default_bspline_parameter_map["GridSpacingSchedule"] = ["8.0", "4.0", "2.0", "1.0"]
default_bspline_parameter_map["MaximumNumberOfIterations"] = ["1000"]
default_bspline_parameter_map["ASGDParameterEstimationMethod"] = [
    "DisplacementDistribution"
]
default_bspline_parameter_map["Metric1Weight"] = ["100.0"]
default_bspline_parameter_map["NumberOfSamplesForExactGradient"] = [f"{10_000}"]

parameter_object.AddParameterMap(default_bspline_parameter_map)


# Call registration function
result_image, result_transform_parameters = itk.elastix_registration_method(
    ref,
    moving,
    parameter_object=parameter_object,
    log_to_console=False,
    # number_of_threads=4,
)

v.add_image(result_image, blending="additive", colormap="bop orange")


# deformation field in the shape of (H, W, [X, Y])
deformation_field = itk.transformix_deformation_field(
    itk.GetImageFromArray(np.ascontiguousarray(moving)), result_transform_parameters
)


# use deformation field to warp moving image (with scikit-image)
import skimage.transform  # noqa: E402

dx, dy = np.moveaxis(deformation_field, 2, 0)
my, mx = np.mgrid[: ref.shape[0], : ref.shape[1]]

warpped_moving_image = skimage.transform.warp(
    moving,
    np.array([my + dy, mx + dx]),
    preserve_range=True,
)


# use deformation field to warp moving image (with opencv)
# faster but may be less accurate: generate ring-lik artifact after subtracting
# the elastix result

import cv2  # noqa: E402

dx, dy = np.moveaxis(deformation_field, 2, 0)
my, mx = np.mgrid[: ref.shape[0], : ref.shape[1]].astype("float32")
warpped_moving_image_cv2 = cv2.remap(
    moving.astype("float"),
    mx + dx,
    my + dy,
    cv2.INTER_LINEAR,
)


# ---------------------------------------------------------------------------- #
#                 inverse deformation field for points mapping                 #
# ---------------------------------------------------------------------------- #

# use ITK default method (valis uses this too)
import SimpleITK as sitk  # noqa: E402

inverted_iterative = sitk.IterativeInverseDisplacementField(
    # setting `isVector=True` so that the last dimension is used as index
    sitk.GetImageFromArray(deformation_field, isVector=True),
    numberOfIterations=10,
)

idxi, idyi = np.moveaxis(sitk.GetArrayFromImage(inverted_iterative), 2, 0)
_ = skimage.transform.warp(ref, np.array([my + idyi, mx + idxi]), preserve_range=True)


# use itk-fixedpointinversedisplacementfield (a newer, more efficient method)
# this seems to be much faster and yeild quality results
# https://github.com/thewtex/ITKFixedPointInverseDisplacementField/actions/runs/9575803730
# https://github.com/InsightSoftwareConsortium/ITKFixedPointInverseDisplacementField

# python -m pip install /Users/yuanchen/projects/ITKElastix/examples/itk-fixedpointinversedisplacementfield-wheel/itk_fixedpointinversedisplacementfield-1.0.0-cp310-cp310-macosx_11_0_arm64.whl

inverted_fixed_point = itk.FixedPointInverseDisplacementFieldImageFilter(
    deformation_field, NumberOfIterations=10, Size=deformation_field.shape[:2][::-1]
)

idxf, idyf = np.moveaxis(inverted_fixed_point, 2, 0)
_ = skimage.transform.warp(ref, np.array([my + idyf, mx + idxf]), preserve_range=True)


# ---------------------------------------------------------------------------- #
#                              Point / ROI mappint                             #
# ---------------------------------------------------------------------------- #

# manually generate points for demo
moving_points = np.array(
    [
        [434.37502459, 141.36280092],
        [427.26091383, 1596.76755366],
        [1127.52280493, 483.41327418],
    ]
)


# use scipy to map point coordinates
import scipy.ndimage as ndi  # noqa: E402

mapped_moving_points = np.vstack(
    [
        ndi.map_coordinates(my + idyf, moving_points.T),
        ndi.map_coordinates(mx + idxf, moving_points.T),
    ]
).T


# use opencv to map point coordinates
# more than 10x faster than scipy, but could be less accurate?
np.vstack([
    np.hstack(
        [
            cv2.getRectSubPix(my + idyf, (1, 1), pp[::-1]),
            cv2.getRectSubPix(mx + idxf, (1, 1), pp[::-1]),
        ]
    )
    for pp in moving_points
])
