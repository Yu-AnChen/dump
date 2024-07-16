import itk
import napari
import tifffile
import numpy as np

fixed_image = tifffile.imread(
    "/Users/yuanchen/projects/STalign/docs/img_data/C02-ref.tif"
)
moving_image = tifffile.imread(
    "/Users/yuanchen/projects/STalign/docs/img_data/C02-moving.tif"
)


parameter_object = itk.ParameterObject.New()
default_rigid_parameter_map = parameter_object.GetDefaultParameterMap("rigid")
parameter_object.AddParameterMap(default_rigid_parameter_map)

# Call registration function
result_image, result_transform_parameters = itk.elastix_registration_method(
    fixed_image, moving_image, parameter_object=parameter_object, log_to_console=False
)

v = napari.Viewer()

v.add_image(fixed_image)
v.add_image(moving_image)
v.add_image(result_image)
v.add_image(
    itk.transformix_deformation_field(
        itk.GetImageFromArray(moving_image), result_transform_parameters
    ),
    channel_axis=2,
    contrast_limits=(-30, 30),
    colormap="coolwarm",
)

deformation_field = itk.transformix_deformation_field(
    itk.GetImageFromArray(moving_image), result_transform_parameters
)


import skimage.transform

warpped_moving_image = skimage.transform.warp(
    moving_image,
    np.mgrid[: fixed_image.shape[0], : fixed_image.shape[1]]
    + np.moveaxis(deformation_field, 2, 0)[::-1],
)
