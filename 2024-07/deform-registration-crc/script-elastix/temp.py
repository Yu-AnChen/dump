case_number = [1, 2, 3, 4, 6, 7, 8, 9, 10]

for nn in case_number:
    ref = tifffile.imread(
        f"/Users/yuanchen/projects/STalign/docs/img_data/test-elastix-img-pair/C{nn:02}-ref.tif"
    )
    moving = tifffile.imread(
        f"/Users/yuanchen/projects/STalign/docs/img_data/test-elastix-img-pair/C{nn:02}-moving.tif"
    )
    nth_condition = 0

    ss, ns = conditions[nth_condition]
    elastix_parameter = itk.ParameterObject.New()
    elastix_parameter.AddParameterMap(
        elastix_parameter.GetDefaultParameterMap("rigid", 4)
    )
    elastix_parameter.AddParameterMap(
        get_default_crc_params(sample_region_size=ss, sample_number_of_pixels=ns)
    )


def run_one_setting(ref_path, moving_path, setting):
    ref = tifffile.imread(ref_path)
    moving = tifffile.imread(moving_path)
    if setting is None:
        setting = {}
    elastix_parameter = itk.ParameterObject.New()
    elastix_parameter.AddParameterMap(
        elastix_parameter.GetDefaultParameterMap("rigid", 4)
    )
    elastix_parameter.AddParameterMap(get_default_crc_params(**setting))
    warpped_moving, transform_parameter = itk.elastix_registration_method(
        ref,
        moving,
        parameter_object=elastix_parameter,
        log_to_console=False,
    )
    return warpped_moving, transform_parameter, elastix_parameter
