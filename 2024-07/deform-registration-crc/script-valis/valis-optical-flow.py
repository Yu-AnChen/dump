import valis.non_rigid_registrars
import napari
import tifffile


v = napari.Viewer()


ref = tifffile.imread("/Users/yuanchen/projects/STalign/docs/img_data/C02-ref.tif")
moving = tifffile.imread(
    "/Users/yuanchen/projects/STalign/docs/img_data/C02-moving.tif"
)

warppeds = []
smoothing = [
    None,
    "gauss",
    # "inpaint",
    # "regularize",
]

for ss in smoothing:
    print("running", f"{ss}")
    warpped, *_ = valis.non_rigid_registrars.OpticalFlowWarper(
        smoothing_method=ss
    ).register(moving, ref)
    warppeds.append(warpped)


v.add_image(ref, colormap="bop blue")
for ii in warppeds:
    v.add_image(ii, blending="additive", colormap="bop orange")
