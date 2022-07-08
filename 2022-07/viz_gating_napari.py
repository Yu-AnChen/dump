import tabbi
import pandas as pd
import tifffile
import skimage.segmentation

# load segmentation mask
mask = tifffile.imread(r"Z:\RareCyte-S3\YC-analysis\P37_CRCstudy_Round1\P37_S36-CRC08\segmentation\P37_S36_A24_C59kX_E15@20220108_012058_082564\nucleiRingMask.tif")
# load single-cell table and use "CellID" column as index column
df = pd.read_csv(r"Z:\RareCyte-S3\YC-analysis\P37_CRCstudy_Round1\P37_S36-CRC08\quantification\raw\P37_S36_A24_C59kX_E15@20220108_012058_082564_cellRingMask_intensity.csv", index_col='CellID')

# load the orion image into napari viewer and use the marker names from the
# single-cell table as channel name
viewer = tabbi.napari_pyramid_tool.lazy_view(r"Z:\RareCyte-S3\P37_CRCstudy_Round1\P37_S36_A24_C59kX_E15@20220108_012058_082564.ome.tiff", channel_names=df.columns[:-2])

# find gate for CD45, use 3 components. the value found is GATE_CD45
tabbi.gmm.plot_hist_gmm(df, ['CD45', 'CD4'], n_components=2)
GATE_CD45 = 667
# compute and add plain mask outline to viewer
bounds = skimage.segmentation.find_boundaries(mask, mode='inner')
# this generate pyramid in-place so the full res can be displayed
viewer.add_image(
    [bounds[::8**i, ::8**i] for i in range(5)],
    blending='additive'
)



# === option 1: plot controids colored by positivity ===
viewer.add_points(
    data=df[['Y_centroid', 'X_centroid']].values, 
    name=f"pCD45-{GATE_CD45}",
    properties={
        'positive': (df.CD45.values > GATE_CD45).astype(float),
    },
    face_color='positive',
    face_colormap='viridis',
    size=10,
    opacity=1,
    visible=True,
)

# === option 2: uses more RAM ===
# helper function for re-coloring the mask
def mapping_indexer(
    df, value_for_missing_key=0, cat_column_name='Cat1'
):
    indexer = np.ones(df.index.max() + 1) * value_for_missing_key
    indexer[df.index.values] = df[cat_column_name].values
    return indexer

indexer = mapping_indexer(
    df.assign(pCD45=(df.CD45 > GATE_CD45) + 1),
    cat_column_name='pCD45'
)
mask_pcd45 = indexer.astype(np.uint8)[bounds*mask]
viewer.add_image(
    [mask_pcd45[::8**i, ::8**i] for i in range(5)],
    name=f"mask-pCD45-{GATE_CD45}",
    blending='additive',
    contrast_limits=(0, 2),
    colormap='magma'
)