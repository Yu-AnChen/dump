import pandas as pd


# ---------------------------------------------------------------------------- #
#                                  for phase 1                                 #
# ---------------------------------------------------------------------------- #
l2 = pd.read_excel('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/HTAN Imaging Level 2 Phase 1.xlsx')
l3 = pd.read_excel('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/HTAN Imaging Level 3 Segmentation Phase 1 HMS.xlsx')

l2_id = pd.DataFrame(
    l2['HTAN Data File ID'].values,
    index=l2['Filename'].apply(
        lambda x: x.split('/')[1].replace('.ome.tif', '')
    ),
    columns=['HTAN Data File ID']
)


l3['File Format'] = l3['Filename'].apply(
    lambda x: 'OME-TIFF' if x.endswith('.ome.tif') else 'tif'
)
l3['Imaging Segmentation Data Type'] = 'Mask'
l3['Imaging Object Class'] = l3['Filename'].apply(
    lambda x: 'whole cell' if 'cell' in x.lower() else 'nucleus'
)

fns = l3['Filename'].apply(
    lambda x: f"{x.split('/')[1][:13]}-{x.split('/')[1][14:17].replace('_', '').replace('-', '')}"
)

for nn in fns:
    assert(nn in l2_id.index)

l3['HTAN Parent Data File ID'] = l2_id.loc[fns].values
l3['Software and Version'] = 'S3segmenter v1.5.4'

count_map = pd.read_csv('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/mask-object-count-phase-1.csv', index_col='Filename')

l3['Number of Objects'] = count_map.loc[l3['Filename'].apply(lambda x: x.split('/')[1])].values

l3.to_excel('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/HTAN Imaging Level 3 Segmentation Phase 1 HMS-YC.xlsx')

# ---------------------------------------------------------------------------- #
#                                  for phase 3                                 #
# ---------------------------------------------------------------------------- #
import pandas as pd


l2 = pd.read_excel('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/HTAN Imaging Level 2 Phase 3.xlsx')
l3 = pd.read_excel('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/HTAN Imaging Level 3 Segmentation Phase 3 HMS.xlsx')

l2_id = pd.DataFrame(
    l2['HTAN Data File ID'].values,
    index=l2['Filename'].apply(
        lambda x: x.split('/')[1].replace('.ome.tif', '')
    ),
    columns=['HTAN Data File ID']
)

l3['File Format'] = 'OME-TIFF'
l3['Imaging Segmentation Data Type'] = 'Mask'
l3['Imaging Object Class'] = 'whole cell'

fns = l3['Filename'].apply(
    lambda x: '-'.join(x.split('/')[1].split('-')[:2])
)
for nn in fns:
    assert(nn in l2_id.index)

l3['HTAN Parent Data File ID'] = l2_id.loc[fns].values
l3['Software and Version'] = 'S3segmenter'
l3['Commit SHA'] = 'e9aef02c5adf6e87b065b7aed61ce6071fe1224d'


count_map = pd.read_csv('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/mask-object-count-phase-3.csv', index_col='Filename')

l3['Number of Objects'] = count_map.loc[l3['Filename'].apply(lambda x: x.split('/')[1])].values

l3.to_excel('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/HTAN Imaging Level 3 Segmentation Phase 3 HMS-YC.xlsx')
"""
In [9]: l3.columns
Out[9]: 
Index(['Component', 'Filename', 'File Format', 'HTAN Parent Data File ID',
       'HTAN Data File ID', 'Imaging Segmentation Data Type', 'Parameter file',
       'Software and Version', 'Commit SHA', 'Imaging Object Class',
       'Number of Objects', 'Imaging Object Class Description', 'eTag',
       'entityId'],
      dtype='object')

"""