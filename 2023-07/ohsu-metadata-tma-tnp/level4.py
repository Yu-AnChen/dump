import pandas as pd


# ---------------------------------------------------------------------------- #
#                                    phase 3                                   #
# ---------------------------------------------------------------------------- #

l3 = pd.read_excel('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/HTAN Imaging Level 3 Segmentation Phase 3 HMS-YC.xlsx')
l4 = pd.read_excel('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/HTAN Imaging Level 4 Phase 3.xlsx')

id_map = pd.DataFrame(l3['HTAN Data File ID']).set_index(
    l3['Filename'].apply(lambda x: x.split('/')[1].split('.')[0])
)

mask = l4['Filename'].apply(lambda x: 'LSP' in x)

l4.loc[mask, [
    'File Format',
    'Software and Version',
    'Imaging Object Class',
    'Imaging Summary Statistic'
]] = [
    'csv',
    'labsyspharm/quantification v1.5.4',
    'whole cell',
    'mean intensity'
]

count_map = pd.read_csv('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/table-object-count-phase-3.csv', index_col='Filename')

l4.loc[mask, ['Number of Objects', 'Number of Features']] = count_map.loc[l4.loc[mask]['Filename'].apply(lambda x: x.split('/')[1])].values

l4.loc[mask, 'HTAN Parent Data File ID'] = id_map.loc[
    l4.loc[mask]['Filename'].apply(lambda x: x.split('/')[1].split('_')[1].replace('.csv', ''))
].values


l4.to_excel('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/HTAN Imaging Level 4 Phase 3-YC.xlsx', index=False)


# ---------------------------------------------------------------------------- #
#                                    phase 1                                   #
# ---------------------------------------------------------------------------- #
import pandas as pd

l3 = pd.read_excel('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/HTAN Imaging Level 3 Segmentation Phase 1 HMS.xlsx')
l4 = pd.read_excel('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/HTAN Imaging Level 4 Phase 1.xlsx')

l3 = l3[l3['Filename'].apply(lambda x: 'cell' in x)]
fns = l3['Filename'].apply(
    lambda x: f"{x.split('/')[1][:13]}-{x.split('/')[1][14:17].replace('_', '').replace('-', '')}"
)

id_map = pd.DataFrame(l3['HTAN Data File ID']).set_index(
    fns
)

mask = l4['Filename'].apply(lambda x: 'OHSU_TMA1' in x)

l4.loc[mask, [
    'File Format',
    'Software and Version',
    'Imaging Object Class',
    'Imaging Summary Statistic'
]] = [
    'csv',
    'labsyspharm/quantification v1.5.4',
    'whole cell',
    'mean intensity'
]


l4.loc[mask, [
    'File Format',
    'Software and Version',
    'Imaging Object Class',
    'Imaging Summary Statistic'
]] = [
    'csv',
    'labsyspharm/quantification v1.5.4',
    'whole cell',
    'mean intensity'
]

count_map = pd.read_csv('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/table-object-count-phase-1.csv', index_col='Filename')

l4.loc[mask, ['Number of Objects', 'Number of Features']] = count_map.loc[l4.loc[mask]['Filename'].apply(lambda x: x.split('/')[1])].values

l4.loc[mask, 'HTAN Parent Data File ID'] = id_map.loc[
    l4.loc[mask]['Filename'].apply(lambda x: '_'.join(x.split('/')[1].split('_')[3:]).replace('-cellRingMask.csv', ''))
].values


l4.to_excel('/Users/yuanchen/Downloads/ohsu-metadata-tma-tnp/HTAN Imaging Level 4 Phase 1-YC.xlsx', index=False)


"""
In [6]: l4.columns
Out[6]: 
Index(['Component', 'Filename', 'File Format', 'HTAN Parent Data File ID',
       'HTAN Parent Channel Metadata ID', 'HTAN Data File ID',
       'Parameter file', 'Software and Version', 'Commit SHA',
       'Number of Objects', 'Number of Features', 'Imaging Object Class',
       'Imaging Summary Statistic', 'Imaging Object Class Description', 'eTag',
       'entityId'],
      dtype='object')
"""