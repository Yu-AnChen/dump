import pathlib
import subprocess

import joblib


# ---------------------------------------------------------------------------- #
#                                    phase 3                                   #
# ---------------------------------------------------------------------------- #
curr = pathlib.Path(r'X:\cycif-production\111-TMA-TNP\199-OMS_Atlas_TMA-2021NOV')
files = sorted(curr.glob('LSP*/level4/*.csv'))

names = [
    ff.name
    for ff in files
]

cmd_feature = r"""
head {} -n 1 | sed "s/[^,]//g" | wc -c
"""

cmd_counts = r"""
wc -l {}
"""

def count_feature(path):
    return int(subprocess.check_output(
        cmd_feature.format(path).strip(), shell=True
    ).decode())

def count_object(path):
    return int(subprocess.check_output(
        cmd_counts.format(path).strip(), shell=True
    ).decode().split(' ')[0]) - 1


feature_counts = joblib.Parallel(n_jobs=-1, verbose=10)(
    joblib.delayed(count_feature)(pp) for pp in files
)

obj_counts = joblib.Parallel(n_jobs=-1, verbose=10)(
    joblib.delayed(count_object)(pp) for pp in files
)

import pandas as pd

pd.DataFrame({'Filename': names, 'Number of Objects': obj_counts, 'Number of Features': feature_counts}).to_csv('table-object-count-phase-3.csv', index=False)


# ---------------------------------------------------------------------------- #
#                                    phase 1                                   #
# ---------------------------------------------------------------------------- #
import pathlib
import subprocess

import joblib

curr = pathlib.Path(r'X:\cycif-production\111-TMA-TNP\168-OHSU_TMATNP-2020SEP\transfer')
files = sorted(curr.glob('OHSU*/level4/*.csv'))

names = [
    ff.name
    for ff in files
]

cmd_feature = r"""
head {} -n 1 | sed "s/[^,]//g" | wc -c
"""

cmd_counts = r"""
wc -l {}
"""

def count_feature(path):
    return int(subprocess.check_output(
        cmd_feature.format(path).strip(), shell=True
    ).decode())

def count_object(path):
    return int(subprocess.check_output(
        cmd_counts.format(path).strip(), shell=True
    ).decode().split(' ')[0]) - 1


feature_counts = joblib.Parallel(n_jobs=-1, verbose=10)(
    joblib.delayed(count_feature)(pp) for pp in files
)

obj_counts = joblib.Parallel(n_jobs=-1, verbose=10)(
    joblib.delayed(count_object)(pp) for pp in files
)

import pandas as pd

pd.DataFrame({'Filename': names, 'Number of Objects': obj_counts, 'Number of Features': feature_counts}).to_csv('table-object-count-phase-1.csv', index=False)
