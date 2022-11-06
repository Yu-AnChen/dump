conda update conda
conda create -n ashlar python=3.10 pyjnius -c conda-forge
conda activate ashlar

conda install napari[all] -c conda-forge
conda install scikit-learn matplotlib seaborn -c conda-forge

python -m pip install ashlar

conda install openslide -c sdvillal

python -m pip install palom