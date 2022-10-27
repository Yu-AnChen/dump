```bash
# get qt from homebrew
brew uninstall qt@5

# install pyqt with conda
conda create -n qt -c conda-forge python pyqt psutil
conda activate qt

# install via conda
conda install napari -c conda-forge openjdk

# if install from pypi
# instead of `python -m pip install napari[all]`
python -m pip install napari
```

```bash
conda create -n ashlar -c conda-forge python pyqt psutil
conda activate ashlar
conda install napari openslide openjdk -c conda-forge
python -m pip install ashlar palom ome_types imreg_dft tqdm
```
