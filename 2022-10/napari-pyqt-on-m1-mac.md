```bash
# get qt from homebrew
brew uninstall qt@5

# install pyqt with conda
conda create -n qt -c conda-forge python pyqt psutil
conda activate qt

# instead of `python -m pip install napari[all]`
python -m pip install napari
```
