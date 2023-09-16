```bash
conda create -n ashlar-lt python=3.10 pyjnius pyqt -c conda-forge
conda activate ashlar-lt
python -m pip install ashlar napari[all] palom
python -m pip install "napari-wsi-reader @ git+https://github.com/Yu-AnChen/napari-wsi-reader@a12f31d57d5f41c911e55611515055b31444b726"
```
