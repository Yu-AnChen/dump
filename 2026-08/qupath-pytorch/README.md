# QuPath + PyTorch (CUDA) on Windows, via pixi

```
pixi run check-cuda  # CUDA visible to the pixi env?
pixi run check-gpu   # CUDA visible to QuPath? (pulls QuPath + engine, GBs, once)
pixi run launch      # open QuPath
```

`launch-qupath.bat` = `pixi run launch`, for double-clicking.
Extra args go to QuPath: `pixi run launch script my.groovy`.

QuPath fetches the PyTorch natives itself into `%USERPROFILE%\.djl.ai`, but
can't supply CUDA/cuDNN. The torch wheels do, so `launch` puts
`Lib\site-packages\torch\lib` on `PATH` and sets `PYTORCH_FLAVOR=cu128` - same
as QuPath's *Extensions > Deep Java Library > Create launch script*. `check-gpu`
does the engine download that *Manage DJL Engines* otherwise makes you click
through. The DJL extension itself ships inside QuPath 0.7.0.

Version chain is fixed by QuPath, don't mix and match:

| QuPath | DJL | PyTorch | CUDA |
| --- | --- | --- | --- |
| 0.7.x | 0.36.0 | 2.7.1 | 12.8 |

`check-gpu` prints `capabilities: [CUDA, CUDNN, ...]` and exits 0.
No CUDA/CUDNN, or a failed assert? Delete `%USERPROFILE%\.djl.ai\pytorch` and re-run.
