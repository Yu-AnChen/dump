## Installation

1. Follow the installation instruction here - https://github.com/labsyspharm/palom#installation

1. Install additional packages in the conda env
    ```
    conda activate palom
    python -m pip install fire
    ```

## Usage

1. Use the script like so: crop 5000-pixel square from the coordinate (9000, 4000)

    ```
    python crop-to-npz.py \\research.files.med.harvard.edu\ImStor\sorger\data\RareCyte\JL503_JERRY\TNP_2020\Addtional_CRC_Ometiff\CRC12.ome.tif \\research.files.med.harvard.edu\ImStor\sorger\data\RareCyte\JL503_JERRY\TNP_2020\Addtional_CRC_MCMICRO\mcmicro\CRC12_n\segmentation\unmicst-TNPCRC_11\cellRingMask.tif \\research.files.med.harvard.edu\ImStor\sorger\data\computation\Yu-An\YC-20231122-Alex_recyze\output\CRC12 9000 4000 5000
    ```

1. The above command will output two files under `\\research.files.med.harvard.edu\ImStor\sorger\data\computation\Yu-An\YC-20231122-Alex_recyze\output\CRC12`

    ```
    image-x_9000-y_4000.npz
    mask-x_9000-y_4000.npz
    ```

### Script usage reference

```
NAME
    crop-to-npz.py

SYNOPSIS
    crop-to-npz.py IMG_PATH MASK_PATH OUT_DIR UL_X UL_Y SIZE

POSITIONAL ARGUMENTS
    IMG_PATH: path to cycif image
    MASK_PATH: path to segmentation mask
    OUT_DIR: output directory for the .npz files
    UL_X: upper-left x position for cropping
    UL_Y: upper-left y position for cropping
    SIZE: the size of the resulting square

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS
```