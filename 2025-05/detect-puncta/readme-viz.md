# Visualize spot detection single-cell results with napari

This guide outlines the setup and execution of a Python script for visualizing
FISH spot detection images and results using Napari.

1. Create a local project env directory, e.g.
   `C:\Users\<your_username>\Documents\Projects\viz-env`

2. [Download the viz
   script](https://github.com/Yu-AnChen/dump/blob/main/2025-05/detect-puncta/visual_check_cli.py)
   from github and move it to the above folder

3. Copy these two files to the above directory
    - `\\research.files.med.harvard.edu\HITS\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\YC-registered-HE-elastix\YC-viz\workspace\dist\environment.tar`
    - `\\research.files.med.harvard.edu\HITS\lsp-data\cycif-production\17-tuberculosis-aldridge\p17e21_3D_HE\YC-registered-HE-elastix\YC-viz\workspace\dist\pixi-x86_64-pc-windows-msvc.exe`

4. Run the following command in a command prompt to unpack the environment.
   NOTE: replace `<your_username>` in the following.

   ```
   cd C:\Users\<your_username>\Documents\Projects\viz-env
   pixi-x86_64-pc-windows-msvc.exe exec pixi-unpack environment.tar
   ```

   This will create an `env` folder and `activate.bat`.

5. Activate the environment: Run the activation script in the command prompt.
   NOTE: next time when you open a new command prompt, you'll run the same
   command first, and then, run the visualization script.

   ```
   C:\Users\<your_username>\Documents\Projects\viz-env\activate.bat
   ```

6. Run 1visual_check_cli.py`. It will launch napari and display the spot
   detection results. NOTE: I mounted
   `\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION-FISH-72525_Tanjina`
   to `T` drive.

    ```
    cd C:\Users\<your_username>\Documents\Projects\viz-env

    python visual_check_cli.py --img-path "T:\Orion_FISH_test1\Tanjina_ORION_FISH_test_HGSOC\LSP14728\fish\qc\LSP14728_001_A107_P54N_HMS_TK-spots-channel_22_24_27_28_29-qc.ome.tif" --mask-path "T:\Orion_FISH_test1\Tanjina_ORION_FISH_test_HGSOC\LSP14728\segmentation\LSP14728_001_A107_P54N_HMS_TK\nucleiRing.ome.tif" --counts-path "T:\Orion_FISH_test1\Tanjina_ORION_FISH_test_HGSOC\LSP14728\fish\counts\LSP14728-quantification-fish.csv" --channel-names "DNA,CEP8,CCNE1,MYC,CEP19_602,CEP19_624,peak_CEP8,peak_CCNE1,peak_MYC,peak_CEP19_602,peak_CEP19_624"
    ```

    NOTE: Ensure that the paths are correctly formatted and accessible.  If
    using paths with spaces, enclose them in double quotes.

---

### Script usage

```
usage: visual_check_cli.py [-h] --img-path IMG_PATH --mask-path MASK_PATH --counts-path COUNTS_PATH --channel-names CHANNEL_NAMES

Visual QC for puncta detection: overlay spot counts per cell.

options:
  -h, --help            show this help message and exit
  --img-path IMG_PATH   Path to the multi-channel .ome.tif image
  --mask-path MASK_PATH
                        Path to the labeled mask .ome.tif
  --counts-path COUNTS_PATH
                        Path to the per-cell counts .csv file
  --channel-names CHANNEL_NAMES
                        Comma separated list of all channel names in the image
```
