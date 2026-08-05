*CyLinter (v0.0.56) on 184-CRC-POLYPS — running off the share, no data copied*

CyLinter is now running on `\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\184-CRC-polyps\cycif` from a Windows box. 24 samples, `mccellpose` cell segmentation. *Zero bytes copied* — one folder rename plus hardlinks. Notes below apply to any mcmicro-derived project.

*Key finding: don't restage your data*

CyLinter's `mcmicro_TMA` layout (`utils.py::input_check`) already matches standard mcmicro output. "TMA" only selects path resolution — `check` is read solely by `get_filepath()` and `aggregateData.py:36`, so no TMA-specific analysis runs. Fine for WSIs. Point `inDir` at the mcmicro folder itself.

*What we changed*

1️⃣ *Renamed* `registration\` → `dearray\` — instant, no data movement. (`dearray` is hardcoded; no config knob.)

2️⃣ *Hardlinked the Cellpose masks into* `qc\s3seg\` — every module loads a SEG layer, but mcmicro Cellpose only produces label masks in `segmentation\`. Hardlinks work on our filer (test: `mklink /H test.csv markers.csv`), so this costs nothing:
```
segmentation\mccellpose-<sample>\cell.ome.tif  ->  qc\s3seg\mccellpose-<sample>\cell.ome.tif
```
Reusing the mask is fine: in 14 of 15 modules SEG is display-only (`single_channel_pyramid(..., channel=0)`, added to Napari `visible=False`) — nothing computes on the pixels. It renders as filled cells, not outlines; widen the contrast since label IDs run to the cell count.
⚠️ *Washed-out thumbnails?* That's the only place the mask isn't good enough — `curateThumbnails` alpha-blends SEG over each thumbnail (`curateThumbnails.py:365-377`) and a filled mask swamps it. Either set `segOutlines: False`, or re-run the script with `--outlines generate` to compute real boundaries (it replaces the placeholders in place, leaving everything else alone).
⚠️ *The seg file and the mask are now the same file.* If these samples ever get re-segmented, re-run `prep_inplace.py` afterwards — otherwise `qc\s3seg\` is either silently rewritten along with the mask, or left stale pointing at the old content.

3️⃣ *Fixed* `markers.csv` — `E-cadherin` appeared twice (cycle 8 ch36, cycle 11 ch49). Renamed to `E-cadherin_1` / `E-cadherin_2` to match what mcquant already wrote in the CSV header.

4️⃣ *Updated* `cylinter_config.yml`:
```
inDir/outDir            -> the cycif folder itself / cycif\cylinter-output
sampleMetadata keys     -> LSP32015_P184--mccellpose_cell   (was LSP32015)
sampleMetadata names    -> LSP32015_P184                    (was LSP32015)
counterstainChannel     -> "Hoechst1"                       (was "Hoechst")
markersToExclude        -> ["A488","A555","A647"]           (cycle-1 background)
samplesForROISelection  -> all 24 names, one per line so they can be commented out
classes: Tumor          -> [+panCK, +E-cadherin_1]          ("Pan-cytokeratin" didn't exist)
```

*Gotchas worth knowing (all cost us time)*

• `sampleMetadata` *keys* must be the quantification CSV basename: `<sample>--<method>_<object>`. A key without `--` dies with a bare `IndexError` at `utils.py:158`, no message.
• The *first value element must equal the key up to* `--` (`LSP32015_P184`). Undocumented — `aggregateData.py:39` splits the key and `get_filepath()` reverse-maps through the names (`utils.py:307`). Mismatch = every path resolves to `None`.
• `samplesForROISelection` takes *sample names*, not keys; anything unknown aborts `selectROIs` (`selectROIs.py:109`).
• *Duplicate `marker_name` is fatal and misreported* — `marker_channel_number` calls `.item()` on the index lookup (`utils.py:404`), which raises on 2 matches and prints *"Aborting; E-cadherin not found in markers.csv"*. Sends you chasing a typo that isn't there. *Diff markers.csv against your quantification CSV header before starting.*
• `counterstainChannel` must match a `marker_name` exactly and be the cycle-1 DNA channel. CyLinter strips trailing digits for a "DNA moniker" (`Hoechst1` → `Hoechst`) and treats every marker containing it as a DNA cycle.
• `markersToExclude` does *not* shift channel indices (boolean mask preserves row indices) — safe to use freely.

*Where everything lives*

All three files sit in the project folder, ready to copy for your own project:
```
\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\184-CRC-polyps\cycif\
    prep_inplace.py        the prep script (steps 1 and 2 above)
    markers.csv            fixed, with E-cadherin_1 / E-cadherin_2
    cylinter_config.yml    validated config for all 24 samples
```

*The script*

`prep_inplace.py` does steps 1 and 2, journals everything to `.cylinter_prep_manifest.json`, and is fully revertible:
```
python prep_inplace.py --src "\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\184-CRC-polyps\cycif" --dry-run
python prep_inplace.py --src "\\research.files...\184-CRC-polyps\cycif" --samples LSP32015_P184   # cheap trial
python prep_inplace.py --src "\\research.files...\184-CRC-polyps\cycif"                          # all 24
python prep_inplace.py --revert "\\research.files...\184-CRC-polyps\cycif\.cylinter_prep_manifest.json"
```
Defaults to `--outlines mask`; `--outlines generate` computes real boundaries later and replaces the placeholders. `--keep-registration` hardlinks into `dearray\` instead of renaming, if other pipelines read `registration\`.

*Recommended first run*: stage one sample, list only it in `sampleMetadata`, run CyLinter. Exercises Napari-over-RDP, SMB image loading, and the mask-as-seg question at once, with a one-command undo.

Questions welcome — happy to help adapt this to another project.
