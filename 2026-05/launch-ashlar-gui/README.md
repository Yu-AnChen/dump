# run-ashlar

Batch-stitch cyclic microscopy slides with
[ashlar](https://github.com/labsyspharm/ashlar) from a simple CSV config file.
Includes a graphical interface for point-and-click operation.

## Installation

This project uses [pixi](https://pixi.sh) to manage the environment. Install
pixi, then run once inside this folder:

```sh
pixi install --locked
```

That installs Python 3.10, ashlar (from source), and all dependencies into an
isolated environment.

## Quick start — GUI

Launch the GUI with pixi:

```sh
pixi run python run-ashlar.py
```

or, if the environment is already activated (`pixi shell`):

```sh
python run-ashlar.py
```

The window that opens contains everything needed to run a batch.

---

## GUI walkthrough

### Input fields

| Field                | Required | Description                                                    |
| -------------------- | -------- | -------------------------------------------------------------- |
| **Config CSV**       | Yes      | CSV listing the slides to process (see format below)           |
| **Markers CSV**      | No       | One channel name per line; written into the output OME-TIFF    |
| **Output directory** | No       | Where output files go. Defaults to next to each slide folder   |
| **Fiji executable**  | No       | Only needed when flat-field correction is requested in the CSV |

Paths can be typed directly, pasted with Windows *Copy as path* (surrounding
quotes are stripped automatically), or picked with the **…** browse button.

### Options row

| Option                    | Default | Description                                                                 |
| ------------------------- | ------- | --------------------------------------------------------------------------- |
| **From slide / To slide** | 0 / all | Process only a slice of the CSV (0-based, *To* is exclusive)                |
| **Max jobs**              | 1       | Number of slides processed in parallel                                      |
| **Max shift µm**          | 30      | Maximum allowed per-tile shift passed to ashlar (`-m`)                      |
| **Filter σ**              | 1.0     | Gaussian pre-filter sigma in pixels (`--filter-sigma`). Set to 0 to disable |
| **Dry run**               | off     | Print ashlar commands without executing them                                |
| **Skip existing**         | off     | Skip any slide whose output OME-TIFF already exists                         |

### Running a batch

1. Fill in the **Config CSV** (and optionally **Markers CSV**).
2. Adjust options if needed.
3. Click **Run ashlar**. The progress bar starts and the console below streams
   live output.
4. To stop early, click **Cancel**. Any slide currently running will finish its
   current step and then stop; slides not yet started are skipped.

### Console and log viewer

The scrolling console shows timestamped log lines for the whole batch. Click
**Clear console** to reset it without affecting any running jobs.

Click **View logs** to open the log viewer window. It contains:

- A **Summary** tab listing the status of every slide (*waiting*, *running*,
  *done*, *failed*).
- A **per-slide tab** for each slide that has started, showing the full ashlar
  output streamed in real time.

A plain-text log file (`<slide-name>-ashlar.log`) is also written next to each
output OME-TIFF for permanent reference. It records the ashlar version, the
exact command used, and all output.

---

## Config CSV format

The CSV must have a header row. Two columns are used:

| Column       | Required | Description                                                        |
| ------------ | -------- | ------------------------------------------------------------------ |
| `Directory`  | Yes      | Path to the slide folder                                           |
| `Correction` | No       | Set to `1`, `yes`, or `true` to run flat-field correction via Fiji |

Example:

```csv
Directory,Correction
D:\data\slide_001,
D:\data\slide_002,1
"C:\Users\Me\Documents\slide 003",
```

**Tips:**

- Windows *Copy as path* wraps paths in quotes — that is accepted and stripped
  automatically.
- Extra columns are ignored.
- Blank lines are ignored.

### Windows shortcut support

If cycle files (`.rcpnl` or `.xdce`) are not stored directly inside the slide
folder, Windows `.lnk` shortcuts are resolved transparently. Both of the
following arrangements work:

- A shortcut pointing directly to a cycle file, e.g. `cycle1.rcpnl -
  Shortcut.lnk`
- A shortcut pointing to a **directory** that contains cycle files

The script searches one level of real subdirectories as well, so cycle files
nested one folder deep are found without shortcuts.

---

## Markers CSV format

A plain text file with **one channel name per line**, no header:

```txt
DAPI
CD45
CD3
panCK
CD68
```

- Leading and trailing blank lines are ignored.
- The number of names must exactly match the number of channels in the output
  OME-TIFF.
- When provided, channel names are embedded in the OME-XML metadata of the
  output file after stitching completes.

---

## Output

For each slide, the script writes two files next to the slide folder (or in the
configured output directory):

| File                      | Description                                |
| ------------------------- | ------------------------------------------ |
| `<slide-name>.ome.tif`    | Pyramidal OME-TIFF produced by ashlar      |
| `<slide-name>-ashlar.log` | Full ashlar log (version, command, output) |

---

## add-channel-name.py — batch-add channel names to existing OME-TIFFs

`add-channel-name.py` is a small standalone GUI for embedding channel names into
OME-TIFF files that have already been stitched. It is useful when channel names
were not available at stitching time, or when names need to be corrected.

Launch it with:

```sh
pixi run python add-channel-name.py
```

### Inputs

| Field | Description |
| ----- | ----------- |
| **OME-TIFF directory** | Folder containing `.ome.tif` files to update |
| **Markers CSV** | Same format as above — one channel name per line, no header |

Windows `.lnk` shortcuts pointing to `.ome.tif` files inside the directory are
resolved automatically.

### Behaviour

- All `.ome.tif` files found in the directory are processed in order.
- The number of names in the markers CSV must exactly match the channel count in
  every file; a mismatch stops that file with an error and moves on to the next.
- Channel names are written in-place into the OME-XML metadata of each file.
- Progress is shown in the log panel; a summary line reports how many files were
  updated successfully.

---

## Command-line mode

The GUI is the recommended interface. For scripted or headless use, pass a CSV
file directly:

```sh
pixi run python run-ashlar.py slides.csv --markers markers.csv --max-n-jobs 4
```

Run `pixi run python run-ashlar.py --help` for all options.
m
