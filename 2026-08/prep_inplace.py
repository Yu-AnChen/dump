"""
Minimal in-place prep of the 184-CRC-POLYPS CYCIF folder for CyLinter v0.0.56.

CyLinter's 'mcmicro_TMA' path-resolution mode (cylinter/utils.py:82-139) already
matches this folder's layout on 3 of 4 counts:

    inDir/markers.csv                                        already correct
    inDir/quantification/<sample>--<method>_<object>.csv      already correct
    inDir/segmentation/<method>-<sample>/<object>.ome.tif     already correct
    inDir/dearray/<sample>.ome.tif                            <- rename registration/
    inDir/qc/s3seg/<method>-<sample>/<object>.ome.tif         <- must be generated

"TMA" here only selects how file paths are resolved -- `check` is read solely by
get_filepath() and aggregateData.py:36, so no TMA-specific analysis is triggered.

This script therefore does exactly two things:
  1. renames  <SRC>/registration -> <SRC>/dearray   (instant, reversible)
  2. populates <SRC>/qc/s3seg/<method>-<sample>/<object>.ome.tif

For step 2, --outlines mask (the default) reuses the existing label mask instead
of computing boundaries. That is enough for every QC module: they load SEG only
as a display layer via single_channel_pyramid(..., channel=0) and add it to
Napari with visible=False -- nothing computes on the pixel values. It renders as
filled cells with a cell-ID gradient rather than thin outlines.

Real boundaries (--outlines generate) only matter for curateThumbnails when
segOutlines is True: there SEG is piped through cellcutter and alpha-blended over
each thumbnail (curateThumbnails.py:365-377), where a filled mask washes the
thumbnail out. curateThumbnails is the last module, and segOutlines: False skips
it entirely -- so start with mask, and only generate if the thumbnails matter.

Set inDir in cylinter_config.yml to <SRC> itself.

    python prep_inplace.py --src <SRC> --dry-run
    python prep_inplace.py --src <SRC>                      # --outlines mask
    python prep_inplace.py --src <SRC> --outlines generate  # later, if needed
    python prep_inplace.py --revert <SRC>/.cylinter_prep_manifest.json

Trial cheaply first: --samples LSP32015_P184 stages one sample only.

If renaming registration/ would break other consumers, use --keep-registration:
it populates dearray/ with hardlinks instead (needs SMB hardlink support -- test
with `mklink /H test.csv markers.csv`) and falls back to nothing if unsupported.
"""

import argparse
import glob
import json
import os
import shutil
import sys

MANIFEST_NAME = ".cylinter_prep_manifest.json"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--revert", metavar="MANIFEST", help="undo a previous run and exit")
    p.add_argument("--src", help="CYCIF directory (this becomes inDir)")
    p.add_argument("--method", default="mccellpose", help="segmentation method prefix")
    p.add_argument(
        "--object",
        default="cell",
        choices=["cell", "nucleus"],
        help="segmentation object to analyze (pick ONE)",
    )
    p.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="sample names (default: every sample in registration/ or dearray/)",
    )
    p.add_argument(
        "--keep-registration",
        action="store_true",
        help="leave registration/ in place and hardlink into dearray/ instead "
        "of renaming; requires hardlink support on the share",
    )
    p.add_argument(
        "--outlines",
        default="mask",
        choices=["mask", "generate"],
        help="'mask' (default) reuses the label mask as the seg layer -- free with "
        "hardlinks, otherwise one copy per sample. 'generate' computes real "
        "boundaries: slow, loads each full-resolution mask into RAM, but only "
        "needed for curateThumbnails with segOutlines: True.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not args.revert and not args.src:
        p.error("--src is required unless --revert is given")
    return args


class Journal:
    """Append-only record of filesystem changes, so repeated runs stay revertible."""

    def __init__(self, path, dry_run):
        self.path, self.dry_run = path, dry_run
        # carry forward earlier runs; otherwise a second pass would drop the
        # registration -> dearray rename and --revert would leave it renamed
        self.ops = json.load(open(path)) if os.path.isfile(path) else []
        self.prior = {op["dst"]: op["action"] for op in self.ops}

    def record(self, action, src, dst):
        self.ops = [op for op in self.ops if op["dst"] != dst]
        self.ops.append({"action": action, "src": src, "dst": dst})
        if not self.dry_run:
            with open(self.path, "w") as f:
                json.dump(self.ops, f, indent=2)


def place_mask_as_seg(mask_path, out_path, journal):
    """Reuse the label mask as the seg layer: hardlink if the share allows, else copy."""
    if os.path.lexists(out_path):
        print(f"  exists    {out_path}")
        return
    if journal.dry_run:
        print(f"  link/copy {out_path}")
        journal.record("hardlink", mask_path, out_path)
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        os.link(mask_path, out_path)
        action = "hardlink"
    except (OSError, NotImplementedError):
        shutil.copy2(mask_path, out_path)
        action = "copy"
    print(f"  {action:9s} {out_path}")
    journal.record(action, mask_path, out_path)


def make_outlines(mask_path, out_path, journal):
    """Boundary image from a label mask. Loads the full mask into RAM."""
    if os.path.lexists(out_path):
        # upgrade a placeholder left by --outlines mask; keep real outlines as-is
        if journal.prior.get(out_path) in ("hardlink", "copy"):
            print(f"  replace   {out_path} (was the label mask)")
            if not journal.dry_run:
                os.remove(out_path)
        else:
            print(f"  exists    {out_path}")
            return
    print(f"  outlines  {out_path}")
    if journal.dry_run:
        journal.record("generate", mask_path, out_path)
        return
    import tifffile
    from skimage.segmentation import find_boundaries
    from skimage.util import img_as_ubyte

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    labels = tifffile.imread(mask_path)
    seg = img_as_ubyte(find_boundaries(labels))
    del labels
    tmp = out_path + ".partial"
    tifffile.imwrite(tmp, seg, compression="deflate", tile=(1024, 1024))
    os.replace(tmp, out_path)
    journal.record("generate", mask_path, out_path)


def revert(manifest_path):
    with open(manifest_path) as f:
        ops = json.load(f)
    print(f"Reverting {len(ops)} operation(s)\n")
    for op in reversed(ops):
        src, dst = op["src"], op["dst"]
        if not os.path.lexists(dst):
            print(f"  gone      {dst}")
        elif op["action"] == "rename":
            print(f"  rename    {dst} -> {src}")
            os.rename(dst, src)
        else:
            print(f"  delete    {dst}")
            os.remove(dst)
    root = os.path.dirname(os.path.abspath(manifest_path))
    for dirpath, _, _ in os.walk(os.path.join(root, "qc"), topdown=False):
        if os.path.isdir(dirpath) and not os.listdir(dirpath):
            os.rmdir(dirpath)
    os.remove(manifest_path)
    print("\nDone.")


def main():
    args = parse_args()
    if args.revert:
        return revert(args.revert)

    src = os.path.abspath(args.src)
    if not os.path.isfile(os.path.join(src, "markers.csv")):
        sys.exit(f"Aborting; {os.path.join(src, 'markers.csv')} not found.")

    journal = Journal(os.path.join(src, MANIFEST_NAME), args.dry_run)
    registration = os.path.join(src, "registration")
    dearray = os.path.join(src, "dearray")

    # ---- step 1: dearray/ ---------------------------------------------------
    print("Step 1: dearray/")
    image_source = dearray if os.path.isdir(dearray) else registration
    if os.path.isdir(dearray) and not args.keep_registration:
        print(f"  exists    {dearray}")
    elif args.keep_registration:
        for path in sorted(glob.glob(os.path.join(registration, "*.tif"))):
            dst = os.path.join(dearray, os.path.basename(path))
            if os.path.lexists(dst):
                continue
            print(f"  hardlink  dearray/{os.path.basename(path)}")
            if args.dry_run:
                journal.record("hardlink", path, dst)
                continue
            os.makedirs(dearray, exist_ok=True)
            try:
                os.link(path, dst)
            except (OSError, NotImplementedError) as e:
                sys.exit(
                    f"\nAborting; hardlink failed ({e}). This share does not support "
                    "them -- drop --keep-registration to rename the directory "
                    f"instead. Revert what was done with --revert {journal.path}"
                )
            journal.record("hardlink", path, dst)
        image_source = dearray
    else:
        if not os.path.isdir(registration):
            sys.exit(f"Aborting; neither {registration} nor {dearray} exists.")
        print("  rename    registration/ -> dearray/")
        if not args.dry_run:
            os.rename(registration, dearray)
        journal.record("rename", registration, dearray)
        image_source = dearray
    print()

    # ---- step 2: qc/s3seg outlines -----------------------------------------
    # under --dry-run nothing has actually moved, so scan wherever the images are now
    if not os.path.isdir(image_source):
        image_source = registration

    samples = args.samples or sorted(
        os.path.basename(p).rsplit(".ome.tif", 1)[0].rsplit(".tif", 1)[0]
        for p in glob.glob(os.path.join(image_source, "*.tif"))
    )
    if not samples:
        sys.exit(f"Aborting; no images found in {image_source}")

    print(
        f"Step 2: qc/s3seg for {len(samples)} sample(s) ({args.object}, "
        f"--outlines {args.outlines})"
    )
    problems = []
    for sample in samples:
        seg_dir = f"{args.method}-{sample}"
        masks = sorted(
            glob.glob(os.path.join(src, "segmentation", seg_dir, f"{args.object}*.tif"))
        )
        csvs = sorted(
            glob.glob(
                os.path.join(
                    src, "quantification", f"{sample}--{args.method}_{args.object}*.csv"
                )
            )
        )
        if not masks or not csvs:
            missing = "mask" if not masks else "quantification CSV"
            print(f"  MISSING   {sample}: no {missing}")
            problems.append(sample)
            continue
        out_path = os.path.join(src, "qc", "s3seg", seg_dir, os.path.basename(masks[0]))
        if args.outlines == "generate":
            make_outlines(masks[0], out_path, journal)
        else:
            place_mask_as_seg(masks[0], out_path, journal)

    print()
    if problems:
        print(
            f"Samples with missing inputs -- leave these OUT of sampleMetadata: "
            f"{problems}\n"
        )
    print(f"Journal: {journal.path}")
    print(
        f"Set inDir to {src} and use sampleMetadata keys of the form "
        f'"<sample>--{args.method}_{args.object}".'
    )
    if args.outlines == "mask":
        print(
            "\nqc/s3seg holds the label mask, not real outlines. The 'segmentation' "
            "Napari layer will show filled cells (it is hidden by default; toggle it "
            "on and widen the contrast range to see anything -- label IDs run to the "
            "cell count, so most cells sit near black). If that is good enough, you "
            "never need real outlines. Set segOutlines: False for curateThumbnails, "
            "or re-run with --outlines generate to upgrade just those files."
        )


if __name__ == "__main__":
    main()
