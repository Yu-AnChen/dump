"""Batch-run ashlar from a CSV config file."""

import argparse
import csv
import logging
import platform
import queue
import shlex
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────


def _add_channel_names(tiff_path, channel_names):
    import ome_types
    import tifffile

    ome = ome_types.from_tiff(tiff_path)
    n_channels = len(ome.images[0].pixels.channels)
    n_names = len(channel_names)
    assert n_channels == n_names, (
        f"Number of channels ({n_channels}) in '{tiff_path}' does not match "
        f"number of channel names ({n_names})."
    )
    for channel, name in zip(ome.images[0].pixels.channels, channel_names):
        channel.name = name
    tifffile.tiffcomment(tiff_path, ome.to_xml().encode())


def _text_to_bool(text):
    return bool(text) and str(text).lower() in ("1", "yes", "y", "true", "t")


def _load_markers(markers_path):
    """Return channel names from a headerless one-name-per-line markers CSV."""
    lines = Path(markers_path).read_text().splitlines()
    # strip blank lines from head and tail
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return [ln.strip() for ln in lines]


def _resolve_shortcut(path):
    """Resolve a Windows .lnk shortcut to its target path."""
    if platform.system() == "Windows" and str(path).endswith(".lnk"):
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        try:
            shell = win32com.client.Dispatch("WScript.Shell")
            target = shell.CreateShortCut(str(path)).Targetpath
            assert target != "", f"Shortcut has no target: {path}"
            return Path(target)
        finally:
            pythoncom.CoUninitialize()
    return Path(path)


def _find_cycle_files(slide_dir):
    """Find all cycle scan files (rcpnl or xdce) within a slide directory.

    Searches directly in slide_dir, one level of real subdirectories, and any
    subdirectories reached via Windows .lnk shortcuts.
    """
    # resolve any .lnk entries that point to cycle subdirectories
    shortcut_dirs = []
    for lnk in slide_dir.glob("*.lnk"):
        resolved = _resolve_shortcut(lnk)
        if resolved.is_dir():
            shortcut_dirs.append(resolved)

    for ftype in ("rcpnl", "xdce"):
        real = [*slide_dir.glob(f"*{ftype}"), *slide_dir.glob(f"*/*{ftype}")]
        lnks = [*slide_dir.glob(f"*{ftype}*.lnk"), *slide_dir.glob(f"*/*{ftype}*.lnk")]
        from_shortcut_dirs = [f for d in shortcut_dirs for f in d.glob(f"*{ftype}")]
        files = sorted(
            {*real, *(_resolve_shortcut(p) for p in lnks), *from_shortcut_dirs}
        )
        if files:
            files.sort(key=lambda p: p.stat().st_mtime)
            return files, ftype
    return [], "rcpnl"


def _generate_ffp(cycle_files, slide_dir, file_type, fiji_path, dry_run=False):
    illum_dir = slide_dir / "illumination_profiles"
    ffp_list = []
    for cycle_file in cycle_files:
        stem = cycle_file.name.replace(f".{file_type}", "")
        ffp_name = f"{stem}-ffp.tif"
        ffp_path = illum_dir / ffp_name
        if ffp_path.exists():
            logging.info(f"    FFP exists: {ffp_name}")
        else:
            logging.info(f"    Generating FFP: {ffp_name}")
            if not dry_run:
                illum_dir.mkdir(exist_ok=True)
                plugin = (
                    Path(fiji_path).parent
                    / "plugins"
                    / "imagej_basic_ashlar_ffp_only.py"
                )
                subprocess.run(
                    [
                        str(fiji_path),
                        "--ij2",
                        "--headless",
                        "--run",
                        str(plugin),
                        f"filename='{cycle_file}', output_dir='{illum_dir}',"
                        f" experiment_name='{stem}', lambda_flat=0.1, lambda_dark=0.01",
                    ],
                    shell=False,
                    check=True,
                )
        ffp_list.append(str(ffp_path))
    return ffp_list


# ── core processing ───────────────────────────────────────────────────────────


def _run_ashlar(cmd, log_path, slide_name, pipe_to_console=True, cancel_event=None):
    """Run ashlar via Popen, streaming output to log file and optionally the console."""
    try:
        version = subprocess.check_output(["ashlar", "--version"]).decode().strip()
    except Exception:
        version = "unknown"

    with open(log_path, "w") as log_f:
        log_f.write(
            f"ashlar version:\n{version}\n\nashlar command:\n{shlex.join(cmd)}\n\nashlar output:\n"
        )
        log_f.flush()

        proc = subprocess.Popen(
            cmd,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            if pipe_to_console:
                line_s = line.rstrip()
                if line_s:
                    logging.info(f"[{slide_name}]   {line_s}")
            if cancel_event and cancel_event.is_set():
                proc.terminate()
                log_f.write("\n[cancelled]\n")
                log_f.flush()
                break
        proc.wait()

    return proc.returncode


def process_slide(
    slide,
    *,
    markers_names=None,
    fiji_path=None,
    dry_run=False,
    skip_existing=False,
    maximum_shift=30,
    filter_sigma=1,
    output_dir=None,
    pipe_ashlar_to_console=True,
    cancel_event=None,
):
    """Stitch one slide: find cycle files, run ashlar, optionally add channel names."""
    if cancel_event and cancel_event.is_set():
        return False
    slide_dir = Path(slide["Directory"].strip().strip('"')).resolve()
    out_parent = Path(output_dir).resolve() if output_dir else slide_dir.parent
    out_tif = out_parent / f"{slide_dir.name}.ome.tif"
    log_path = out_parent / f"{slide_dir.name}-ashlar.log"

    if skip_existing and out_tif.exists():
        logging.info(f"[{slide_dir.name}] Skipping — output already exists")
        return True

    cycle_files, file_type = _find_cycle_files(slide_dir)
    if not cycle_files:
        logging.warning(f"[{slide_dir.name}] No rcpnl or xdce files found")
        return False

    logging.info(
        f"[{slide_dir.name}] Found {len(cycle_files)} {file_type} cycle file(s)"
    )

    # flat-field correction profiles
    ffp_list = None
    if _text_to_bool(slide.get("Correction", "")):
        if fiji_path:
            ffp_list = _generate_ffp(
                cycle_files, slide_dir, file_type, fiji_path, dry_run
            )
        else:
            logging.warning(
                f"[{slide_dir.name}] Correction requested but --fiji-path not set"
            )

    # build ashlar command
    # pyramidal OME-TIFF output is automatic when -o ends in .ome.tif
    cmd = [
        "ashlar",
        *[str(f) for f in cycle_files],
        "-m",
        str(maximum_shift),
        "-o",
        str(out_tif),
    ]
    if filter_sigma is not None:
        cmd += ["--filter-sigma", str(filter_sigma)]
    if ffp_list:
        cmd += ["--ffp", *ffp_list]

    logging.info(f"[{slide_dir.name}] {shlex.join(cmd)}")

    if dry_run:
        logging.info(f"[{slide_dir.name}] [dry-run] skipping execution")
        return True

    returncode = _run_ashlar(
        cmd, log_path, slide_dir.name, pipe_ashlar_to_console, cancel_event=cancel_event
    )
    if returncode != 0:
        logging.error(
            f"[{slide_dir.name}] ashlar failed (exit {returncode}) — see {log_path.name}"
        )
        return False

    # add channel names if markers provided
    if markers_names:
        if out_tif.exists():
            logging.info(f"[{slide_dir.name}] Adding channel names")
            try:
                _add_channel_names(str(out_tif), markers_names)
            except Exception as e:
                logging.warning(f"[{slide_dir.name}] Channel name error: {e}")
        else:
            logging.warning(
                f"[{slide_dir.name}] Output not found; skipping channel names"
            )

    logging.info(f"[{slide_dir.name}] Done → {out_tif}")
    return True


def run_batch(slides, *, max_n_jobs=1, cancel_event=None, **kwargs):
    """Run process_slide for each slide, in parallel when max_n_jobs > 1.

    Ashlar output is always written to per-slide log files. It is also piped
    to the console when running a single job; suppressed in parallel mode to
    avoid interleaved output from concurrent slides.
    """
    kwargs.setdefault("pipe_ashlar_to_console", max_n_jobs == 1)
    results = {}
    if max_n_jobs == 1:
        for slide in slides:
            if cancel_event and cancel_event.is_set():
                logging.info("Batch cancelled")
                break
            try:
                results[slide["Directory"]] = process_slide(
                    slide, cancel_event=cancel_event, **kwargs
                )
            except Exception as e:
                logging.error(f"[{slide['Directory']}] Unexpected error: {e}")
                results[slide["Directory"]] = False
    else:
        with ThreadPoolExecutor(max_workers=max_n_jobs) as pool:
            futures = {
                pool.submit(
                    process_slide, slide, cancel_event=cancel_event, **kwargs
                ): slide
                for slide in slides
            }
            try:
                for fut in as_completed(futures):
                    slide = futures[fut]
                    try:
                        results[slide["Directory"]] = fut.result()
                    except Exception as e:
                        logging.error(f"[{slide['Directory']}] Unexpected error: {e}")
                        results[slide["Directory"]] = False
                    if cancel_event and cancel_event.is_set():
                        for f in futures:
                            f.cancel()
                        logging.info("Batch cancelled")
                        break
            except KeyboardInterrupt:
                if cancel_event:
                    cancel_event.set()
                for f in futures:
                    f.cancel()
                logging.info("Interrupted — cancelling remaining slides")

    n_ok = sum(v for v in results.values())
    n_total = len(results)
    logging.info(f"Finished: {n_ok}/{n_total} slide(s) succeeded")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser():
    p = argparse.ArgumentParser(
        description="Batch-run ashlar from a CSV config (columns: Directory, Correction)."
    )
    p.add_argument(
        "csv_filepath",
        metavar="CSVFILE",
        nargs="?",
        help="CSV file with header: Directory, Correction",
    )
    p.add_argument(
        "-f",
        "--from-dir",
        type=int,
        default=0,
        metavar="FROM",
        help="Starting slide index (0-based, default 0)",
    )
    p.add_argument(
        "-t",
        "--to-dir",
        type=int,
        default=None,
        metavar="TO",
        help="Ending slide index, exclusive (default: all)",
    )
    p.add_argument(
        "--markers",
        metavar="MARKERS_CSV",
        help="headerless CSV with one channel name per line",
    )
    p.add_argument(
        "--output-dir",
        metavar="DIR",
        help="directory for output OME-TIFFs (default: next to each slide folder)",
    )
    p.add_argument(
        "--fiji-path",
        metavar="PATH",
        default="C:/Users/Public/Downloads/Fiji.app/ImageJ-win64.exe",
        help="Fiji executable path (default: C:/Users/Public/Downloads/Fiji.app/ImageJ-win64.exe)",
    )
    p.add_argument(
        "--max-n-jobs",
        type=int,
        default=1,
        metavar="N",
        help="Max parallel ashlar jobs (default: 1)",
    )
    p.add_argument(
        "--maximum-shift",
        type=int,
        default=30,
        metavar="SHIFT",
        help="Maximum per-tile corrective shift in microns (default: 30)",
    )
    p.add_argument(
        "--filter-sigma",
        type=float,
        default=1,
        metavar="SIGMA",
        help="Gaussian pre-filter sigma in pixels (default: 1)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip slides whose output OME-TIFF already exists",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )
    p.add_argument("--gui", action="store_true", help="Launch the graphical interface")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def cli_main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.gui or args.csv_filepath is None:
        launch_gui()
        return

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    csv_path = Path(args.csv_filepath)
    if not csv_path.is_file() or csv_path.suffix != ".csv":
        parser.error("csv_filepath must be an existing .csv file")

    with open(csv_path, newline="") as f:
        slides = list(csv.DictReader(f))

    markers_names = _load_markers(args.markers) if args.markers else None

    cancel_event = threading.Event()
    try:
        run_batch(
            slides[args.from_dir : args.to_dir],
            max_n_jobs=args.max_n_jobs,
            cancel_event=cancel_event,
            markers_names=markers_names,
            fiji_path=args.fiji_path,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            maximum_shift=args.maximum_shift,
            filter_sigma=args.filter_sigma,
            output_dir=args.output_dir,
        )
    except KeyboardInterrupt:
        cancel_event.set()
        logging.info("\nInterrupted.")
        sys.exit(1)


# ── GUI ───────────────────────────────────────────────────────────────────────


def launch_gui():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, scrolledtext, ttk
    except ImportError:
        sys.exit("tkinter is not available; use command-line mode instead")

    root = tk.Tk()
    root.title("run-ashlar")
    root.resizable(True, True)

    csv_var = tk.StringVar()
    markers_var = tk.StringVar()
    output_dir_var = tk.StringVar()
    fiji_var = tk.StringVar(value="C:/Users/Public/Downloads/Fiji.app/ImageJ-win64.exe")
    from_var = tk.IntVar(value=0)
    to_var = tk.StringVar(value="")
    jobs_var = tk.IntVar(value=1)
    margin_var = tk.IntVar(value=30)
    sigma_var = tk.DoubleVar(value=1.0)
    dry_var = tk.BooleanVar(value=False)
    skip_var = tk.BooleanVar(value=False)

    frm = ttk.Frame(root, padding=12)
    frm.grid(sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frm.columnconfigure(1, weight=1)
    frm.rowconfigure(6, weight=1)

    def _file_row(row, label, var, filetypes):
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=var, width=54).grid(
            row=row, column=1, padx=4, sticky="ew"
        )
        ttk.Button(
            frm,
            text="…",
            width=2,
            command=lambda: var.set(filedialog.askopenfilename(filetypes=filetypes)),
        ).grid(row=row, column=2)

    def _dir_row(row, label, var):
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=var, width=54).grid(
            row=row, column=1, padx=4, sticky="ew"
        )
        ttk.Button(
            frm,
            text="…",
            width=2,
            command=lambda: var.set(filedialog.askdirectory()),
        ).grid(row=row, column=2)

    _file_row(0, "Config CSV *", csv_var, [("CSV", "*.csv"), ("All", "*.*")])
    _file_row(1, "Markers CSV", markers_var, [("CSV", "*.csv"), ("All", "*.*")])
    _dir_row(2, "Output directory", output_dir_var)
    _file_row(3, "Fiji executable", fiji_var, [("All", "*.*")])

    opts = ttk.Frame(frm)
    opts.grid(row=4, column=0, columnspan=3, sticky="w", pady=6)

    ttk.Label(opts, text="From slide").pack(side="left", padx=(0, 2))
    ttk.Spinbox(opts, textvariable=from_var, from_=0, to=9999, width=5).pack(
        side="left", padx=(0, 6)
    )
    ttk.Label(opts, text="To slide").pack(side="left", padx=(0, 2))
    ttk.Entry(opts, textvariable=to_var, width=5).pack(side="left", padx=(0, 16))

    for label, var, lo, hi, inc, w in [
        ("Max jobs", jobs_var, 1, 64, 1, 4),
        ("Max shift µm", margin_var, 0, 500, 5, 5),
        ("Filter σ", sigma_var, 0, 10, 0.5, 4),
    ]:
        ttk.Label(opts, text=label).pack(side="left", padx=(0, 2))
        ttk.Spinbox(
            opts, textvariable=var, from_=lo, to=hi, increment=inc, width=w
        ).pack(side="left", padx=(0, 10))

    ttk.Checkbutton(opts, text="Dry run", variable=dry_var).pack(
        side="left", padx=(6, 6)
    )
    ttk.Checkbutton(opts, text="Skip existing", variable=skip_var).pack(side="left")

    ttk.Separator(frm, orient="horizontal").grid(
        row=5, column=0, columnspan=3, sticky="ew", pady=6
    )

    log_text = scrolledtext.ScrolledText(
        frm, height=18, width=84, state="disabled", font=("Courier", 10)
    )
    log_text.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(0, 6))

    prog = ttk.Progressbar(frm, mode="indeterminate")
    prog.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(0, 6))

    # shared state for the log viewer
    active_log_paths = []  # (log_path, out_tif_path) per slide in current batch
    batch_done = [True]  # flipped False→True around each batch run
    batch_start_time = [0.0]  # set to time.time() at batch start to ignore stale logs
    cancel_event = threading.Event()

    def _open_log_viewer():
        if not active_log_paths:
            messagebox.showinfo("No logs", "Run a batch first.")
            return

        win = tk.Toplevel(root)
        win.title("Ashlar log viewer")
        win.resizable(True, True)
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(win)
        notebook.grid(sticky="nsew", padx=8, pady=8)

        # ── Summary tab ──────────────────────────────────────────────────────
        sum_frame = ttk.Frame(notebook)
        sum_frame.columnconfigure(0, weight=1)
        sum_frame.rowconfigure(0, weight=1)
        sum_txt = scrolledtext.ScrolledText(
            sum_frame, height=20, width=60, font=("Courier", 10), state="disabled"
        )
        sum_txt.grid(sticky="nsew")
        notebook.add(sum_frame, text="Summary")

        # ── per-slide log tabs added dynamically ─────────────────────────────
        positions = {log: 0 for log, _ in active_log_paths}
        slide_tabs = {}  # log_path → ScrolledText, added on first file appearance

        def _add_slide_tab(log_path):
            slide_name = log_path.name.replace("-ashlar.log", "")
            frame = ttk.Frame(notebook)
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
            txt = scrolledtext.ScrolledText(
                frame, height=30, width=100, font=("Courier", 9), state="disabled"
            )
            txt.grid(sticky="nsew")
            notebook.add(frame, text=slide_name)
            slide_tabs[log_path] = txt
            return txt

        def _update_summary():
            lines = ["  {:<10}  {}".format("status", "slide"), "  " + "─" * 46]
            for log_path, out_tif in active_log_paths:
                slide_name = log_path.name.replace("-ashlar.log", "")
                log_current = (
                    log_path.exists()
                    and log_path.stat().st_mtime >= batch_start_time[0]
                )
                if out_tif.exists():
                    status = "done"
                elif log_current:
                    status = "failed" if batch_done[0] else "running"
                else:
                    status = "waiting" if not batch_done[0] else "---"
                lines.append(f"  {status:<10}  {slide_name}")
            content = "\n".join(lines) + "\n"
            sum_txt.configure(state="normal")
            sum_txt.delete("1.0", "end")
            sum_txt.insert("end", content)
            sum_txt.configure(state="disabled")

        def _poll_files():
            _update_summary()
            for log_path, _ in active_log_paths:
                if (
                    log_path.exists()
                    and log_path.stat().st_mtime >= batch_start_time[0]
                ):
                    if log_path not in slide_tabs:
                        _add_slide_tab(log_path)
                    try:
                        with open(log_path) as f:
                            f.seek(positions[log_path])
                            new = f.read()
                            positions[log_path] = f.tell()
                        if new:
                            txt = slide_tabs[log_path]
                            txt.configure(state="normal")
                            txt.insert("end", new)
                            txt.see("end")
                            txt.configure(state="disabled")
                    except Exception:
                        pass
            if win.winfo_exists():
                win.after(100, _poll_files)

        win.after(100, _poll_files)

    btn_bar = ttk.Frame(frm)
    btn_bar.grid(row=8, column=0, columnspan=3, pady=(0, 2))
    btn_run = ttk.Button(btn_bar, text="Run ashlar")
    btn_run.pack(side="left", padx=(0, 8))
    btn_cancel = ttk.Button(btn_bar, text="Cancel", state="disabled")
    btn_cancel.pack(side="left", padx=(0, 8))
    ttk.Button(
        btn_bar,
        text="Clear console",
        command=lambda: (
            log_text.configure(state="normal"),
            log_text.delete("1.0", "end"),
            log_text.configure(state="disabled"),
        ),
    ).pack(side="left", padx=(0, 8))
    ttk.Button(btn_bar, text="View logs", command=_open_log_viewer).pack(side="left")

    # redirect logging to the text widget via a queue
    log_queue: queue.Queue = queue.Queue()

    class _QueueHandler(logging.Handler):
        def emit(self, record):
            log_queue.put(self.format(record))

    _handler = _QueueHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(_handler)
    root_logger.setLevel(logging.INFO)

    def _poll_log():
        while True:
            try:
                msg = log_queue.get_nowait()
                log_text.configure(state="normal")
                log_text.insert("end", msg + "\n")
                log_text.see("end")
                log_text.configure(state="disabled")
            except queue.Empty:
                break
        root.after(120, _poll_log)

    root.after(120, _poll_log)

    def _on_run():
        csv_str = csv_var.get().strip().strip('"')
        if not csv_str:
            messagebox.showerror("Missing", "Please select a config CSV.")
            return
        csv_p = Path(csv_str)
        if not csv_p.is_file():
            messagebox.showerror("Not found", f"Config CSV not found:\n{csv_p}")
            return

        with open(csv_p, newline="") as f:
            slides = list(csv.DictReader(f))

        to_raw = to_var.get().strip()
        to_idx = int(to_raw) if to_raw else None
        subset = slides[from_var.get() : to_idx]

        markers_names = None
        m_path = markers_var.get().strip().strip('"')
        if m_path:
            try:
                markers_names = _load_markers(m_path)
            except Exception as e:
                messagebox.showerror("Markers error", str(e))
                return

        fiji = fiji_var.get().strip().strip('"') or None
        output_dir = output_dir_var.get().strip().strip('"') or None
        # sigma=0 in the spinbox means "no filtering"
        sigma = sigma_var.get() or None

        # precompute log + output paths so the viewer can open immediately
        active_log_paths.clear()
        for slide in subset:
            slide_dir = Path(slide["Directory"].strip().strip('"')).resolve()
            out_parent = Path(output_dir).resolve() if output_dir else slide_dir.parent
            active_log_paths.append(
                (
                    out_parent / f"{slide_dir.name}-ashlar.log",
                    out_parent / f"{slide_dir.name}.ome.tif",
                )
            )

        cancel_event.clear()
        batch_done[0] = False
        batch_start_time[0] = time.time()
        btn_run.configure(state="disabled")
        btn_cancel.configure(state="normal")
        prog.start(10)

        def _worker():
            try:
                run_batch(
                    subset,
                    max_n_jobs=jobs_var.get(),
                    cancel_event=cancel_event,
                    markers_names=markers_names,
                    fiji_path=fiji,
                    dry_run=dry_var.get(),
                    skip_existing=skip_var.get(),
                    maximum_shift=margin_var.get(),
                    filter_sigma=sigma,
                    output_dir=output_dir,
                )
            except Exception as e:
                logging.error(f"Batch error: {e}")
            finally:
                batch_done[0] = True
                root.after(
                    0,
                    lambda: (
                        btn_run.configure(state="normal"),
                        btn_cancel.configure(state="disabled"),
                        prog.stop(),
                    ),
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _on_cancel():
        cancel_event.set()
        btn_cancel.configure(state="disabled")
        logging.info("Cancelling — waiting for running slides to finish…")

    btn_run.configure(command=_on_run)
    btn_cancel.configure(command=_on_cancel)
    root.mainloop()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli_main()
