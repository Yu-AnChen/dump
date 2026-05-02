#!/usr/bin/env python3
"""GUI to batch-add channel names to all OME-TIFF files in a directory."""

import logging
import platform
import queue
import threading
from pathlib import Path


def _load_markers(markers_path):
    lines = Path(markers_path).read_text().splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return [ln.strip() for ln in lines]


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


def _resolve_shortcut(path):
    """Resolve a Windows .lnk shortcut to its target; return path unchanged otherwise."""
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


def _find_tiff_files(tiff_dir):
    """Return resolved .ome.tif paths, including those pointed to by .lnk shortcuts."""
    real = list(tiff_dir.glob("*.ome.tif"))
    lnks = [_resolve_shortcut(p) for p in tiff_dir.glob("*.ome.tif.lnk")]
    return sorted({*real, *lnks})


def main():
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    root = tk.Tk()
    root.title("Add Channel Names")
    root.resizable(True, True)

    tiff_dir_var = tk.StringVar()
    markers_var = tk.StringVar()

    frm = ttk.Frame(root, padding=12)
    frm.grid(sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frm.columnconfigure(1, weight=1)
    frm.rowconfigure(3, weight=1)

    def _file_row(row, label, var, pick_dir=False):
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=var, width=54).grid(row=row, column=1, padx=4, sticky="ew")
        if pick_dir:
            def _pick():
                var.set(filedialog.askdirectory())
        else:
            def _pick():
                var.set(filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")]))
        ttk.Button(frm, text="…", width=2, command=_pick).grid(row=row, column=2)

    _file_row(0, "OME-TIFF directory *", tiff_dir_var, pick_dir=True)
    _file_row(1, "Markers CSV *", markers_var)

    ttk.Separator(frm, orient="horizontal").grid(
        row=2, column=0, columnspan=3, sticky="ew", pady=6
    )

    log_text = scrolledtext.ScrolledText(
        frm, height=16, width=72, state="disabled", font=("Courier", 10)
    )
    log_text.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=(0, 6))

    prog = ttk.Progressbar(frm, mode="indeterminate")
    prog.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 6))

    btn_run = ttk.Button(frm, text="Add channel names")
    btn_run.grid(row=5, column=0, columnspan=3, pady=(0, 2))

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
        dir_str = tiff_dir_var.get().strip()
        markers_str = markers_var.get().strip()

        if not dir_str or not markers_str:
            messagebox.showerror("Missing", "Both fields are required.")
            return

        tiff_dir = Path(dir_str)
        if not tiff_dir.is_dir():
            messagebox.showerror("Not found", f"Directory not found:\n{tiff_dir}")
            return

        try:
            markers = _load_markers(markers_str)
        except Exception as e:
            messagebox.showerror("Markers error", str(e))
            return

        tiff_files = _find_tiff_files(tiff_dir)
        if not tiff_files:
            messagebox.showwarning("No files", f"No .ome.tif files found in:\n{tiff_dir}")
            return

        btn_run.configure(state="disabled")
        prog.start(10)

        def _worker():
            n_ok = 0
            for tif in tiff_files:
                logging.info(f"Processing: {tif.name}")
                try:
                    _add_channel_names(str(tif), markers)
                    logging.info("  Done")
                    n_ok += 1
                except Exception as e:
                    logging.error(f"  Failed: {e}")
            logging.info(f"Finished: {n_ok}/{len(tiff_files)} file(s) updated")
            root.after(0, lambda: (btn_run.configure(state="normal"), prog.stop()))

        threading.Thread(target=_worker, daemon=True).start()

    btn_run.configure(command=_on_run)
    root.mainloop()


if __name__ == "__main__":
    main()
