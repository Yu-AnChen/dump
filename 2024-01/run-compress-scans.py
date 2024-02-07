import argparse
import asyncio
import datetime
import pathlib
import shlex
import textwrap

from hachiko.hachiko import AIOEventHandler, AIOWatchdog


def _now():
    return f"{datetime.datetime.now():%Y-%m-%d %X}"


# def _is_transferring(filepath):
#     import time

#     filepath = pathlib.Path(filepath)
#     if filepath.is_dir():
#         past_size = sum(f.stat().st_size for f in filepath.glob("**/*") if f.is_file())
#     else:
#         past_size = filepath.stat().st_size
#     print(_now(), past_size)
#     time.sleep(1)
#     if filepath.is_dir():
#         current_size = sum(
#             f.stat().st_size for f in filepath.glob("**/*") if f.is_file()
#         )
#     else:
#         current_size = filepath.stat().st_size
#     print(_now(), current_size)
#     return past_size != current_size


async def _is_transferring(filepath):
    filepath = pathlib.Path(filepath)
    if filepath.is_dir():
        # await asyncio.sleep(1)
        files = filter(lambda x: x.is_file(), filepath.rglob("*"))
    else:
        files = [filepath]
    for ff in files:
        try:
            with open(ff, "a+") as _:
                pass
        except PermissionError:
            return True
    return False


async def run(cmd):
    cmd = cmd.replace("'", '"')
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    fmt = FORMAT

    print(fmt.format(datetime=_now(), pid=proc.pid, msg=f"[[{cmd}]]"))
    stdout, stderr = await proc.communicate()

    if stdout:
        print(
            fmt.format(datetime=_now(), pid=proc.pid, msg=f"STDOUT {stdout.decode()}")
        )
    if stderr:
        print(
            fmt.format(datetime=_now(), pid=proc.pid, msg=f"STDERR {stderr.decode()}")
        )
    return proc


class ScansFolderEventHandler(AIOEventHandler):
    def __init__(self, temp_dir, out_dir, loop=None):
        super().__init__(loop)
        self.temp_dir = pathlib.Path(temp_dir)
        self.out_dir = pathlib.Path(out_dir)
        if not self.temp_dir.exists():
            self.temp_dir.mkdir(parents=True)
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)
        self._fmt = FORMAT

    def validate_created_path(self, filepath):
        filepath = pathlib.Path(filepath)
        # must be a scan folder
        if not filepath.is_dir():
            return False
        # must contains >= 1 rcpnl file
        if next(filepath.glob("*.rcpnl"), None) is None:
            return False
        # must not be the utility/target dirs
        if filepath == self.temp_dir:
            return False
        if filepath == self.out_dir:
            return False
        if filepath.name.startswith("compressing-"):
            return
        return True

    async def on_created(self, event):
        await asyncio.sleep(1)
        filepath = event.src_path
        filepath = pathlib.Path(filepath)

        was_transferring = await _is_transferring(filepath)
        print_transferring = True
        while await _is_transferring(filepath):
            if print_transferring:
                print(
                    self._fmt.format(
                        datetime=_now(),
                        pid="Main",
                        msg=f"Transferring {filepath.name}...",
                    )
                )
                print_transferring = False
            await asyncio.sleep(1)
        if was_transferring:
            print(
                self._fmt.format(
                    datetime=_now(),
                    pid="Main",
                    msg=f"Done transferring {filepath.name}",
                )
            )
            print(f"Done transferring {filepath.name}")

        if self.validate_created_path(event.src_path):
            filepath = event.src_path
            filepath = pathlib.Path(filepath)

            temp_filepath = self.temp_dir / f"{filepath.name}.tar"
            if temp_filepath.exists():
                print(
                    self._fmt.format(
                        datetime=_now(),
                        pid="Main",
                        msg=f"Cannot process {filepath} because {temp_filepath} already exists.",
                    )
                )
                return

            placeholder_filepath = filepath.parent / f"compressing-{filepath.name}"
            placeholder_filepath.touch()

            compressed_filepath = f"{self.out_dir / temp_filepath.name}.zst"

            cmd_tar = [
                str(PATH_7Z),
                "a",
                "-ttar",
                "-bb0",
                # "-bso1",
                # "-bse2",
                # "-bsp0",
                str(temp_filepath),
                str(filepath),
            ]
            cmd_zstd = [
                str(PATH_ZSTD),
                "--force",
                "--no-progress",
                "-T4",
                "-3",
                "-v",
                str(temp_filepath),
                "-o",
                str(compressed_filepath),
            ]
            cmd_delete = [
                str(PATH_PEA),
                "WIPE",
                "QUICK",
                str(temp_filepath),
            ]

            tar_process = await run(shlex.join(cmd_tar))
            print(
                self._fmt.format(
                    datetime=_now(),
                    pid=tar_process.pid,
                    msg=f"RETURNCODE {tar_process.returncode}",
                )
            )
            if tar_process.returncode == 0:
                print(
                    self._fmt.format(
                        datetime=_now(),
                        pid=tar_process.pid,
                        msg=f"Archived {filepath} to {temp_filepath}",
                    )
                )
            zstd_process = await run(shlex.join(cmd_zstd))
            print(
                self._fmt.format(
                    datetime=_now(),
                    pid=zstd_process.pid,
                    msg=f"RETURNCODE {zstd_process.returncode}",
                )
            )
            if zstd_process.returncode == 0:
                placeholder_filepath.unlink()
                print(
                    self._fmt.format(
                        datetime=_now(),
                        pid=zstd_process.pid,
                        msg=f"Compressed {filepath} to {compressed_filepath}",
                    )
                )
            delete_process = await run(shlex.join(cmd_delete))
            if delete_process.returncode == 0:
                print(
                    self._fmt.format(
                        datetime=_now(),
                        pid=delete_process.pid,
                        msg=f"Deleted {temp_filepath}",
                    )
                )

    async def on_deleted(self, event):
        pass

    async def on_moved(self, event):
        pass
        # print("Moved:", event.src_path)
        # print("Moved:", event.dest_path, "\n")

    async def on_modified(self, event):
        pass

    async def on_closed(self, event):
        pass

    async def on_opened(self, event):
        pass


async def watch_scan_folder(watch_dir, temp_dir, out_dir, recursive=True):
    evh = ScansFolderEventHandler(
        temp_dir=temp_dir,
        out_dir=out_dir,
    )
    watch = AIOWatchdog(watch_dir, event_handler=evh, recursive=recursive)
    watch.start()
    msg = f"""\
        Watching {watch_dir}
            temp_dir {temp_dir}
            out_dir {out_dir}
            recursive {recursive}
    """
    print(textwrap.dedent(msg))
    try:
        while True:
            await asyncio.sleep(1)
    except Exception:
        watch.stop()


FORMAT = "{datetime} | {pid:>10} | {msg}"

PEAZIP_PATH = pathlib.Path(r"C:\Users\Public\Downloads\peazip_portable-9.6.0.WIN64")
PATH_7Z = PEAZIP_PATH / "res" / "bin" / "7z" / "7z.exe"
PATH_ZSTD = PEAZIP_PATH / "res" / "bin" / "zstd" / "zstd.exe"
PATH_PEA = PEAZIP_PATH / "pea.exe"


DEFAULTS = {
    "watch_dir": r"C:\rcpnl\scans",
    "temp_dir": r"C:\rcpnl\_temp_compression",
    "out_dir": r"D:",
    "recursive": False,
}

asyncio.get_event_loop().run_until_complete(watch_scan_folder(**DEFAULTS))


"""
python -u "C:\rcpnl\_temp_compression\run-compression.py" | tee -a C:\rcpnl\_temp_compression\compress.log
"""

"""
conda activate quick-look
python -m pip install hachiko
mkdir C:\rcpnl\_temp_compression
touch "C:\rcpnl\_temp_compression\log_event.txt"

python -u "C:\rcpnl\_temp_compression\log_event.txt" | tee C:\rcpnl\_temp_compression\event.log
"""
