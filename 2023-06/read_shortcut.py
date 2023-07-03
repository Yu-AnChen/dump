import platform
import os
import pathlib


def get_path(path):
    system = platform.system()
    if system in ["Linux", "Darwin"]:
        path = os.path.realpath(path)
    elif system == "Windows":
        import win32com.client
        shell = win32com.client.Dispatch("WScript.Shell")
        path = shell.CreateShortCut(path).Targetpath
    else:
        print('Unknown os.')
        path = ''
    return pathlib.Path(path)
