"""Fail if a SHARED block differs between the push and pull scripts (ADR-023).

QuPath scripts cannot import each other, so the code both need (config, session
cache, roitool/JRE provisioning, roitool runner, URL parsing, login) is duplicated
between `// ==== SHARED BEGIN <name>` / `// ==== SHARED END <name> ====` markers.
Edit one copy, paste it into the other, then run:  python check_shared.py
"""
import difflib
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
SCRIPTS = ("send_rois_to_omero.groovy", "pull_rois_from_omero.groovy")
BLOCK = re.compile(r"^// ==== SHARED BEGIN (\S+).*?\n(.*?)^// ==== SHARED END \1 ====$", re.M | re.S)

push, pull = (dict(BLOCK.findall((HERE / f).read_text(encoding="utf-8").replace("\r\n", "\n")))
              for f in SCRIPTS)
ok = True
for name in sorted(push.keys() | pull.keys()):
    if name not in push or name not in pull:
        print(f"block '{name}' is missing from {SCRIPTS[name in push]}")
        ok = False
    elif push[name] != pull[name]:
        print(f"block '{name}' differs:")
        sys.stdout.writelines(difflib.unified_diff(
            push[name].splitlines(True), pull[name].splitlines(True), *SCRIPTS))
        ok = False
print(f"{len(push)} shared blocks in sync" if ok else "shared blocks out of sync")
sys.exit(0 if ok else 1)
