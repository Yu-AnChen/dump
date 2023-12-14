import json
import pathlib
import shutil
import subprocess

import pathviewer2minerva_roi as mroi

# ---------------------------------------------------------------------------- #
#                            1. get omero image IDs                            #
# ---------------------------------------------------------------------------- #

lsp_ids = [
    pp.name
    for pp in pathlib.Path(r"C:\Users\Public\Downloads\tmp-stic-minerva-all").glob("*")
]

omero_url = "https://idp.tissue-atlas.org"
session_token = "TOKEN"
GROUP_NAME = "Gray Data Portal"

session = mroi.omero_login(omero_url=omero_url, session_token=session_token)

r = session.get(f"{omero_url}/api/v0/m/experimentergroups/", params=dict(limit=500))
groups = {ii["Name"]: ii["@id"] for ii in r.json()["data"]}
group_id = groups[GROUP_NAME]

r = session.get(f"{omero_url}/api/v0/m/images/", params=dict(group=group_id, limit=500))
images = {ii["Name"]: ii["@id"] for ii in r.json()["data"]}

omero_ids = [images.get(f"{dd}.ome.tif", None) for dd in lsp_ids]

# ---------------------------------------------------------------------------- #
#                    2. query image rois and make waypoints                    #
# ---------------------------------------------------------------------------- #
img_waypoints = []

for ii in omero_ids[:]:
    wps = mroi.rois_to_waypoints(
        img_id=ii,
        session_token=session_token,
        waypoint_default={"Group": "Cycle 3", "Zoom": 4},
    )
    wps = list(filter(lambda x: " | " in x["Name"], wps))
    img_waypoints.append(wps)


for ww in img_waypoints:
    if not ww:
        continue
    for rr in ww:
        rr["Zoom"] = 4

# ---------------------------------------------------------------------------- #
#                     3. update index.html and exhibit.json                    #
# ---------------------------------------------------------------------------- #
minerva_dir = pathlib.Path(r"C:\Users\Public\Downloads\tmp-stic-minerva-all")

for ii, ww in zip(lsp_ids, img_waypoints):
    shutil.copy(
        r"U:\YC-20231204-STIC_minerva_rendering\pathviewer2minerva\index.html",
        minerva_dir / ii / "index.html",
    )

    with open(minerva_dir / ii / "exhibit.json") as f:
        exhibit = json.load(f)

    w, h = exhibit["Images"][0]["Width"], exhibit["Images"][0]["Height"]
    exhibit["Images"][0]["Description"] = f"BRCA-Mutant-Ovarian-Precursors ({ii})"
    exhibit["Stories"][0]["Waypoints"] = ww
    exhibit["FirstViewport"] = {"Zoom": 0.3, "Pan": [0.5 * w / h, 0.5]}
    if ww:
        exhibit["Header"] = "CyCIF image with ROIs mapped from adjacent GeoMx section"

    with open(minerva_dir / ii / "exhibit.json", "w") as f:
        json.dump(exhibit, f)

# ---------------------------------------------------------------------------- #
#                                4. upload to s3                               #
# ---------------------------------------------------------------------------- #
aws_cmd = "aws s3 cp {} s3://www.cycif.org/110-BRCA-Mutant-Ovarian-Precursors/{} --acl public-read --only-show-errors"

for ii in lsp_ids:
    cmd = aws_cmd.format(minerva_dir / ii / "index.html", f"{ii}/index.html")
    subprocess.run(cmd)
    cmd = aws_cmd.format(minerva_dir / ii / "exhibit.json", f"{ii}/exhibit.json")
    subprocess.run(cmd)
