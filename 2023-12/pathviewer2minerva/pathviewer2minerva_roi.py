import copy
import re

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import requests
import skimage.transform


def _parse_roi_points(all_points):
    return np.array(re.findall(r"-?\d+\.?\d+", all_points), dtype=float).reshape(-1, 2)


def _tform_mx(roi_transform):
    return [
        [roi_transform["A00"], roi_transform["A01"], roi_transform["A02"]],
        [roi_transform["A10"], roi_transform["A11"], roi_transform["A12"]],
        [0, 0, 1],
    ]


def _tform(roi):
    if "Transform" not in roi:
        mx = np.eye(3)
    else:
        mx = _tform_mx(roi["Transform"])
    return skimage.transform.AffineTransform(matrix=np.array(mx))


def _centroid(vertices):
    x, y = 0, 0
    n = len(vertices)
    signed_area = 0
    for i in range(len(vertices)):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        # shoelace formula
        area = (x0 * y1) - (x1 * y0)
        signed_area += area
        x += (x0 + x1) * area
        y += (y0 + y1) * area
    signed_area *= 0.5
    x /= 6 * signed_area
    y /= 6 * signed_area
    return x, y


def roi_to_patch(roi):
    supported_types = ["Ellipse", "Point", "Rectangle", "Line", "Polyline", "Polygon"]
    if roi["@type"] not in [
        f"http://www.openmicroscopy.org/Schemas/OME/2016-06#{tt}"
        for tt in supported_types
    ]:
        print(f"ROI type ({roi['@type']}) is currently not supported")
        return
    roi_type = roi["@type"].replace(
        "http://www.openmicroscopy.org/Schemas/OME/2016-06#", ""
    )

    func = globals().get(f"{roi_type.lower()}_roi_to_patch")
    return func(roi)


def ellipse_roi_to_patch(roi):
    tform = _tform(roi)
    return matplotlib.patches.Ellipse(
        tform([(roi["X"], roi["Y"])])[0],
        width=2 * roi["RadiusX"],
        height=2 * roi["RadiusY"],
        angle=np.rad2deg(tform.rotation),
        edgecolor="salmon",
        facecolor="none",
    )


def point_roi_to_patch(roi):
    tform = _tform(roi)
    return matplotlib.patches.Polygon(xy=tform([(roi["X"], roi["Y"])])[0], closed=False)


def line_roi_to_patch(roi):
    tform = _tform(roi)
    return matplotlib.patches.Polygon(
        xy=tform([(roi["X1"], roi["Y1"]), (roi["X2"], roi["Y2"])]),
        closed=False,
    )


def polyline_roi_to_patch(roi):
    tform = _tform(roi)
    points = _parse_roi_points(roi["Points"])
    return matplotlib.patches.Polygon(
        xy=tform(points),
        closed=False,
    )


def polygon_roi_to_patch(roi):
    tform = _tform(roi)
    points = _parse_roi_points(roi["Points"])
    return matplotlib.patches.Polygon(
        xy=tform(points),
        closed=True,
    )


def rectangle_roi_to_patch(roi):
    tform = _tform(roi)
    x, y = roi["X"], roi["Y"]
    w, h = roi["Width"], roi["Height"]
    points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return matplotlib.patches.Polygon(
        xy=tform(points),
        closed=True,
    )


def plot_rois(roi_json, canvas_shape=None, img=None):
    _rois = roi_json["data"]

    rois = []
    roi_names = []
    shape_texts = []
    for rr in _rois:
        rois.extend(rr["shapes"])
        roi_names.extend([rr["Name"]] * len(rr["shapes"]))
        shape_texts.extend([ss["Text"] for ss in rr["shapes"]])
    patches = [roi_to_patch(rr) for rr in rois]
    labels = [
        ", ".join(set([nn, tt]) - set([""])) for nn, tt in zip(roi_names, shape_texts)
    ]
    centroids = [_centroid(pp.get_verts()) if pp else None for pp in patches]
    if canvas_shape is None:
        xmax, ymax = np.max(
            [pp.get_verts().max(axis=0) for pp in patches if pp], axis=0
        )
        xmin, ymin = np.min(
            [pp.get_verts().min(axis=0) for pp in patches if pp], axis=0
        )
    else:
        xmin, ymin = 0, 0
        xmax, ymax = canvas_shape

    fig, ax = plt.subplots()
    if img is None:
        ax.imshow([[0]], alpha=0)
    else:
        ax.imshow(img, extent=(xmin, xmax, ymax, ymin))
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(top=ymin, bottom=ymax)
    for pp, ll, cc in zip(patches, labels, centroids):
        if pp is not None:
            pp = copy.copy(pp)
            pp.set_facecolor("none")
            pp.set_edgecolor("gold")
            ax.add_patch(pp)
            ax.text(*cc, s=ll, va="center", ha="center")
    return fig


def omero_login(omero_url, session_token, server_name="omero"):
    if omero_url[-1] == "/":
        omero_url.pop(-1)

    OMERO_WEB_HOST = omero_url
    SERVER_NAME = server_name
    SESSION_TOKEN = session_token

    session = requests.Session()

    api_url = f"{OMERO_WEB_HOST}/api/"
    print("Starting at:", api_url)
    r = session.get(api_url)

    versions = r.json()["data"]
    version = versions[-1]
    base_url = version["url:base"]

    r = session.get(base_url)
    urls = r.json()

    # To login we need to get CSRF token
    token = session.get(urls["url:token"]).json()["data"]

    # Setting the referer is required
    session.headers.update({"Referer": urls["url:login"]})

    # List the servers available to connect to
    servers = session.get(urls["url:servers"]).json()["data"]
    print("Servers:")
    for s in servers:
        print("-id:", s["id"])
        print(" name:", s["server"])
        print(" host:", s["host"])
        print(" port:", s["port"])
    # find one called SERVER_NAME
    servers = [s for s in servers if s["server"] == SERVER_NAME]
    if len(servers) < 1:
        raise Exception("Found no server called '%s'" % SERVER_NAME)
    server = servers[0]

    # Login with username, password and token
    payload = {
        "username": SESSION_TOKEN,
        "password": SESSION_TOKEN,
        "csrfmiddlewaretoken": token,
        "server": server["id"],
    }

    r = session.post(urls["url:login"], data=payload)
    login_rsp = r.json()
    assert r.status_code == 200
    assert login_rsp["success"]

    return session


def rois_to_waypoints(
    img_id,
    session_token,
    omero_url="https://idp.tissue-atlas.org",
    waypoint_default=None,
):
    import lzstring

    lzstr = lzstring.LZString()

    session = omero_login(omero_url=omero_url, session_token=session_token)
    r = session.get(f"{omero_url}/api/v0/m/images/{img_id}")
    img_metadata = r.json()["data"]
    height = img_metadata["Pixels"]["SizeY"]

    r = session.get(f"{omero_url}/api/v0/m/images/{img_id}/rois/?limit=500")
    _rois = r.json()["data"]
    rois = []
    roi_names = []
    shape_texts = []
    for rr in _rois:
        rois.extend(rr["shapes"])
        roi_names.extend([rr["Name"]] * len(rr["shapes"]))
        shape_texts.extend([ss["Text"] for ss in rr["shapes"]])

    labels = [
        ", ".join(set([nn, tt]) - set([""])) for nn, tt in zip(roi_names, shape_texts)
    ]
    patches = [roi_to_patch(rr) for rr in rois]
    centroids = [_centroid(pp.get_verts()) if pp else None for pp in patches]

    coordinate_strs = [
        lzstr.compressToEncodedURIComponent(
            ",".join([f"{dd:.5f}" for dd in pp.get_verts().flatten() / height])
        )
        if pp
        else None
        for pp in patches
    ]

    if waypoint_default is None:
        waypoint_default = {}
    waypoint = {
        "Name": "",
        "Description": "",
        "Zoom": 1,
        "Pan": [],
        "Polygon": "",
    }

    waypoints = []
    for ll, pp, cc in zip(labels, coordinate_strs, centroids):
        if pp is None:
            continue
        ww = {
            "Name": ll,
            "Pan": [cc[0] / height, cc[1] / height],
            "Polygon": pp,
        }
        waypoints.append({**waypoint, **waypoint_default, **ww})
    return waypoints


def test():
    wps = rois_to_waypoints(
        img_id=6561,
        session_token="TOKEN",
        waypoint_default={"Group": "Cycle 3", "Zoom": 4},
    )


def dev():
    omero_url = "https://idp.tissue-atlas.org"
    image_id = 6561
    session = omero_login(
        omero_url=omero_url, session_token="TOKEN"
    )

    r = session.get(f"{omero_url}/api/v0/m/images/{image_id}/rois/?limit=500")
    _rois = r.json()["data"]
    rois = []
    roi_names = []
    shape_texts = []
    for rr in _rois:
        rois.extend(rr["shapes"])
        roi_names.extend([rr["Name"]] * len(rr["shapes"]))
        shape_texts.extend([ss["Text"] for ss in rr["shapes"]])

    labels = [
        ", ".join(set([nn, tt]) - set([""])) for nn, tt in zip(roi_names, shape_texts)
    ]
    patches = [roi_to_patch(rr) for rr in rois]
    centroids = [_centroid(pp.get_verts()) if pp else None for pp in patches]
