# https://github.com/ome/openmicroscopy/blob/develop/examples/Training/python/Json_Api/Login.py

import requests
import json
import pathlib


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


def get_img_metadata(omero_url, img_id, session):
    if omero_url[-1] == "/":
        omero_url.pop(-1)

    r = session.get(
        f"{omero_url}/pathviewer/imgData/{img_id}/?callback=angular.callbacks._0"
    )
    if r.status_code != 200:
        print(f"Failed getting metadata for {img_id} from {omero_url}")
        print(f"{r.status_code}: {r.reason}")
        return None
    metadata = json.loads(r.text.replace("angular.callbacks._0(", "")[:-1])
    return metadata


def get_channel_groups(omero_url, img_id, session, annotation_idx=0):
    if omero_url[-1] == "/":
        omero_url.pop(-1)

    r = session.get(f"{omero_url}/pathviewer/viewer/{img_id}/multichannelgroups")
    annotations = r.json()["annotations"]
    if len(annotations) == 0:
        print(f"No channel groups found for image {img_id}")
        return None
    if len(annotations) == 1:
        return json.loads(annotations[0]["textValue"])["groups"]

    experimenters = {
        ee["id"]: f"{ee['lastName']}, {ee['firstName']}"
        for ee in r.json()["experimenters"]
    }
    print(
        f"{len(annotations)} channel groups found from image {img_id}. Annotation"
        f" <{annotation_idx}> is selected (`annotation_idx={annotation_idx}`)"
    )
    for ii, aa in enumerate(annotations):
        num_groups = len(json.loads(aa["textValue"])["groups"])
        print(f"Annotation <{ii}>:")
        print(
            f"  Name: {aa['description']}; Owner: {experimenters[aa['owner']['id']]}; # of groups: {num_groups}"
        )
    return json.loads(annotations[annotation_idx]["textValue"])["groups"]


def make_minerva_groups(groups, channel_names, pixel_range):
    pmin, pmax = pixel_range
    prange = pmax - pmin
    minerva_groups = []

    for group in groups:
        minerva_group = {}
        minerva_group["label"] = group["name"]

        channels = []
        for kk, vv in group["channelSettings"].items():
            kk = int(kk)
            channel = {}
            channel["id"] = kk
            channel["label"] = channel_names[kk]
            channel["color"] = vv["color"].lower().replace("#", "")
            channel["min"] = (vv["window"]["start"] - pmin) / prange
            channel["max"] = (vv["window"]["end"] - pmin) / prange

            channels.append(channel)

        minerva_group["channels"] = channels
        minerva_group["render"] = channels

        minerva_groups.append(minerva_group)
    return minerva_groups


def make_story_json_for_rendering(groups=None):
    minimal_json = {
        "waypoints": [],
        "sample_info": {"name": "", "text": "", "rotation": 0},
        "groups": [],
    }
    if groups is not None:
        minimal_json.update({"groups": groups})
    return json.dumps(minimal_json)


def pathviewer_channel_groups_to_story_json(
    omero_url, img_id, session_token, out_path, filter_unnamed=True, annotation_idx=0
):
    session = omero_login(omero_url, session_token)
    metadata = get_img_metadata(omero_url, img_id, session)
    groups = get_channel_groups(omero_url, img_id, session, annotation_idx)

    if filter_unnamed:
        groups = list(filter(lambda x: x["name"] != "\x00", groups))

    channel_names = [cc["label"] for cc in metadata["channels"]]
    px_range = metadata["pixel_range"]
    minerva_groups = make_minerva_groups(groups, channel_names, px_range)
    story_json = make_story_json_for_rendering(groups=minerva_groups)

    if out_path is not None:
        out_path = pathlib.Path(out_path)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        print(f"Writing to {out_path}")
        with open(out_path, "w") as f:
            f.write(story_json)
    return story_json


ooo = pathviewer_channel_groups_to_story_json(
    omero_url="https://idp.tissue-atlas.org",
    img_id=6410,
    session_token="TOKEN",
    out_path=r"U:\YC-20231204-STIC_minerva_rendering\stic_b2.json",
    annotation_idx=1,
)
