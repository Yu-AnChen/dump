# ezomero branch
# https://github.com/Yu-AnChen/ezomero/commit/9a2ab53d673e718cdf5a9dbae6ca37e9a335c815

import dataclasses
import logging
from functools import cached_property

import ezomero
import numpy as np
import palom
import tqdm
from omero.gateway import BlitzGateway


class ImageHandler:
    def __init__(self, conn: BlitzGateway, image_id: int):
        self.conn = conn
        self.image_id = image_id
        self.image_object = self.fetch_image_object()
        self.image_group_id = self.image_object.getDetails().getGroup().getId()

    def fetch_image_object(self):
        self.conn.SERVICE_OPTS.setOmeroGroup("-1")
        # Load the image object once
        return self.conn.getObject("Image", self.image_id)

    @cached_property
    def num_channels(self):
        return len(self.image_object.getChannels())

    @property
    def num_pyramid_levels(self):
        return len(self.pyramid_config["level_shapes"])

    @property
    def pixel_size(self):
        return self.pyramid_config["level_pixel_sizes"][0]

    @cached_property
    def pyramid_config(self):
        set_session_group(self.conn, self.image_group_id)
        image = self.image_object

        pix = image._conn.c.sf.createRawPixelsStore()
        pid = image.getPixelsId()
        pix.setPixelsId(pid, False)
        level_shapes = [(rr.sizeY, rr.sizeX) for rr in pix.getResolutionDescriptions()]
        pix.close()

        _physical_size_x = image.getPrimaryPixels().getPhysicalSizeX()
        pixel_size = _physical_size_x.getValue()
        pixel_size_unit = _physical_size_x.getUnit()
        if pixel_size_unit.name != "MICROMETER":
            logging.warning(f"target image's pixel size unit is {pixel_size_unit}")

        level_downsamples = {
            kk: np.round(1 / vv).astype("int")
            for kk, vv in image.getZoomLevelScaling().items()
        }
        level_pixel_sizes = [
            pixel_size * vv for _, vv in sorted(level_downsamples.items())
        ]
        return {
            "level_downsamples": [vv for _, vv in sorted(level_downsamples.items())],
            "level_pixel_sizes": level_pixel_sizes,
            "level_shapes": level_shapes,
        }

    def fetch_image_level_channel(self, level: int, channel: int):
        set_session_group(self.conn, self.image_group_id)

        _, img = ezomero.get_image(
            self.conn,
            self.image_id,
            pyramid_level=range(self.num_pyramid_levels)[level],
            start_coords=(0, 0, 0, channel, 0),
            axis_lengths=(*self.pyramid_config["level_shapes"][level][::-1], 1, 1, 1),
        )
        return img.squeeze()

    def select_pyramid_level_by_pixel_size(self, max_pixel_size: float):
        level_downsamples = self.pyramid_config["level_downsamples"]
        level_pixel_sizes = self.pyramid_config["level_pixel_sizes"]
        return np.max(
            np.arange(len(level_downsamples))[
                np.less_equal(level_pixel_sizes, max_pixel_size)
            ]
        )

    def channel_to_palom_reader(
        self, channel: int, max_pixel_size: float
    ) -> palom.reader.DaPyramidChannelReader:
        import dask.array as da
        import palom

        level = self.select_pyramid_level_by_pixel_size(max_pixel_size)
        img = self.fetch_image_level_channel(level, channel)

        _pyramid = [
            da.zeros((1, *ss), chunks=1024, dtype=img.dtype)
            for ss in self.pyramid_config["level_shapes"]
        ]
        _pyramid[level] = img.reshape(1, *img.shape)

        levels = sorted(set([0, level]))
        pyramid = [_pyramid[ll] for ll in levels]

        tip_pixel_size = self.pyramid_config["level_pixel_sizes"][levels[-1]]
        while tip_pixel_size < max_pixel_size / 2:
            tip_pixel_size *= 2
            img = pyramid[-1][0]
            pyramid.append(palom.img_util.cv2_downscale_local_mean(img, 2)[np.newaxis])

        reader = palom.reader.DaPyramidChannelReader(pyramid, channel_axis=0)

        reader.pixel_size = self.pixel_size
        return reader


def set_group_by_image(conn: BlitzGateway, image_id: int) -> bool:
    conn.SERVICE_OPTS.setOmeroGroup("-1")
    image = conn.getObject("Image", image_id)
    if image is None:
        logging.warning(
            f"Cannot load image {image_id} - check if you have permissions to do so"
        )
        return False
    group_id = image.getDetails().getGroup().getId()
    return set_session_group(conn, group_id)


def set_session_group(conn: BlitzGateway, group_id: int) -> bool:
    current_id = conn.getGroupFromContext().getId()
    if group_id == current_id:
        return True
    if ezomero.set_group(conn, group_id):
        conn.setGroupForSession(group_id)
        return True
    return False


def _tform_mx(transform: ezomero.rois.AffineTransform) -> np.ndarray:
    return np.array(
        [
            [transform.a00, transform.a01, transform.a02],
            [transform.a10, transform.a11, transform.a12],
            [0, 0, 1],
        ]
    )


def map_roi(
    conn: BlitzGateway, from_image_id: int, to_image_id: int, affine_mx: np.ndarray
) -> list[ezomero.rois.ezShape]:
    if not conn.keepAlive():
        logging.error(f"Connection to {conn.host} lost")
        return
    set_group_by_image(conn, from_image_id)
    rois = ezomero.get_roi_ids(conn, from_image_id)
    shapes = [
        ezomero.get_shape(conn, ss)
        for rr in tqdm.tqdm(rois, "Downloading ROI")
        for ss in ezomero.get_shape_ids(conn, rr)
    ]
    transforms_ori = [ss.transform for ss in shapes]
    mxs = [_tform_mx(tt) if tt else np.eye(3) for tt in transforms_ori]
    transforms = [
        ezomero.rois.AffineTransform(*(affine_mx @ mm)[:2].T.ravel()) for mm in mxs
    ]
    shapes_transformed = [
        dataclasses.replace(ss, transform=tt) for ss, tt in zip(shapes, transforms)
    ]
    set_group_by_image(conn, to_image_id)
    for st in tqdm.tqdm(shapes_transformed, "Uploading ROI"):
        ezomero.post_roi(conn, to_image_id, [st])
    return shapes_transformed


def test():
    KEY = "f46069ad-6ad5-49ba-9348-6ce20d380ea9"
    HOST = "omero-app.hms.harvard.edu"
    # URL = "https://omero.hms.harvard.edu"

    conn = ezomero.connect(KEY, KEY, host=HOST, port=4064, secure=True, group="")

    ID_HE = 1277802
    ID_IF = 1619769

    r1 = ImageHandler(conn, ID_IF).channel_to_palom_reader(0, 50)
    r2 = ImageHandler(conn, ID_HE).channel_to_palom_reader(1, 50)

    # ID_HE = 1619769
    # ID_IF = 1281795

    # r1 = ImageHandler(conn, ID_IF).channel_to_palom_reader(0, 50)
    # r2 = ImageHandler(conn, ID_HE).channel_to_palom_reader(0, 50)

    # ID_HE = 1277802
    # ID_IF = 1281795

    # r1 = ImageHandler(conn, ID_IF).channel_to_palom_reader(0, 50)
    # r2 = ImageHandler(conn, ID_HE).channel_to_palom_reader(1, 50)

    c21l = palom.align.get_aligner(r1, r2)
    c21l.coarse_register_affine(
        test_intensity_invert=True, auto_mask=True, n_keypoints=10_000, test_flip=True
    )

    map_roi(conn, ID_HE, ID_IF, c21l.affine_matrix)
