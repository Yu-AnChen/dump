import pathlib

import numpy as np
import skimage.io
from ashlar import reg, utils, thumbnail


class TestMetadata(reg.Metadata):

    def __init__(self, path, tile_size, overlap, pixel_size, channel=0, zarr=None, img=None, series=None):
        self.path = pathlib.Path(path)
        self._tile_size = np.array(tile_size)
        self.overlap = overlap
        self._pixel_size = pixel_size
        self.channel = channel
        self.zarr = zarr
        self.img = img
        self.series = series
        self.deconstruct_mosaic()

    def deconstruct_mosaic(self):

        if self.zarr is not None:
            self.mosaic = self.zarr

        if self.img is not None:
            self.mosaic = self.img

        if self.zarr is None and self.img is None:
            self.mosaic = skimage.io.imread(self.path, key=self.channel)
        
        m_shape = self.mosaic.shape
        
        step_shape = (1 - self.overlap) * self._tile_size
        # round position to integer since no subpixel needed for already stitched image
        step_shape = np.around(step_shape).astype('int')

        self._slice_positions = np.mgrid[
            :m_shape[0]:step_shape[0], :m_shape[1]:step_shape[1]
        ].reshape(2, -1).T

        self._positions = self._slice_positions.astype(float)

        if self.series is not None:
            self._slice_positions = self._slice_positions[self.series]
            self._positions = self._positions[self.series]

    @property
    def _num_images(self):
        return len(self._positions)

    @property
    def num_channels(self):
        return 1

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def pixel_dtype(self):
        return self.zarr.dtype

    @property
    def mosaic_shape(self):
        return self.zarr.shape

    def tile_size(self, i):
        return self._tile_size


class TestReader(reg.Reader):

    def __init__(
            self,
            path=None,
            tile_size=(1000, 1000),
            overlap=0.1,
            pixel_size=1,
            channel=0,
            zarr=None,
            img=None,
            series=None,
            flip_x=False,
            flip_y=False,
            angle=0,
            center_crop_shape=None
        ):
        path = '' if path is None else path 
        self.metadata = TestMetadata(
            path, tile_size, overlap, pixel_size, channel, zarr, img, series
        )
        self.path = pathlib.Path(path)
        self.mosaic = self.metadata.mosaic
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.angle = angle

    def read(self, series, c):
        position = self.metadata._slice_positions[series]
        assert np.issubdtype(position.dtype, np.integer)
        r, c = position
        h, w = self.metadata._tile_size
        img = self.mosaic[r:r+h, c:c+w]
        if not np.all(img.shape == (h, w)):
            img_h, img_w = img.shape
            pad_h, pad_w = np.clip([h-img_h, w-img_w], 0, None)
            img = np.pad(img, [(0, pad_h), (0, pad_w)])
        if self.flip_x:
            img = np.fliplr(img)
        if self.flip_y:
            img = np.flipud(img)
        if self.angle != 0:
            img = skimage.transform.rotate(img, self.angle, center=(0, 0), resize=True)
        return img
 

def align_cycles(reader1, reader2, scale=0.05):
    import skimage.transform
    if not hasattr(reader1, 'thumbnail'):
        raise ValueError('reader1 does not have a thumbnail')
    if not hasattr(reader2, 'thumbnail'):
        raise ValueError('reader2 does not have a thumbnail')
    img1 = reader1.thumbnail
    img2 = reader2.thumbnail
    if img1.shape != img2.shape:
        padded_shape = np.array((img1.shape, img2.shape)).max(axis=0)
        padded_img1, padded_img2 = np.zeros(padded_shape), np.zeros(padded_shape)
        utils.paste(padded_img1, img1, [0, 0], 0)
        utils.paste(padded_img2, img2, [0, 0], 0)
        img1 = padded_img1
        img2 = padded_img2
    angle = utils.register_angle(img1, img2, sigma=1)
    if angle != 0:
        print(f'\r    estimated cycle rotation = {angle:.2f} degrees')
        img2 = skimage.transform.rotate(img2, angle, resize=False, center=(0, 0))
    shifts = thumbnail.calculate_image_offset(img1, img2, int(1 / scale))
    print(f'\r    estimated shift {shifts / scale}')
    tform_steps = [
        ('translation', -reader2.metadata.origin[::-1]),
        ('scale', scale),
        ('rotation', np.deg2rad(-angle)),
        ('translation', shifts[::-1]),
        ('scale', 1/scale),
        ('translation', reader1.metadata.origin[::-1])
    ]
    tform = skimage.transform.AffineTransform()
    for step in tform_steps:
        tform += skimage.transform.AffineTransform(
            **{step[0]: step[1]}
        )

    return tform


import numpy as np
import skimage.data
import skimage.transform
from ashlar import thumbnail

TILE_SIZE = (108, 128)

img = skimage.data.astronaut()[..., 1]
c1r = TestReader(img=img, tile_size=TILE_SIZE)

affine = skimage.transform.AffineTransform
tform = affine(
    translation=200*(np.random.random(2)-.5),
    rotation=np.deg2rad(10*np.random.random(1)[0])
)

# apply known transform to image 
img2 = skimage.transform.warp(img, tform.inverse)
c2r = TestReader(img=img2, tile_size=TILE_SIZE)

# set random stage origin
c1r.metadata._positions += 2000*(np.random.random(2)-.5)
c2r.metadata._positions += 2000*(np.random.random(2)-.5)

c1r.thumbnail = thumbnail.make_thumbnail(c1r, scale=.5)
c2r.thumbnail = thumbnail.make_thumbnail(c2r, scale=.5)

thumbnail.align_cycles(c1r, c2r, scale=0.5)
cycle_tform = align_cycles(c1r, c2r, scale=0.5)


c2rr = TestReader(
    img=img2, tile_size=TILE_SIZE,
    angle=-np.rad2deg(cycle_tform.rotation)
)
c2rr.metadata._positions = (
    np.fliplr(cycle_tform(np.fliplr(c2r.metadata.positions)))
)

# compute and add offsets since we use (0, 0) for rotation with resizing
h, w = c2rr.metadata._tile_size
rotation_offset = skimage.transform.AffineTransform(
    rotation=cycle_tform.rotation
)([(0, 0), (w, 0), (0, h)]).min(axis=0)[::-1]
c2rr.metadata._positions += rotation_offset


import napari

v = napari.Viewer()
v.add_image(
    thumbnail.make_thumbnail(c1r, scale=1),
    translate=c1r.metadata.origin,
    colormap='plasma'
)

v.add_points(
    np.fliplr(cycle_tform(np.fliplr(c2r.metadata.positions)))
)

for idx, pos in enumerate(c2rr.metadata.positions):
    v.add_image(c2rr.read(idx, 0), translate=pos, blending='additive', colormap='cividis')
