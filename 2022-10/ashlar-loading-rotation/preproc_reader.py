from ashlar import reg
import numpy as np
import skimage.transform


class HeBioformatsReader(reg.BioformatsReader):
    def __init__(
        self, path, plate=None, well=None,
        flip_x=False, flip_y=False, angle=0,
        center_crop_shape=None
    ):
        super().__init__(path, plate=plate, well=well)
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.angle = angle
        self.center_crop_shape = center_crop_shape

        _ = self.metadata.positions
        self.metadata._positions *= [-1, 1]

        self.offsets = [0, 0]
        if self.center_crop_shape is not None:
            offsets = 0.5 * (self.metadata.size - self.center_crop_shape)
            self.offsets = [int(o) for o in offsets]
            self.metadata._size = np.array(self.center_crop_shape)
        self.metadata._positions += self.offsets
    
    def read(self, series, c):
        img = super().read(series=series, c=c)
        if not np.all(np.array(self.offsets) == 0):
            o_r, o_c = [int(o) for o in self.offsets]
            img = img[o_r:-o_r, o_c:-o_c]
        if self.flip_x:
            img = np.fliplr(img)
        if self.flip_y:
            img = np.flipud(img)
        if self.angle != 0:
            img = skimage.transform.rotate(img, self.angle)
        return img
