import numpy as np
import transform
from ashlar import utils
import skimage.registration


def register(layer_aligner, t):
    """Return relative shift/angle between images and the alignment error."""
    its, ref_img, img = layer_aligner.overlap(t)
    if np.any(np.array(its.shape) == 0):
        return (0, 0), np.inf
    
    return detect_rotation_angle(ref_img, img, layer_aligner.filter_sigma)


def detect_rotation_angle(ref_img, img, sigma=0):
    window_y = np.hanning(ref_img.shape[0])[..., None]
    window_x = np.hanning(ref_img.shape[1])[..., None]
    window = window_y * window_x.T
    a = np.clip(
        transform.polar2cart(np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(ref_img*window))).real)*window),
        0,
        None
    )
    b = np.clip(
        transform.polar2cart(np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(img*window))).real)*window),
        0,
        None
    )
    (angle, _), _, _ = skimage.registration.phase_cross_correlation(
        utils.whiten(a, sigma)*window,
        utils.whiten(b, sigma)*window,
        upsample_factor=10,
        normalization=None
    )
    angle = angle / a.shape[0] * 360
    return angle