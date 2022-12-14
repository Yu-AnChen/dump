import torch
from models.matching import Matching


torch.set_grad_enabled(False)

nms_radius = 4
keypoint_threshold = 0.005
max_keypoints = 5000
superglue = 'indoor'
sinkhorn_iterations = 20
match_threshold = 0.2
config = {
    'superpoint': {
        'nms_radius': nms_radius,
        'keypoint_threshold': keypoint_threshold,
        'max_keypoints': max_keypoints
    },
    'superglue': {
        'weights': superglue,
        'sinkhorn_iterations': sinkhorn_iterations,
        'match_threshold': match_threshold,
    }
}
device = 'cuda'
matching = Matching(config).eval().to(device)


import skimage.util
import skimage.feature
import numpy as np
import matplotlib.pyplot as plt


def channel_to_input(img, device):
    assert img.ndim == 2
    img = skimage.util.img_as_float32(img)
    return torch.from_numpy(img)[None, None].to(device)


def img_side_by_side(img_left, img_right):
    assert img_left.ndim == img_right.ndim
    img_left, img_right = np.atleast_3d(img_left, img_right)
    assert img_left.shape[2] in [1, 3, 4]

    height = max(img_left.shape[0], img_right.shape[0])
    width = img_left.shape[1] + img_right.shape[1]
    depth = img_left.shape[2]

    out = np.ones((height, width, depth)) * np.nan
    out[:img_left.shape[0], :img_left.shape[1]] = img_left
    out[:img_right.shape[0], img_left.shape[1]:] = img_right

    return out


import matplotlib.cm
def plot_match(matching_pred, img_left, img_right, match_mask=None):
    pred = matching_pred
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    if match_mask is None:
        match_mask = matches > -1
    # else: match_mask = pred['match_mask']

    img = img_side_by_side(
        skimage.util.img_as_float(img_left),
        skimage.util.img_as_float(img_right)
    )
    offset = img_left.shape[1]
    fig, ax = plt.subplots()
    ax.imshow(img)
    scatter_kwargs = dict(linewidths=0, facecolors='lime', s=4)
    ax.scatter(*kpts0.T, **scatter_kwargs)
    ax.scatter(*(kpts1 + [offset, 0]).T, **scatter_kwargs)

    m0x, m0y = kpts0[match_mask].T
    m1x, m1y = kpts1[matches[match_mask]].T
    m1x += offset
    # figure(); plot(np.arange(4).reshape(2, 2).T, np.arange(4).reshape(2, 2).T)
    line_kwargs = dict(linewidth=1)
    for x0, x1, y0, y1, cc in zip(m0x, m1x, m0y, m1y, conf[match_mask]):
        ax.plot([x0, x1], [y0, y1], c=matplotlib.cm.inferno(cc), **line_kwargs)
    return fig


def superglue_match(img_left, img_right, device='cuda'):
    img_left = np.asarray(img_left)
    img_right = np.asarray(img_right)
    
    imgt_l = channel_to_input(img_left, device)
    imgt_r = channel_to_input(img_right, device)

    pred = matching({'image0': imgt_l, 'image1': imgt_r})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    return pred


def get_flipping_mx(img_shape, flip_axis):
    assert flip_axis in [0, 1, (0, 1), (1, 0)]
    mx = np.eye(3)
    offset_xy = np.array(img_shape)[::-1] - 1
    if type(flip_axis) == int:
        index = int(not flip_axis)
        mx[index, index] = -1
        mx[index, 2] = offset_xy[index]
        return mx
    mx[:2, :2] *= -1
    mx[:2, 2] = offset_xy
    return mx


import functools
def match_test_flip(img_left, img_right, device='cuda'):
    funcs = [np.array]
    mxs = [np.eye(3)]
    funcs.extend([
        functools.partial(np.flip, axis=aa)
        for aa in (0, 1, (0, 1))
    ])
    mxs.extend([
        get_flipping_mx(img_right.shape, aa)
        for aa in (0, 1, (0, 1))
    ])
    preds = [
        superglue_match(img_left, ff(img_right), device=device)
        for ff in funcs
    ]
    best_pred = np.argmax([np.sum(pp['matches0'] > -1) for pp in preds])
    pred = preds[best_pred]
    return pred, mxs[best_pred]


import cv2
def estimate_affine(pred):
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    initial_mask = matches > -1
    initial_pairs = np.vstack([
        np.arange(matches.size), matches
    ]).T[initial_mask]

    t_matrix, affine_mask = cv2.estimateAffine2D(
        kpts1[initial_pairs[:, 1]], 
        kpts0[initial_pairs[:, 0]],
        method=cv2.RANSAC,
        ransacReprojThreshold=5,
        maxIters=5000
    )
    affine_mask = affine_mask.astype(bool).flatten()
    summary_mask = initial_mask.copy()
    summary_mask[initial_mask] *= affine_mask
    return t_matrix, summary_mask


import skimage.transform
def superglue_registration(img_left, img_right, device='cuda', test_flip=True):
    if test_flip:
        pred, mx_flip = match_test_flip(img_left, img_right, device)
    else:
        pred, mx_flip = superglue_match(img_left, img_right, device), np.eye(3)

    t_matrix, mask = estimate_affine(pred)
    t_matrix = (np.vstack([t_matrix, [0, 0, 1]]) @ mx_flip)[:2, :]

    tform = skimage.transform.AffineTransform(matrix=mx_flip)
    pred['keypoints1'] = tform(pred['keypoints1'])

    plot_match(pred, img_left, img_right, mask)





import skimage.io

path_1 = r"X:\crc-scans\histowiz scans\20221117\IF_HE\thumbnails\CRC_07_C0_1.png"
path_2 = r"X:\crc-scans\histowiz scans\20221117\IF_HE\thumbnails\CRC_07_C4_1.png"

img1 = skimage.io.imread(path_1)
img2 = skimage.io.imread(path_2)

superglue_registration(img1[..., 0], img2[..., 0])
superglue_registration(img1[..., 0], np.rot90(img2[..., 0]), test_flip=True)
