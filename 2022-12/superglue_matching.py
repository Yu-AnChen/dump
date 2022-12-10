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


def superglue_registration(img_left, img_right, device='cuda'):
    img_left = np.asarray(img_left)
    img_right = np.asarray(img_right)
    
    imgt_l = channel_to_input(img_left, device)
    imgt_r = channel_to_input(img_right, device)

    pred = matching({'image0': imgt_l, 'image1': imgt_r})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    ski_matches = np.where(matches > -1, np.vstack([np.arange(matches.size), matches]), -1)
    ski_matches = ski_matches.T[ski_matches[0] > -1]

    import cv2
    t_matrix, mask = cv2.estimateAffine2D(
        kpts1[ski_matches[:, 1]], 
        kpts0[ski_matches[:, 0]],
        method=cv2.RANSAC,
        ransacReprojThreshold=5,
        maxIters=5000
    )

    plt.figure()
    skimage.feature.plot_matches(
        plt.gca(),
        img_left,
        img_right,
        np.fliplr(kpts0),
        np.fliplr(kpts1),
        ski_matches[mask.flatten()>0]
    )

    # valid = matches > -1
    # mkpts0 = kpts0[valid]
    # mkpts1 = kpts1[matches[valid]]
    # mconf = conf[valid]

    return t_matrix


import skimage.io

path_1 = r"X:\crc-scans\histowiz scans\20221117\IF_HE\thumbnails\CRC_07_C0_1.png"
path_2 = r"X:\crc-scans\histowiz scans\20221117\IF_HE\thumbnails\CRC_07_C4_1.png"

img1 = skimage.io.imread(path_1)
img2 = skimage.io.imread(path_2)

superglue_registration(img1[..., 1], img2[..., 1])
