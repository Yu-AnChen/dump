# https://github.com/magicleap/SuperGluePretrainedNetwork

import torch
from models.matching import Matching
from models.utils import read_image


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

path_1 = r"X:\crc-scans\histowiz scans\20221117\IF_HE\thumbnails\CRC_07_C0_1.png"
path_2 = r"X:\crc-scans\histowiz scans\20221117\IF_HE\thumbnails\CRC_07_C4_1.png"

image0, inp0, scales0 = read_image(
    path_1, device, [-1], 0, False)
image1, inp1, scales1 = read_image(
    path_2, device, [-1], 0, False)

pred = matching({'image0': inp0, 'image1': inp1})
pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
matches, conf = pred['matches0'], pred['matching_scores0']

valid = matches > -1
mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches[valid]]
mconf = conf[valid]

import numpy as np
ski_matches = np.where(matches > -1, np.vstack([np.arange(matches.size), matches]), -1)
ski_matches = ski_matches.T[ski_matches[0] > -1]

import skimage.feature
import skimage.io
import matplotlib.pyplot as plt

plt.figure()
skimage.feature.plot_matches(
    plt.gca(),
    skimage.io.imread(path_1),
    skimage.io.imread(path_2),
    np.fliplr(pred['keypoints0']),
    np.fliplr(pred['keypoints1']),
    ski_matches
)

import cv2
t_matrix, mask = cv2.estimateAffine2D(
    np.fliplr(pred['keypoints0'])[ski_matches[:, 0]], np.fliplr(pred['keypoints1'])[ski_matches[:, 1]], 
    method=cv2.RANSAC,
    ransacReprojThreshold=20,
    maxIters=5000
)


plt.figure()
skimage.feature.plot_matches(
    plt.gca(),
    skimage.io.imread(path_1),
    skimage.io.imread(path_2),
    np.fliplr(pred['keypoints0']),
    np.fliplr(pred['keypoints1']),
    ski_matches[mask.flatten()>0]
)

print(t_matrix)