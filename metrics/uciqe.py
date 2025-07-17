import torch
import numpy as np
from skimage.color import rgb2lab

def batch_uciqe(batch):
    B = batch.shape[0]
    total = 0.0
    for i in range(B):
        img = batch[i].cpu().permute(1, 2, 0).numpy()
        total += compute_uciqe(img)
    return total / B

def compute_uciqe(img):
    img = np.clip(img, 0, 1)
    lab = rgb2lab(img)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    c = np.sqrt(a**2 + b**2)
    chroma_std = np.std(c)
    s = img.std(axis=2)
    sat_mean = np.mean(s)
    L_sorted = np.sort(L.flatten())
    contrast = L_sorted[int(0.99 * len(L_sorted))] - L_sorted[int(0.01 * len(L_sorted))]
    return 0.4680 * chroma_std + 0.2745 * contrast + 0.2576 * sat_mean
