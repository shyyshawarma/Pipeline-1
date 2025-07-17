import torch
from skimage.color import rgb2lab, deltaE_ciede2000

def batch_ciede2000(batch1, batch2):
    B = batch1.shape[0]
    total = 0.0
    for i in range(B):
        img1 = batch1[i].cpu().permute(1, 2, 0).numpy()
        img2 = batch2[i].cpu().permute(1, 2, 0).numpy()
        lab1 = rgb2lab(img1)
        lab2 = rgb2lab(img2)
        deltaE = deltaE_ciede2000(lab1, lab2)
        total += deltaE.mean()
    return total / B
