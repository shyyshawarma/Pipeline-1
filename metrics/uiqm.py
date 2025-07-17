import numpy as np
import cv2


def batch_uiqm(batch):
    """
    batch: torch tensor in [0,1], shape (B, C, H, W)
    """
    B = batch.shape[0]
    total = 0.0
    for i in range(B):
        img = batch[i].cpu().permute(1, 2, 0).numpy()
        total += compute_uiqm(img)
    return total / B


def compute_uiqm(img):
    """
    img: HWC, RGB, [0,1]
    """
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return getUIQM(img)


def getUIQM(img):
    """
    Adapted from original UIQM paper implementation.
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y = img[:, :, 0].astype(np.float32)

    # 1) UIConM: contrast measure
    maxY = np.max(Y)
    minY = np.min(Y)
    uiconm = (maxY - minY) / (maxY + minY + 1e-12)

    # 2) UICM: colorfulness measure
    img_rgb = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB).astype(np.float32)
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]
    RG = R - G
    YB = 0.5 * (R + G) - B
    stdRG = np.std(RG)
    stdYB = np.std(YB)
    uicm = np.sqrt(stdRG ** 2 + stdYB ** 2)

    # 3) UISM: sharpness measure (variance of Laplacian)
    lap = cv2.Laplacian(Y, cv2.CV_64F)
    uism = lap.var()

    # Original paper weights
    uiqm = 0.0282 * uicm + 0.2953 * uiconm + 3.5753 * uism
    return uiqm
