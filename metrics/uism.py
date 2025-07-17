import numpy as np
import cv2


def batch_uism(batch):
    """
    batch: torch tensor in [0,1], shape (B, C, H, W)
    """
    B = batch.shape[0]
    total = 0.0
    for i in range(B):
        img = batch[i].cpu().permute(1, 2, 0).numpy()
        total += compute_uism(img)
    return total / B


def compute_uism(img):
    """
    UISM only: sharpness measure (variance of Laplacian)
    """
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()
