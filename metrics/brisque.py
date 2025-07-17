import torch
from piq import brisque

def compute_brisque_batch(batch):
    # batch: (B, C, H, W)
    B = batch.shape[0]
    total = 0.0
    for i in range(B):
        total += brisque(batch[i].unsqueeze(0)).item()
    return total / B
