import json
import os
import warnings

import torch
import torchvision
import lpips

import numpy as np
import cv2

from metrics.uciqe import batch_uciqe
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    mean_squared_error,
    multiscale_structural_similarity_index_measure
)
from tqdm import tqdm

from config import Config
from data import get_data
from models import *
from utils import *


from skimage import color
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

from sewar.full_ref import vifp  # Only VIF retained

warnings.filterwarnings('ignore')



def compute_uicm(im):
    r = im[:, :, 0].astype(np.float64)
    g = im[:, :, 1].astype(np.float64)
    b = im[:, :, 2].astype(np.float64)

    rg = r - g
    yb = 0.5 * (r + g) - b

    std_rg = np.std(rg)
    std_yb = np.std(yb)

    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)

    uicm = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    return uicm


def compute_uism(im):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    uism = np.var(lap)
    return uism


def compute_uiqm(im):
    uicm = compute_uicm(im)
    uism = compute_uism(im)

    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    mean = np.mean(gray)
    contrast = np.sqrt(np.mean((gray - mean) ** 2))

    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * contrast
    return uiqm


def batch_uiqm(tensor):
    img = tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return compute_uiqm(img)


def batch_uicm(tensor):
    img = tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return compute_uicm(img)


def batch_uism(tensor):
    img = tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return compute_uism(img)





def batch_vif(pred, target):
    pred_np = pred.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    target_np = target.detach().cpu().squeeze().permute(1, 2, 0).numpy()

    pred_gray = cv2.cvtColor((pred_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor((target_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    return vifp(target_gray, pred_gray)


def test():
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    accelerator = Accelerator()
    device = accelerator.device

    val_dir = opt.TESTING.VAL_DIR

    val_dataset = get_data(
        val_dir,
        opt.TESTING.INPUT,
        opt.TESTING.TARGET,
        'test',
        opt.TRAINING.ORI,
        {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H}
    )

    testloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=True
    )

    model = UIR_PolyKernel()
    load_checkpoint(model, opt.TESTING.WEIGHT)
    model, testloader = accelerator.prepare(model, testloader)
    model.eval()

    lpips_model = lpips.LPIPS(net='alex').to(device)

    size = len(testloader)
    stat_psnr = 0
    stat_ssim = 0
    stat_uciqe = 0
    stat_mse = 0
    stat_ms_ssim = 0
    stat_lpips = 0
    stat_uiqm = 0
    stat_uicm = 0
    stat_uism = 0


    stat_vif = 0

    for _, test_data in enumerate(tqdm(testloader)):
        inp = test_data[0].contiguous().to(device)
        tar = test_data[1].to(device)

        with torch.no_grad():
            res = model(inp)

        if not os.path.isdir(opt.TESTING.RESULT_DIR):
            os.makedirs(opt.TESTING.RESULT_DIR)
        torchvision.utils.save_image(res, os.path.join(opt.TESTING.RESULT_DIR, test_data[2][0]))

        # Metrics
        stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
        stat_ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
        stat_uciqe += batch_uciqe(res)
        stat_mse += mean_squared_error(res, tar).item()
        stat_ms_ssim += multiscale_structural_similarity_index_measure(res, tar, data_range=1).item()

        res_lpips = (res * 2) - 1
        tar_lpips = (tar * 2) - 1
        stat_lpips += lpips_model(res_lpips, tar_lpips).mean().item()

        stat_uiqm += batch_uiqm(res)
        stat_uicm += batch_uicm(res)
        stat_uism += batch_uism(res)
      
        
        stat_vif += batch_vif(res, tar)

    # Averages
    stat_psnr /= size
    stat_ssim /= size
    stat_uciqe /= size
    stat_mse /= size
    stat_ms_ssim /= size
    stat_lpips /= size
    stat_uiqm /= size
    stat_uicm /= size
    stat_uism /= size


    stat_vif /= size

    test_info = (
        f"Test Result on {opt.MODEL.SESSION}, checkpoint {opt.TESTING.WEIGHT}, testing data {opt.TESTING.VAL_DIR}"
    )
    log_stats = (
        f"PSNR: {stat_psnr}, SSIM: {stat_ssim}, UCIQE: {stat_uciqe}, "
        f"MSE: {stat_mse}, MS-SSIM: {stat_ms_ssim}, LPIPS: {stat_lpips}, "
        f"UIQM: {stat_uiqm}, UICM: {stat_uicm}, UISM: {stat_uism}, "
       
        f"VIF: {stat_vif}"
    )

    print(test_info)
    print(log_stats)

    with open(os.path.join(opt.LOG.LOG_DIR, opt.TESTING.LOG_FILE), mode='a', encoding='utf-8') as f:
        f.write(json.dumps(test_info) + '\n')
        f.write(json.dumps(log_stats) + '\n')


if __name__ == '__main__':
    os.makedirs('result', exist_ok=True)
    test()
