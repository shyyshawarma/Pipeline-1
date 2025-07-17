import json
import os
import warnings

import torch
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    mean_squared_error,
    multiscale_structural_similarity_index_measure,
    fsim,
    vif
)

from tqdm import tqdm

from config import Config
from data import get_data
from metrics.uciqe import batch_uciqe
from metrics.uiqm import batch_uiqm
from metrics.uism import batch_uism
from metrics.brisque import compute_brisque_batch
from metrics.lpips import LPIPS
from metrics.ciede2000 import batch_ciede2000

from models import *
from utils import *

warnings.filterwarnings('ignore')


# --- Charbonnier Loss ---
class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


# --- Gradient Loss ---
def gradient(x):
    D_dx = x[:, :, :, :-1] - x[:, :, :, 1:]
    D_dy = x[:, :, :-1, :] - x[:, :, 1:, :]
    return D_dx, D_dy


def gradient_loss(x, y):
    dx1, dy1 = gradient(x)
    dx2, dy2 = gradient(y)
    return torch.mean(torch.abs(dx1 - dx2) + torch.abs(dy1 - dy2))


def train():
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator()
    if accelerator.is_local_main_process:
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)
    device = accelerator.device

    accelerator.init_trackers("UW", config={"dataset": opt.TRAINING.TRAIN_DIR})

    train_dataset = get_data(opt.TRAINING.TRAIN_DIR, opt.MODEL.INPUT, opt.MODEL.TARGET, 'train', opt.TRAINING.ORI,
                             {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    trainloader = DataLoader(train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, num_workers=16,
                             sampler=ImbalancedDatasetSampler(train_dataset), pin_memory=True)

    val_dataset = get_data(opt.TRAINING.VAL_DIR, opt.MODEL.INPUT, opt.MODEL.TARGET, 'test', opt.TRAINING.ORI,
                           {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    testloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    model = UIR_PolyKernel()

    # Losses
    criterion_charbonnier = CharbonnierLoss().to(device)
    lpips_loss = LPIPS(net='vgg').to(device)

    # Trainable weights Ω
    omega1 = torch.nn.Parameter(torch.tensor(0.2741, device=device), requires_grad=True)
    omega2 = torch.nn.Parameter(torch.tensor(0.2222, device=device), requires_grad=True)
    omega3 = torch.nn.Parameter(torch.tensor(0.3357, device=device), requires_grad=True)
    omega4 = torch.nn.Parameter(torch.tensor(0.1680, device=device), requires_grad=True)

    # Optimizer
    optimizer_b = optim.AdamW(
        list(model.parameters()) + [omega1, omega2, omega3, omega4],
        lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8
    )
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    start_epoch, best_psnr, best_psnr_epoch = 1, 0, 1
    early_stopping_patience, epochs_since_improvement = 10, 0

    if opt.TRAINING.RESUME and opt.TRAINING.WEIGHT is not None:
        checkpoint = torch.load(opt.TRAINING.WEIGHT, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            optimizer_b.load_state_dict(checkpoint['optimizer'])
            scheduler_b.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_psnr = checkpoint.get('best_psnr', 0)
            best_psnr_epoch = checkpoint.get('best_psnr_epoch', 1)
        print(f"✅ Resumed from epoch {start_epoch - 1}")

    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model = accelerator.prepare(model)
    optimizer_b, scheduler_b = accelerator.prepare(optimizer_b, scheduler_b)

    size = len(testloader)

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()

        for _, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
            inp, tar = data[0].contiguous(), data[1]
            optimizer_b.zero_grad()
            res = model(inp)

            loss_C = criterion_charbonnier(res, tar)
            loss_G = gradient_loss(res, tar)
            loss_M = 1 - multiscale_structural_similarity_index_measure(res, tar, data_range=1)
            loss_P = lpips_loss(res, tar).mean()

            train_loss = omega1 * loss_C + omega2 * loss_G + omega3 * loss_M + omega4 * loss_P

            accelerator.backward(train_loss)
            optimizer_b.step()

        scheduler_b.step()

        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            metrics = {k: 0 for k in ['psnr', 'ssim', 'msssim', 'mse', 'fsim', 'vif',
                                       'uiqm', 'uism', 'uciqe', 'brisque', 'lpips', 'ciede2000']}

            for _, data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
                inp, tar = data[0].contiguous(), data[1]
                with torch.no_grad():
                    res = model(inp)

                res, tar = accelerator.gather((res, tar))

                metrics['psnr'] += peak_signal_noise_ratio(res, tar, data_range=1).item()
                metrics['ssim'] += structural_similarity_index_measure(res, tar, data_range=1).item()
                metrics['msssim'] += multiscale_structural_similarity_index_measure(res, tar, data_range=1).item()
                metrics['mse'] += mean_squared_error(res, tar).item()
                metrics['fsim'] += fsim(res, tar, data_range=1).item()
                metrics['vif'] += vif(res, tar, data_range=1).item()
                metrics['uiqm'] += batch_uiqm(res)
                metrics['uism'] += batch_uism(res)
                metrics['uciqe'] += batch_uciqe(res)
                metrics['brisque'] += compute_brisque_batch(res)
                metrics['lpips'] += lpips_loss(res, tar).mean().item()
                metrics['ciede2000'] += batch_ciede2000(res, tar)

            for k in metrics:
                metrics[k] /= size

            improved = metrics['psnr'] > best_psnr
            if improved:
                best_psnr = metrics['psnr']
                best_psnr_epoch = epoch
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            save_checkpoint({
                'epoch': epoch,
                'state_dict': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer_b.state_dict(),
                'scheduler': scheduler_b.state_dict(),
                'best_psnr': best_psnr,
                'best_psnr_epoch': best_psnr_epoch
            }, epoch, opt.MODEL.SESSION, opt.TRAINING.SAVE_DIR)

            accelerator.log(metrics | {
                "omega1": omega1.item(),
                "omega2": omega2.item(),
                "omega3": omega3.item(),
                "omega4": omega4.item(),
            }, step=epoch)

            if accelerator.is_local_main_process:
                print(f"📊 epoch: {epoch}, {json.dumps(metrics, indent=2)}, "
                      f"weights: [{omega1.item():.4f}, {omega2.item():.4f}, "
                      f"{omega3.item():.4f}, {omega4.item():.4f}], "
                      f"best_psnr: {best_psnr} at epoch {best_psnr_epoch}")
                with open(os.path.join(opt.LOG.LOG_DIR, opt.TRAINING.LOG_FILE), 'a') as f:
                    f.write(json.dumps(metrics) + '\n')

            if epochs_since_improvement >= early_stopping_patience:
                if accelerator.is_local_main_process:
                    print(f"⏹️ Early stopping at epoch {epoch} due to no PSNR improvement for {early_stopping_patience} checks.")
                break

    accelerator.end_training()


if __name__ == '__main__':
    train()
