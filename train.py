"""
train.py — run all 5 experiments for HC701 Assignment 3.

Usage:
    python train.py                       # run all experiments
    python train.py --exp 2               # run a single experiment (1-indexed)
    python train.py --exp 2 --epochs 30   # override max epochs
"""
import argparse
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.dataset import NerveDataset, get_transforms
from src.losses  import get_loss
from src.metrics import batch_metrics

# ── paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(BASE, 'train-test')
TRAIN_IMG  = os.path.join(DATA_ROOT, 'training')
TRAIN_MASK = os.path.join(DATA_ROOT, 'trainingmask')
CKPT_DIR   = os.path.join(BASE, 'checkpoints')
RESULT_DIR = os.path.join(BASE, 'results')

os.makedirs(CKPT_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ── device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f"[device] {DEVICE}")

# ── experiment configurations ─────────────────────────────────────────────────
# Each dict fully describes one experiment.
EXPERIMENTS = [
    {
        'name':         'Exp1_UNet_Random_BCE',
        'description':  'U-Net (random init) | BCE | minimal augmentations',
        'architecture': 'Unet',
        'encoder':      'resnet34',
        'weights':      None,          # random initialisation
        'loss':         'bce',
        'augment':      'light',
        'lr':           1e-3,
        'batch_size':   8,
    },
    {
        'name':         'Exp2_UNet_Pretrained_BCEDice',
        'description':  'U-Net (ImageNet ResNet-34) | BCE+Dice | moderate augmentations',
        'architecture': 'Unet',
        'encoder':      'resnet34',
        'weights':      'imagenet',
        'loss':         'bce_dice',
        'augment':      'moderate',
        'lr':           1e-4,
        'batch_size':   8,
    },
    {
        'name':         'Exp3_UNetPP_EfficientNet_FocalDice',
        'description':  'UNet++ (ImageNet EfficientNet-B0) | Focal+Dice | heavy augmentations',
        'architecture': 'UnetPlusPlus',
        'encoder':      'efficientnet-b0',
        'weights':      'imagenet',
        'loss':         'focal_dice',
        'augment':      'heavy',
        'lr':           1e-4,
        'batch_size':   8,
    },
    {
        'name':         'Exp4_FPN_ResNet50_Dice',
        'description':  'FPN (ImageNet ResNet-50) | Dice | heavy augmentations + CLAHE',
        'architecture': 'FPN',
        'encoder':      'resnet50',
        'weights':      'imagenet',
        'loss':         'dice',
        'augment':      'heavy',
        'lr':           1e-4,
        'batch_size':   8,
    },
    {
        'name':         'Exp5_MAnet_ResNet34_Combined',
        'description':  'MAnet (ImageNet ResNet-34) | BCE+Dice+Focal | very heavy augmentations',
        'architecture': 'MAnet',
        'encoder':      'resnet34',
        'weights':      'imagenet',
        'loss':         'combined',
        'augment':      'very_heavy',
        'lr':           1e-4,
        'batch_size':   8,
    },
]


# ── model builder ─────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    arch = getattr(smp, cfg['architecture'])
    model = arch(
        encoder_name=cfg['encoder'],
        encoder_weights=cfg['weights'],
        in_channels=3,
        classes=1,
        activation=None,   # raw logits; sigmoid applied at inference
    )
    return model.to(DEVICE)


# ── training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    losses, dices = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        with torch.no_grad():
            m = batch_metrics(logits, masks)
        dices.append(m['dice'])
    return float(np.mean(losses)), float(np.mean(dices))


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    losses, all_metrics = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, masks)
        losses.append(loss.item())
        all_metrics.append(batch_metrics(logits, masks))
    mean_loss = float(np.mean(losses))
    mean_m    = {k: float(np.mean([m[k] for m in all_metrics]))
                 for k in all_metrics[0]}
    return mean_loss, mean_m


# ── picklable dataset classes (must be at module level for DataLoader workers) ─

class RawNerveDataset(torch.utils.data.Dataset):
    """Loads images/masks as numpy arrays without any transform."""
    def __init__(self, img_dir: str, mask_dir: str, indices: list):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.images   = sorted(os.listdir(img_dir))
        self.indices  = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        name  = self.images[self.indices[idx]]
        img   = np.array(Image.open(os.path.join(self.img_dir,  name)).convert('RGB'))
        mask  = np.array(Image.open(os.path.join(self.mask_dir, name)).convert('L'))
        mask  = (mask > 127).astype(np.float32)
        return img, mask


class TransformedDataset(torch.utils.data.Dataset):
    """Wraps a RawNerveDataset and applies albumentations + tensor conversion."""
    def __init__(self, raw_ds: RawNerveDataset, transform):
        self.raw_ds    = raw_ds
        self.transform = transform

    def __len__(self):
        return len(self.raw_ds)

    def __getitem__(self, idx):
        img, mask = self.raw_ds[idx]
        out       = self.transform(image=img, mask=mask)
        img_t     = torch.from_numpy(out['image'].transpose(2, 0, 1).copy())
        mask_t    = torch.from_numpy(out['mask'].copy())
        return img_t, mask_t.unsqueeze(0)


# ── main training loop ────────────────────────────────────────────────────────

def run_experiment(cfg: dict, max_epochs: int = 50, val_split: float = 0.15):
    print(f"\n{'='*60}")
    print(f"  {cfg['name']}")
    print(f"  {cfg['description']}")
    print(f"{'='*60}")

    # ── data ──────────────────────────────────────────────────────
    train_tf = get_transforms('train', cfg['augment'])
    val_tf   = get_transforms('val',   'none')

    all_files = sorted(os.listdir(TRAIN_IMG))
    n_total   = len(all_files)
    n_val     = int(n_total * val_split)
    indices   = list(range(n_total))
    rng       = np.random.default_rng(42)
    rng.shuffle(indices)
    val_idx   = indices[:n_val]
    trn_idx   = indices[n_val:]

    raw_train = RawNerveDataset(TRAIN_IMG, TRAIN_MASK, trn_idx)
    raw_val   = RawNerveDataset(TRAIN_IMG, TRAIN_MASK, val_idx)

    train_dataset = TransformedDataset(raw_train, train_tf)
    val_dataset   = TransformedDataset(raw_val,   val_tf)

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                              shuffle=True,  num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=False)

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── model / loss / optimiser ───────────────────────────────────
    model     = build_model(cfg)
    criterion = get_loss(cfg['loss'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # ── training loop ──────────────────────────────────────────────
    best_val_dice = 0.0
    best_ckpt     = os.path.join(CKPT_DIR, f"{cfg['name']}_best.pth")
    history       = []
    patience_cnt  = 0
    EARLY_STOP    = 12

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_m    = validate(model, val_loader, criterion)
        scheduler.step(vl_m['dice'])

        history.append({
            'epoch':    epoch,
            'tr_loss':  tr_loss,
            'tr_dice':  tr_dice,
            'vl_loss':  vl_loss,
            **{f'vl_{k}': v for k, v in vl_m.items()},
        })

        improved = vl_m['dice'] > best_val_dice
        if improved:
            best_val_dice = vl_m['dice']
            torch.save(model.state_dict(), best_ckpt)
            patience_cnt = 0
        else:
            patience_cnt += 1

        elapsed = time.time() - t0
        marker  = ' *' if improved else ''
        print(f"  Ep {epoch:02d}/{max_epochs} | "
              f"tr_loss={tr_loss:.4f} tr_dice={tr_dice:.4f} | "
              f"vl_loss={vl_loss:.4f} vl_dice={vl_m['dice']:.4f} "
              f"vl_iou={vl_m['jaccard']:.4f} | "
              f"{elapsed:.1f}s{marker}")

        if patience_cnt >= EARLY_STOP:
            print(f"  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP} epochs)")
            break

    # ── save history ───────────────────────────────────────────────
    hist_path = os.path.join(RESULT_DIR, f"{cfg['name']}_history.json")
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best val Dice: {best_val_dice:.4f}")
    print(f"  Checkpoint:    {best_ckpt}")
    print(f"  History:       {hist_path}")
    return history, best_val_dice


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',    type=int, default=0,
                        help='Which experiment to run (1-5); 0 = all')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    if args.exp == 0:
        exps = EXPERIMENTS
    else:
        if not (1 <= args.exp <= len(EXPERIMENTS)):
            print(f"--exp must be 1-{len(EXPERIMENTS)}")
            sys.exit(1)
        exps = [EXPERIMENTS[args.exp - 1]]

    summary = {}
    for cfg in exps:
        history, best_dice = run_experiment(cfg, max_epochs=args.epochs)
        summary[cfg['name']] = best_dice

    print(f"\n{'='*60}")
    print("  SUMMARY — best validation Dice per experiment")
    print(f"{'='*60}")
    for name, dice in summary.items():
        print(f"  {name}: {dice:.4f}")


if __name__ == '__main__':
    main()
