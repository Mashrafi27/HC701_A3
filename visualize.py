"""
visualize.py — Task 1.4 (error analysis) and Task 1.5 (qualitative plots).

Usage:
    python visualize.py --task all      # run both tasks
    python visualize.py --task 1.4
    python visualize.py --task 1.5
"""
import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from PIL import Image
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from src.dataset import TestNerveDataset, get_transforms
from train       import EXPERIMENTS, DEVICE, build_model

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(BASE, 'train-test')
TEST_IMG   = os.path.join(DATA_ROOT, 'testing')
TEST_MASK  = os.path.join(DATA_ROOT, 'testingmask')
CKPT_DIR   = os.path.join(BASE, 'checkpoints')
RESULT_DIR = os.path.join(BASE, 'results')
FIG_DIR    = os.path.join(BASE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Matplotlib style for publication quality
plt.rcParams.update({
    'font.family':      'DejaVu Serif',
    'font.size':        9,
    'axes.titlesize':   9,
    'axes.labelsize':   8,
    'xtick.labelsize':  7,
    'ytick.labelsize':  7,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.05,
})

EXP_SHORT = {
    'Exp1_UNet_Random_BCE':           'Exp1\nU-Net (random, BCE)',
    'Exp2_UNet_Pretrained_BCEDice':   'Exp2\nU-Net (IN, BCE+Dice)',
    'Exp3_UNetPP_EfficientNet_FocalDice': 'Exp3\nUNet++ (IN, Focal+Dice)',
    'Exp4_FPN_ResNet50_Dice':         'Exp4\nFPN (IN, Dice)',
    'Exp5_MAnet_ResNet34_Combined':   'Exp5\nMAnet (IN, Combined)',
}


# ── utilities ─────────────────────────────────────────────────────────────────

def load_raw_image(filename: str) -> np.ndarray:
    """Return the original test image as a uint8 grayscale numpy array."""
    path = os.path.join(TEST_IMG, filename)
    return np.array(Image.open(path).convert('L'))


def load_raw_mask(filename: str) -> np.ndarray:
    """Return the ground-truth mask as a binary float32 numpy array."""
    path = os.path.join(TEST_MASK, filename)
    arr  = np.array(Image.open(path).convert('L'))
    return (arr > 127).astype(np.float32)


@torch.no_grad()
def get_all_predictions(cfg: dict) -> dict[str, np.ndarray]:
    """
    Run model inference on every test image.
    Returns {filename: pred_mask (H,W) float32 [0,1]}.
    """
    ckpt = os.path.join(CKPT_DIR, f"{cfg['name']}_best.pth")
    if not os.path.exists(ckpt):
        return {}

    model = build_model(cfg)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    tf      = get_transforms('test', 'none')
    dataset = TestNerveDataset(TEST_IMG, TEST_MASK, transform=tf)
    loader  = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    preds = {}
    from src.dataset import IMG_SIZE
    from PIL import Image as _PIL
    import torch.nn.functional as F

    for imgs, _, names in loader:
        imgs   = imgs.to(DEVICE)
        logits = model(imgs)                                    # (B,1,H,W)
        probs  = torch.sigmoid(logits).cpu().numpy()[:, 0]    # (B,H,W)
        for b, name in enumerate(names):
            preds[name] = probs[b]
    return preds


def overlay(img_gray: np.ndarray, mask: np.ndarray,
            color=(0, 1, 0), alpha: float = 0.4) -> np.ndarray:
    """Overlay a binary mask onto a grayscale image (both original resolution)."""
    rgb = np.stack([img_gray / 255.0] * 3, axis=-1)
    for c, v in enumerate(color):
        rgb[..., c] = np.where(mask > 0.5,
                               (1 - alpha) * rgb[..., c] + alpha * v,
                               rgb[..., c])
    return np.clip(rgb, 0, 1)


# ── Task 1.4 — error analysis ─────────────────────────────────────────────────

def task_1_4_error_analysis():
    """
    Identify the worst 10% of test cases (by Dice) from the best experiment,
    visualise them and explain failure modes.
    """
    # Find the best experiment from saved per-sample metrics
    best_exp, best_mean_dice = None, -1.0
    for cfg in EXPERIMENTS:
        path = os.path.join(RESULT_DIR, f"{cfg['name']}_test_metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            samples = json.load(f)
        mean_dice = np.mean([s['dice'] for s in samples])
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            best_exp       = cfg
            best_samples   = samples

    if best_exp is None:
        print("[Task 1.4] No test metrics found. Run evaluate.py first.")
        return

    print(f"[Task 1.4] Best experiment: {best_exp['name']}  (mean Dice={best_mean_dice:.4f})")

    # Sort by Dice ascending → worst cases first
    sorted_samples = sorted(best_samples, key=lambda s: s['dice'])
    n_worst        = max(1, int(len(sorted_samples) * 0.10))
    worst_samples  = sorted_samples[:n_worst]

    print(f"  Worst 10%: {n_worst} cases  "
          f"(Dice range: {worst_samples[0]['dice']:.4f}–{worst_samples[-1]['dice']:.4f})")

    # Get predictions for best experiment
    preds = get_all_predictions(best_exp)
    if not preds:
        print("[Task 1.4] Could not load model predictions.")
        return

    # Select up to 8 representative failure cases to show
    show_n  = min(8, n_worst)
    to_show = worst_samples[:show_n]

    ncols = 4
    nrows = show_n
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.5, nrows * 2.2))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Original Image', 'Ground Truth', 'Prediction', 'Overlay (GT=green, Pred=red)']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=8, fontweight='bold', pad=4)

    for i, sample in enumerate(to_show):
        fn   = sample['filename']
        img  = load_raw_image(fn)
        gt   = load_raw_mask(fn)
        pred = preds.get(fn)

        if pred is None:
            continue

        # Resize pred back to original resolution for display
        from PIL import Image as _PIL
        pred_full = np.array(_PIL.fromarray((pred * 255).astype(np.uint8)).resize(
            (img.shape[1], img.shape[0]), _PIL.BILINEAR)) / 255.0

        # Composite overlay: GT in green, prediction in red
        rgb = np.stack([img / 255.0] * 3, axis=-1)
        alpha = 0.45
        gt_bin   = (gt > 0.5).astype(float)
        pred_bin = (pred_full > 0.5).astype(float)

        overlay_img = rgb.copy()
        # GT overlay (green)
        overlay_img[..., 1] = np.where(gt_bin > 0,
                                        (1 - alpha) * rgb[..., 1] + alpha, rgb[..., 1])
        # Pred overlay (red)
        overlay_img[..., 0] = np.where(pred_bin > 0,
                                        (1 - alpha) * rgb[..., 0] + alpha, rgb[..., 0])
        overlay_img = np.clip(overlay_img, 0, 1)

        for j, im in enumerate([img, gt, pred_full, overlay_img]):
            ax = axes[i, j]
            cmap = 'gray' if j < 3 else None
            ax.imshow(im, cmap=cmap, vmin=0, vmax=1 if j > 0 else None)
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(f"Dice={sample['dice']:.3f}", fontsize=7, rotation=0,
                              labelpad=55, va='center')

    legend_elems = [Patch(facecolor='green', alpha=0.6, label='Ground truth'),
                    Patch(facecolor='red',   alpha=0.6, label='Prediction')]
    fig.legend(handles=legend_elems, loc='lower center', ncol=2,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(f'Task 1.4 — Worst 10% Failure Cases\n({best_exp["name"]})',
                 fontsize=10, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, 'task1_4_error_analysis.pdf')
    plt.savefig(out_path, format='pdf')
    png_path = out_path.replace('.pdf', '.png')
    plt.savefig(png_path, format='png')
    plt.close()
    print(f"  Saved: {out_path}")
    print(f"  Saved: {png_path}")

    # Print worst-case statistics
    dice_vals = [s['dice'] for s in worst_samples]
    jac_vals  = [s['jaccard'] for s in worst_samples]
    print(f"\n  Worst-10% statistics:")
    print(f"    Dice:    {np.mean(dice_vals):.4f} ± {np.std(dice_vals):.4f}")
    print(f"    Jaccard: {np.mean(jac_vals):.4f}  ± {np.std(jac_vals):.4f}")


# ── Task 1.5 — qualitative visualisation ─────────────────────────────────────

def task_1_5_qualitative():
    """
    For 5 randomly selected test samples, show original / GT / prediction
    for each of the 5 experiments in a publication-quality figure.
    """
    # Load all experiments' predictions
    all_preds   = {}
    valid_exps  = []
    for cfg in EXPERIMENTS:
        p = get_all_predictions(cfg)
        if p:
            all_preds[cfg['name']] = p
            valid_exps.append(cfg)

    if not valid_exps:
        print("[Task 1.5] No trained models found. Run train.py first.")
        return

    # Randomly pick 5 test filenames
    all_files = sorted(os.listdir(TEST_IMG))
    rng       = np.random.default_rng(seed=7)
    chosen    = rng.choice(all_files, size=5, replace=False).tolist()
    print(f"[Task 1.5] Selected test samples: {chosen}")

    n_exp  = len(valid_exps)
    n_samp = len(chosen)

    # Layout: rows = samples, cols = 2 + n_exp  (image | GT | exp1..N)
    ncols = 2 + n_exp
    nrows = n_samp
    col_w = 1.9
    row_h = 1.8

    fig = plt.figure(figsize=(ncols * col_w, nrows * row_h + 0.6))
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                            hspace=0.08, wspace=0.04)

    col_labels = ['Image', 'Ground\nTruth'] + \
                 [EXP_SHORT.get(cfg['name'], cfg['name']) for cfg in valid_exps]

    # Column headers
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis('off')

    for j, label in enumerate(col_labels):
        ax = fig.add_axes([
            gs[0, j].get_position(fig).x0,
            gs[0, j].get_position(fig).y1 + 0.002,
            gs[0, j].get_position(fig).width,
            0.04,
        ])
        ax.text(0.5, 0.5, label, ha='center', va='center',
                fontsize=7.5, fontweight='bold',
                transform=ax.transAxes)
        ax.axis('off')

    for i, fn in enumerate(chosen):
        img  = load_raw_image(fn)
        gt   = load_raw_mask(fn)

        # Column 0: original image
        ax0 = fig.add_subplot(gs[i, 0])
        ax0.imshow(img, cmap='gray')
        ax0.axis('off')
        ax0.set_ylabel(f"Sample {i+1}", fontsize=7, rotation=90,
                       labelpad=3, va='center')

        # Column 1: ground truth
        ax1 = fig.add_subplot(gs[i, 1])
        ax1.imshow(gt, cmap='gray', vmin=0, vmax=1)
        ax1.axis('off')

        # Columns 2+: predictions
        for k, cfg in enumerate(valid_exps):
            pred = all_preds[cfg['name']].get(fn)
            ax   = fig.add_subplot(gs[i, 2 + k])
            if pred is not None:
                # Resize back to original resolution
                from PIL import Image as _PIL
                pred_disp = np.array(
                    _PIL.fromarray((pred * 255).astype(np.uint8)).resize(
                        (img.shape[1], img.shape[0]), _PIL.BILINEAR
                    )
                ) / 255.0
                ax.imshow(pred_disp, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

    fig.suptitle('Task 1.5 — Qualitative Segmentation Results (5 Test Samples × 5 Experiments)',
                 fontsize=10, fontweight='bold', y=1.005)

    out_path = os.path.join(FIG_DIR, 'task1_5_qualitative.pdf')
    plt.savefig(out_path, format='pdf')
    png_path = out_path.replace('.pdf', '.png')
    plt.savefig(png_path, format='png')
    plt.close()
    print(f"  Saved: {out_path}")
    print(f"  Saved: {png_path}")


# ── training curves ───────────────────────────────────────────────────────────

def plot_training_curves():
    """Plot training / validation loss and Dice for all experiments."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(EXPERIMENTS)))

    for cfg, color in zip(EXPERIMENTS, colors):
        hist_path = os.path.join(RESULT_DIR, f"{cfg['name']}_history.json")
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as f:
            history = json.load(f)
        epochs = [h['epoch'] for h in history]
        axes[0].plot(epochs, [h['vl_loss'] for h in history],
                     label=cfg['name'].split('_', 1)[0], color=color, lw=1.5)
        axes[1].plot(epochs, [h['vl_dice'] for h in history],
                     color=color, lw=1.5)

    for ax, ylabel in zip(axes, ['Validation Loss', 'Validation Dice']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle('Training Curves — All Experiments', fontsize=10, fontweight='bold')
    plt.tight_layout()

    out = os.path.join(FIG_DIR, 'training_curves.pdf')
    plt.savefig(out, format='pdf')
    plt.savefig(out.replace('.pdf', '.png'), format='png')
    plt.close()
    print(f"  Saved training curves: {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='all',
                        choices=['all', '1.4', '1.5', 'curves'])
    args = parser.parse_args()

    if args.task in ('all', 'curves'):
        print("\n[Plotting training curves]")
        plot_training_curves()

    if args.task in ('all', '1.4'):
        print("\n[Task 1.4 — Error Analysis]")
        task_1_4_error_analysis()

    if args.task in ('all', '1.5'):
        print("\n[Task 1.5 — Qualitative Visualisation]")
        task_1_5_qualitative()


if __name__ == '__main__':
    main()
