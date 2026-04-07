"""
evaluate.py — Task 1.2: evaluate all 5 trained models on the test set.

Usage:
    python evaluate.py            # evaluate all experiments
    python evaluate.py --exp 2    # evaluate one experiment (1-indexed)

Outputs:
    results/<exp_name>_test_metrics.json   — per-sample metrics
    results/test_summary.json              — mean ± std table for all exps
    results/test_summary.csv               — same in CSV
"""
import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.dirname(__file__))
from src.dataset import TestNerveDataset, get_transforms
from src.metrics import compute_sample_metrics
from train       import EXPERIMENTS, DEVICE, build_model

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(BASE, 'train-test')
TEST_IMG   = os.path.join(DATA_ROOT, 'testing')
TEST_MASK  = os.path.join(DATA_ROOT, 'testingmask')
CKPT_DIR   = os.path.join(BASE, 'checkpoints')
RESULT_DIR = os.path.join(BASE, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)


@torch.no_grad()
def evaluate_experiment(cfg: dict) -> list[dict]:
    """
    Evaluate a single experiment on the test set.
    Returns list of per-sample metric dicts.
    """
    ckpt = os.path.join(CKPT_DIR, f"{cfg['name']}_best.pth")
    if not os.path.exists(ckpt):
        print(f"  [SKIP] checkpoint not found: {ckpt}")
        return []

    model = build_model(cfg)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    tf      = get_transforms('test', 'none')
    dataset = TestNerveDataset(TEST_IMG, TEST_MASK, transform=tf)
    loader  = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    results = []
    for imgs, masks, names in loader:
        imgs   = imgs.to(DEVICE)
        logits = model(imgs)
        probs  = torch.sigmoid(logits).cpu().numpy()   # (B,1,H,W)
        tgts   = masks.cpu().numpy()                   # (B,1,H,W)

        for b in range(probs.shape[0]):
            m = compute_sample_metrics(probs[b, 0], tgts[b, 0])
            m['filename'] = names[b]
            results.append(m)

    # Save per-sample results
    out_path = os.path.join(RESULT_DIR, f"{cfg['name']}_test_metrics.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def summarise(all_results: dict[str, list[dict]]) -> pd.DataFrame:
    """Build a summary DataFrame: experiments × metrics (mean ± std)."""
    metric_keys = ['dice', 'jaccard', 'precision', 'recall', 'accuracy', 'hausdorff95']
    rows = []
    for exp_name, samples in all_results.items():
        if not samples:
            continue
        row = {'experiment': exp_name}
        for k in metric_keys:
            vals = [s[k] for s in samples]
            row[f'{k}_mean'] = np.mean(vals)
            row[f'{k}_std']  = np.std(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame):
    print(f"\n{'='*80}")
    print("  TEST SET RESULTS")
    print(f"{'='*80}")
    metric_keys = ['dice', 'jaccard', 'precision', 'recall', 'accuracy', 'hausdorff95']
    for _, row in df.iterrows():
        print(f"\n  {row['experiment']}")
        for k in metric_keys:
            mn  = row.get(f'{k}_mean', float('nan'))
            std = row.get(f'{k}_std',  float('nan'))
            print(f"    {k:<14}: {mn:.4f} ± {std:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0,
                        help='Which experiment to evaluate (1-5); 0 = all')
    args = parser.parse_args()

    exps = EXPERIMENTS if args.exp == 0 else [EXPERIMENTS[args.exp - 1]]

    all_results = {}
    for cfg in exps:
        print(f"\nEvaluating {cfg['name']} ...")
        results = evaluate_experiment(cfg)
        if results:
            all_results[cfg['name']] = results
            vals = [r['dice'] for r in results]
            print(f"  Dice: {np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})")

    if not all_results:
        print("No results found. Train the models first with python train.py")
        return

    df = summarise(all_results)
    print_table(df)

    # Save summary
    summary_json = os.path.join(RESULT_DIR, 'test_summary.json')
    summary_csv  = os.path.join(RESULT_DIR, 'test_summary.csv')
    df.to_csv(summary_csv, index=False)
    df.to_json(summary_json, orient='records', indent=2)
    print(f"\n  Saved: {summary_csv}")
    print(f"  Saved: {summary_json}")


if __name__ == '__main__':
    main()
