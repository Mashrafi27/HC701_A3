"""
Segmentation evaluation metrics.
All functions accept numpy arrays (H,W) with binary values {0,1}.
Batch-level wrappers accept torch tensors (B,1,H,W) with raw logits.
"""
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


# ── per-sample numpy metrics ─────────────────────────────────────────────────

def dice_numpy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    intersection = (pred * target).sum()
    return (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)


def jaccard_numpy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)


def precision_numpy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    tp = (pred * target).sum()
    return (tp + eps) / (pred.sum() + eps)


def recall_numpy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    tp = (pred * target).sum()
    return (tp + eps) / (target.sum() + eps)


def accuracy_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    return (pred == target).mean()


def hausdorff95_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    95th-percentile Hausdorff distance on surface voxels.
    Returns 0 if either mask is empty.
    """
    pred_pts   = np.argwhere(pred   > 0).astype(float)
    target_pts = np.argwhere(target > 0).astype(float)

    if len(pred_pts) == 0 or len(target_pts) == 0:
        return 0.0

    # pairwise distances both directions
    d_pt = np.array([np.min(np.linalg.norm(target_pts - p, axis=1)) for p in pred_pts])
    d_tp = np.array([np.min(np.linalg.norm(pred_pts   - t, axis=1)) for t in target_pts])
    return float(np.percentile(np.concatenate([d_pt, d_tp]), 95))


def compute_sample_metrics(pred: np.ndarray, target: np.ndarray,
                            threshold: float = 0.5) -> dict:
    """
    Compute all metrics for a single (H,W) sample.
    pred and target should already be binary {0,1} floats.
    """
    p = (pred   > threshold).astype(np.float32)
    t = (target > threshold).astype(np.float32)

    return {
        'dice':       dice_numpy(p, t),
        'jaccard':    jaccard_numpy(p, t),
        'precision':  precision_numpy(p, t),
        'recall':     recall_numpy(p, t),
        'accuracy':   accuracy_numpy(p, t),
        'hausdorff95': hausdorff95_numpy(p, t),
    }


# ── batch-level torch helpers ─────────────────────────────────────────────────

def batch_metrics(logits: torch.Tensor, targets: torch.Tensor,
                  threshold: float = 0.5) -> dict:
    """
    logits:  (B,1,H,W) raw logits
    targets: (B,1,H,W) binary float {0,1}
    Returns dict of mean metrics over the batch.
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy()   # (B,1,H,W)
    tgts  = targets.detach().cpu().numpy()                 # (B,1,H,W)

    results = {k: [] for k in ('dice', 'jaccard', 'precision', 'recall',
                                'accuracy', 'hausdorff95')}

    for b in range(probs.shape[0]):
        m = compute_sample_metrics(probs[b, 0], tgts[b, 0], threshold)
        for k, v in m.items():
            results[k].append(v)

    return {k: float(np.mean(v)) for k, v in results.items()}
