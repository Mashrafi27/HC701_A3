"""
Custom loss functions for nerve segmentation.
All losses operate on raw logits (before sigmoid).
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

_BINARY = smp.losses.BINARY_MODE


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.0)

    def forward(self, logits, targets):
        return self.bce(logits, targets)


class BCEDiceLoss(nn.Module):
    """BCE + Dice (equal weight by default)."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_w  = bce_weight
        self.dice_w = dice_weight
        self.bce    = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.0)
        self.dice   = smp.losses.DiceLoss(mode=_BINARY, from_logits=True)

    def forward(self, logits, targets):
        return self.bce_w * self.bce(logits, targets) + \
               self.dice_w * self.dice(logits, targets)


class FocalDiceLoss(nn.Module):
    """Focal loss + Dice loss."""

    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5,
                 gamma: float = 2.0):
        super().__init__()
        self.focal_w = focal_weight
        self.dice_w  = dice_weight
        self.focal   = smp.losses.FocalLoss(mode=_BINARY, gamma=gamma)
        self.dice    = smp.losses.DiceLoss(mode=_BINARY, from_logits=True)

    def forward(self, logits, targets):
        return self.focal_w * self.focal(logits, targets) + \
               self.dice_w  * self.dice(logits, targets)


class DiceLoss(nn.Module):
    """Pure Dice loss."""

    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode=_BINARY, from_logits=True)

    def forward(self, logits, targets):
        return self.dice(logits, targets)


class CombinedLoss(nn.Module):
    """BCE + Dice + Focal — weighted combination."""

    def __init__(self, bce_weight: float = 0.4, dice_weight: float = 0.4,
                 focal_weight: float = 0.2, gamma: float = 2.0):
        super().__init__()
        self.bce_w   = bce_weight
        self.dice_w  = dice_weight
        self.focal_w = focal_weight
        self.bce     = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.0)
        self.dice    = smp.losses.DiceLoss(mode=_BINARY, from_logits=True)
        self.focal   = smp.losses.FocalLoss(mode=_BINARY, gamma=gamma)

    def forward(self, logits, targets):
        return (self.bce_w   * self.bce(logits, targets) +
                self.dice_w  * self.dice(logits, targets) +
                self.focal_w * self.focal(logits, targets))


LOSS_REGISTRY = {
    'bce':          BCELoss,
    'bce_dice':     BCEDiceLoss,
    'focal_dice':   FocalDiceLoss,
    'dice':         DiceLoss,
    'combined':     CombinedLoss,
}


def get_loss(name: str) -> nn.Module:
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name]()
