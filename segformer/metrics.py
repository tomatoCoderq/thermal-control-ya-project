"""Binary segmentation loss and metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target)
    probability = logits.softmax(dim=1)[:, 1]
    truth = (target == 1).float()
    intersection = (probability * truth).sum(dim=(1, 2))
    denominator = probability.sum(dim=(1, 2)) + truth.sum(dim=(1, 2))
    dice_loss = 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()
    return (1.0 - dice_weight) * ce + dice_weight * dice_loss


@torch.no_grad()
def binary_counts(logits: torch.Tensor, target: torch.Tensor) -> tuple[int, int, int, int]:
    pred = logits.argmax(dim=1).bool()
    truth = target.bool()
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    eps = 1e-9
    return {
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
    }
