"""Binary segmentation loss + dataset-level SegMetrics (HeatControl style)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p = pred.astype(bool).ravel()
    g = gt.astype(bool).ravel()
    inter = np.logical_and(p, g).sum()
    return float((2 * inter + eps) / (p.sum() + g.sum() + eps))


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p = pred.astype(bool).ravel()
    g = gt.astype(bool).ravel()
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float((inter + eps) / (union + eps))


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        m1 = probs.contiguous().reshape(probs.size(0), -1)
        m2 = targets.contiguous().reshape(targets.size(0), -1).to(probs.dtype)
        inter = (m1 * m2).sum(1)
        # (2*inter + smooth) / (…), NOT 2*(inter + smooth) —
        # empty pred+GT otherwise gives Dice=2 and loss=-1.
        score = (2.0 * inter + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        return 1.0 - score.mean()


class BCEDiceLoss(nn.Module):
    """BCE-with-logits + soft Dice.

    ``pos_weight`` scales the positive term of the BCE. Defect pixels are ~2% of a
    frame, so without it the BCE part is almost entirely a background reconstruction
    term and the model is rewarded for predicting empty masks.
    """

    def __init__(
        self,
        *,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = None if pos_weight is None else float(pos_weight)
        self.dice = SoftDiceLoss(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        pw = (
            None
            if self.pos_weight is None
            else torch.tensor(self.pos_weight, dtype=logits.dtype, device=logits.device)
        )
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, targets)


_DEFAULT_LOSS = BCEDiceLoss()


def combined_bce_dice(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Unweighted BCE + soft Dice (default criterion)."""
    return _DEFAULT_LOSS(logits, targets)


class SegMetrics:
    """Accumulate TP/FP/FN over a split; report IoU, Dice, precision, recall."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.tp = self.fp = self.fn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = torch.sigmoid(logits) > self.threshold
        gt = target > 0.5
        self.tp += float((pred & gt).sum())
        self.fp += float((pred & ~gt).sum())
        self.fn += float((~pred & gt).sum())

    def compute(self, eps: float = 1e-6) -> dict[str, float]:
        return {
            "iou": self.tp / (self.tp + self.fp + self.fn + eps),
            "dice": 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps),
            "precision": self.tp / (self.tp + self.fp + eps),
            "recall": self.tp / (self.tp + self.fn + eps),
        }
