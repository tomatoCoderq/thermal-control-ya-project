"""Shared training utilities for Thermal-Contrast."""
from __future__ import annotations

from .device import get_device
from .loop import run_epoch
from .metrics import (
    BCEDiceLoss,
    SegMetrics,
    SoftDiceLoss,
    combined_bce_dice,
    dice_score,
    iou_score,
)
from .split import split_videos
from .tracking import HistoryTracker

__all__ = [
    "BCEDiceLoss",
    "HistoryTracker",
    "SegMetrics",
    "SoftDiceLoss",
    "combined_bce_dice",
    "dice_score",
    "get_device",
    "iou_score",
    "run_epoch",
    "split_videos",
]
