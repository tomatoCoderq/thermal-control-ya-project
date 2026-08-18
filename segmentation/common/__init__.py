"""Shared training utilities for segmentation/U-Net, Attention U-Net, Mamba-UNet.

Mirrors HeatControl/u-net: video-level train/test split, SegMetrics, history.json.
"""
from __future__ import annotations

# re-exports for convenience
from .data import INPUT_SIZE, CropDataset, build_train_test_loaders
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
from .variants import VARIANTS, apply_variant_features

__all__ = [
    "INPUT_SIZE",
    "VARIANTS",
    "BCEDiceLoss",
    "CropDataset",
    "HistoryTracker",
    "SegMetrics",
    "SoftDiceLoss",
    "apply_variant_features",
    "build_train_test_loaders",
    "combined_bce_dice",
    "dice_score",
    "get_device",
    "iou_score",
    "run_epoch",
    "split_videos",
]
