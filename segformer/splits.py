"""Leak-free splits: an original video belongs to exactly one subset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VideoSplit:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]


def split_video_ids(
    video_ids: list[str],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> VideoSplit:
    """Deterministically split unique video ids, never frames/augmentations."""
    ids = np.asarray(sorted(set(video_ids)), dtype=object)
    if len(ids) < 3:
        raise ValueError("At least three videos are required for train/val/test")
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be positive and sum to < 1")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * val_fraction))
    n_test = max(1, round(len(ids) * test_fraction))
    n_train = len(ids) - n_val - n_test
    if n_train < 1:
        raise ValueError("Split leaves no training videos")
    return VideoSplit(
        train=tuple(ids[:n_train].tolist()),
        val=tuple(ids[n_train : n_train + n_val].tolist()),
        test=tuple(ids[n_train + n_val :].tolist()),
    )


def indices_for_videos(dataset, video_ids: tuple[str, ...]) -> list[int]:
    allowed = set(video_ids)
    return [i for i, video_id in enumerate(dataset._index) if video_id in allowed]
