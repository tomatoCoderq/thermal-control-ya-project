"""DataLoader helpers and collate functions for IRTDataset."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from irt_data.config import DatasetConfig
from irt_data.dataset import IRTDataset


def worker_init_fn(worker_id: int) -> None:
    """Seed numpy/random differently per worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def stack_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack samples with identical shapes -> image [B,C,H,W] or [B,T,C,H,W]."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    out: dict[str, Any] = {
        "image": images,
        "mask": masks,
        "video_id": [b["video_id"] for b in batch],
        "has_mask": torch.stack([b["has_mask"] for b in batch], dim=0),
        "crop": torch.stack([b["crop"] for b in batch], dim=0),
    }
    # frame_indices may differ in length only if pad_collate; here assume same T
    if batch[0]["frame_indices"].numel() > 0:
        try:
            out["frame_indices"] = torch.stack(
                [b["frame_indices"] for b in batch], dim=0
            )
        except RuntimeError:
            # fallback: leave as list
            out["frame_indices"] = [b["frame_indices"] for b in batch]
    else:
        out["frame_indices"] = None
    return out


def pad_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length temporal clips to max T in the batch.

    Returns
    -------
    image : [B, T_max, C, H, W]
    lengths : [B]
    valid_mask : [B, T_max] bool
    """
    assert batch[0]["image"].ndim == 4, "pad_collate expects temporal [T,C,H,W]"
    lengths = [int(b["image"].shape[0]) for b in batch]
    t_max = max(lengths)
    c, h, w = batch[0]["image"].shape[1:]
    bsz = len(batch)

    images = batch[0]["image"].new_zeros((bsz, t_max, c, h, w))
    valid = torch.zeros((bsz, t_max), dtype=torch.bool)
    masks = torch.stack([b["mask"] for b in batch], dim=0)

    for i, sample in enumerate(batch):
        t = lengths[i]
        images[i, :t] = sample["image"]
        if t < t_max:
            images[i, t:] = sample["image"][-1:]  # repeat last
        valid[i, :t] = True

    return {
        "image": images,
        "mask": masks,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "valid_mask": valid,
        "video_id": [b["video_id"] for b in batch],
        "has_mask": torch.stack([b["has_mask"] for b in batch], dim=0),
        "crop": torch.stack([b["crop"] for b in batch], dim=0),
        "frame_indices": [b["frame_indices"] for b in batch],
    }


def build_dataloader(
    cfg: DatasetConfig,
    dataset: IRTDataset | None = None,
) -> DataLoader:
    """Build a DataLoader from DatasetConfig."""
    ds = dataset or IRTDataset(cfg)
    collate = pad_collate if cfg.loader.collate == "pad" else stack_collate
    return DataLoader(
        ds,
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.shuffle and cfg.train,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        drop_last=cfg.loader.drop_last,
        collate_fn=collate,
        worker_init_fn=worker_init_fn if cfg.loader.num_workers > 0 else None,
    )
