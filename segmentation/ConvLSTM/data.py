"""Temporal frame dataset for ConvLSTM (mode=temporal, 1ch frames)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
SEG = HERE.parent
ROOT = HERE.parents[1]
for p in (HERE, SEG, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.data import INPUT_SIZE, _center_crop, _restrict_videos  # noqa: F401
from common.split import split_videos
from irt_cfg import load_cfg
from irt_data import IRTDataset
from irt_data.config import AugConfig


class TemporalCropDataset(Dataset):
    """IRTDataset temporal samples → (T, 1, H, W) frames + (1, H, W) mask."""

    def __init__(self, base: IRTDataset, size: int = INPUT_SIZE):
        self.base = base
        self.size = size

    def __len__(self) -> int:
        return len(self.base)

    @property
    def video_ids(self) -> list[str]:
        return list(self.base.video_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.base[index]
        img = s["image"]  # (T, 1, H, W)
        if img.dim() != 4:
            raise ValueError(f"expected temporal tensor (T,C,H,W), got {tuple(img.shape)}")

        frames = torch.stack(
            [_center_crop(img[t].float(), self.size) for t in range(img.shape[0])],
            dim=0,
        )
        mask = s["mask"]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = _center_crop(mask.float(), self.size)
        mask = (mask > 0).float()
        return frames, mask


def build_temporal_loaders(
    cfg_path: str | Path,
    *,
    size: int = INPUT_SIZE,
    test_every: int = 4,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader, TemporalCropDataset, TemporalCropDataset]:
    """Train/test loaders for ConvLSTM (video-level split, no frame leakage)."""
    train_cfg = load_cfg(cfg_path, train=True)
    test_cfg = load_cfg(cfg_path, train=False)
    test_cfg.loader.shuffle = False
    test_cfg.augs = AugConfig()
    test_cfg.samples_per_video = 1
    if test_cfg.crop.strategy.startswith("roi_"):
        test_cfg.crop.strategy = "roi_center"
    elif test_cfg.crop.strategy == "random":
        test_cfg.crop.strategy = "center"

    test_base = IRTDataset(test_cfg)
    train_ids, test_ids = split_videos(list(test_base.video_ids), test_every=test_every)
    _restrict_videos(test_base, test_ids)
    train_base = _restrict_videos(IRTDataset(copy.deepcopy(train_cfg)), train_ids)

    train_ds = TemporalCropDataset(train_base, size=size)
    test_ds = TemporalCropDataset(test_base, size=size)

    bs = batch_size if batch_size is not None else train_cfg.loader.batch_size
    nw = train_cfg.loader.num_workers if num_workers is None else num_workers
    common: dict = {
        "num_workers": nw,
        "pin_memory": False,
        "persistent_workers": bool(nw),
    }
    if nw > 0:
        common["prefetch_factor"] = 3

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        drop_last=train_cfg.loader.drop_last,
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, test_loader, train_ds, test_ds
