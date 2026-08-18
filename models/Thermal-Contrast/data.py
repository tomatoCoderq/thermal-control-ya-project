"""Temporal clip → collapsed contrast maps → static (C, H, W) for U-Net."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
SEG = HERE.parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
for p in (SEG, ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.append(sp)

from common.data import INPUT_SIZE, _center_crop, _restrict_videos
from common.split import split_videos
from features import channel_names, collapse_temporal, normalize_contrast
from irt_cfg import load_cfg
from irt_data import IRTDataset
from irt_data.config import AugConfig


class ContrastCropDataset(Dataset):
    """IRTDataset temporal → contrast features (C, H, W) + mask (1, H, W)."""

    def __init__(
        self,
        base: IRTDataset,
        *,
        size: int = INPUT_SIZE,
        preset: str = "combo",
        channels: list[str] | None = None,
    ):
        self.base = base
        self.size = size
        self.channels = channel_names(preset, channels=channels)

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    def __len__(self) -> int:
        return len(self.base)

    @property
    def video_ids(self) -> list[str]:
        return list(self.base.video_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.base[index]
        img = s["image"]  # (T, 1, H, W)
        if img.dim() != 4:
            raise ValueError(f"expected (T,C,H,W), got {tuple(img.shape)}")

        frames = torch.stack(
            [_center_crop(img[t, 0].float(), self.size) for t in range(img.shape[0])],
            dim=0,
        )
        feat = collapse_temporal(frames.numpy(), self.channels)
        feat = normalize_contrast(feat)
        feat_t = torch.from_numpy(np.ascontiguousarray(feat))

        mask = s["mask"]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = _center_crop(mask.float(), self.size)
        mask = (mask > 0).float()
        return feat_t, mask


def build_contrast_loaders(
    cfg_path: str | Path,
    *,
    preset: str = "combo",
    channels: list[str] | None = None,
    size: int = INPUT_SIZE,
    test_every: int = 4,
    batch_size: int | None = None,
    num_workers: int | None = None,
    backend: str = "irt",
    termo_root: str | Path | None = None,
    termo_include: list[str] | None = None,
    clip_len: int = 12,
) -> tuple[DataLoader, DataLoader, ContrastCropDataset, ContrastCropDataset]:
    if backend == "termo":
        return _build_contrast_loaders_termo(
            preset=preset,
            channels=channels,
            size=size,
            test_every=test_every,
            batch_size=batch_size,
            num_workers=num_workers,
            termo_root=termo_root,
            termo_include=termo_include,
            clip_len=clip_len,
        )
    return _build_contrast_loaders_irt(
        cfg_path,
        preset=preset,
        channels=channels,
        size=size,
        test_every=test_every,
        batch_size=batch_size,
        num_workers=num_workers,
    )


class ContrastTermoDataset(Dataset):
    """TermoDataset temporal clip → contrast maps (C,H,W) + mask."""

    def __init__(
        self,
        base,
        *,
        size: int = INPUT_SIZE,
        preset: str = "combo",
        channels: list[str] | None = None,
        clip_len: int = 12,
        random_clip: bool = True,
    ):
        from common.termo_data import TermoTemporalDataset

        self.temporal = TermoTemporalDataset(
            base, clip_len=clip_len, size=size, random_clip=random_clip
        )
        self.channels = channel_names(preset, channels=channels)

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    def __len__(self) -> int:
        return len(self.temporal)

    @property
    def video_ids(self) -> list[str]:
        return list(self.temporal.video_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames, mask = self.temporal[index]
        feat = collapse_temporal(frames[:, 0].numpy(), self.channels)
        feat = normalize_contrast(feat)
        feat_t = torch.from_numpy(np.ascontiguousarray(feat))
        return feat_t, mask


def _build_contrast_loaders_termo(
    *,
    preset: str = "combo",
    channels: list[str] | None = None,
    size: int = INPUT_SIZE,
    test_every: int = 4,
    batch_size: int | None = None,
    num_workers: int | None = None,
    termo_root: str | Path | None = None,
    termo_include: list[str] | None = None,
    clip_len: int = 12,
):
    from common.termo_data import (
        DEFAULT_TERMO_ROOT,
        TermoIndexDataset,
        _restrict_videos,
    )
    from datasets import TermoDataset

    base = TermoIndexDataset(
        TermoDataset(str(termo_root or DEFAULT_TERMO_ROOT), include=termo_include)
    )
    train_ids, test_ids = split_videos(list(base.video_ids), test_every=test_every)
    train_base = _restrict_videos(base, train_ids)
    test_base = _restrict_videos(base, test_ids)

    kw = dict(
        preset=preset, channels=channels, size=size, clip_len=clip_len
    )
    train_ds = ContrastTermoDataset(train_base, random_clip=True, **kw)
    test_ds = ContrastTermoDataset(test_base, random_clip=False, **kw)

    bs = batch_size if batch_size is not None else 2
    nw = num_workers if num_workers is not None else 0
    common: dict = {
        "num_workers": nw,
        "pin_memory": False,
        "persistent_workers": bool(nw),
    }
    if nw > 0:
        common["prefetch_factor"] = 3

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, drop_last=True, **common
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, drop_last=False, **common
    )
    return train_loader, test_loader, train_ds, test_ds


def _build_contrast_loaders_irt(
    cfg_path: str | Path,
    *,
    preset: str = "combo",
    channels: list[str] | None = None,
    size: int = INPUT_SIZE,
    test_every: int = 4,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader, ContrastCropDataset, ContrastCropDataset]:
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

    kw = dict(preset=preset, channels=channels, size=size)
    train_ds = ContrastCropDataset(train_base, **kw)
    test_ds = ContrastCropDataset(test_base, **kw)

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
