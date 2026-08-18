"""TermoDataset helpers for frame-based models models."""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from datasets import TermoDataset

from .data import INPUT_SIZE, _center_crop
from .split import split_videos

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TERMO_ROOT = _REPO_ROOT / "datasets" / "datasets_list"


def _video_id(mat_path: str) -> str:
    return Path(mat_path).stem


class TermoIndexDataset(Dataset):
    """TermoDataset + stable video ids for train/test split."""

    def __init__(self, base: TermoDataset):
        self.base = base
        self.video_ids = [_video_id(p) for p, _, _ in base.items]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        data, mask = self.base[index]
        return data, mask, self.video_ids[index % len(self.video_ids)]


def _restrict_videos(ds: TermoIndexDataset, keep: list[str]) -> TermoIndexDataset:
    keep_set = set(keep)
    indices = [i for i, vid in enumerate(ds.video_ids) if vid in keep_set]
    if not indices:
        raise ValueError(f"No TermoDataset videos left after filter (requested {len(keep)})")
    sub = TermoDataset.__new__(TermoDataset)
    sub.transform = ds.base.transform
    sub.standard_size = ds.base.standard_size
    sub.items = [ds.base.items[i] for i in indices]
    return TermoIndexDataset(sub)


class TermoTemporalDataset(Dataset):
    """(C,H,W) Termo sample → clip (T,1,H,W) + mask (1,H,W)."""

    def __init__(
        self,
        base: TermoIndexDataset,
        *,
        clip_len: int = 12,
        size: int = INPUT_SIZE,
        random_clip: bool = True,
    ):
        self.base = base
        self.clip_len = clip_len
        self.size = size
        self.random_clip = random_clip

    @property
    def video_ids(self) -> list[str]:
        return list(self.base.video_ids)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data, mask, _vid = self.base[index]
        if data.dim() != 3:
            raise ValueError(f"TermoDataset data must be (C,H,W), got {tuple(data.shape)}")

        t_total = data.shape[0]
        clip_len = min(self.clip_len, t_total)
        if self.random_clip and t_total > clip_len:
            t0 = random.randint(0, t_total - clip_len)
        else:
            t0 = max(0, (t_total - clip_len) // 2)
        clip = data[t0 : t0 + clip_len]  # (T,H,W)
        frames = torch.stack(
            [_center_crop(clip[t].float(), self.size).unsqueeze(0) for t in range(clip.shape[0])],
            dim=0,
        )
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = _center_crop(mask.float(), self.size)
        mask = (mask > 0).float()
        return frames, mask


def build_termo_temporal_loaders(
    *,
    root_dir: str | Path = DEFAULT_TERMO_ROOT,
    include: list[str] | None = None,
    clip_len: int = 12,
    size: int = INPUT_SIZE,
    test_every: int = 4,
    batch_size: int = 2,
    num_workers: int = 0,
    transform=None,
) -> tuple[DataLoader, DataLoader, TermoTemporalDataset, TermoTemporalDataset]:
    """Train/test loaders from TermoDataset (video-level split)."""
    base = TermoIndexDataset(
        TermoDataset(str(root_dir), include=include, transform=transform)
    )
    train_ids, test_ids = split_videos(list(base.video_ids), test_every=test_every)
    train_base = _restrict_videos(base, train_ids)
    test_base = _restrict_videos(base, test_ids)

    train_ds = TermoTemporalDataset(
        train_base, clip_len=clip_len, size=size, random_clip=True
    )
    test_ds = TermoTemporalDataset(
        test_base, clip_len=clip_len, size=size, random_clip=False
    )

    common: dict = {
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": bool(num_workers),
    }
    if num_workers > 0:
        common["prefetch_factor"] = 3

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, test_loader, train_ds, test_ds
