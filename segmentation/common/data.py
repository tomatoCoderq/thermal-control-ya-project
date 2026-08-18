"""Center-crop dataset + train/test loaders (video split)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

_COMMON = Path(__file__).resolve().parent
_SEG = _COMMON.parent
_ROOT = _SEG.parent
# Only repo root on path — never put `_COMMON` there (shadows local `metrics.py`).
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from irt_data import IRTDataset
from irt_data.config import AugConfig, DatasetConfig

from .split import split_videos
from .variants import apply_variant_features

# Spatial input size for U-Net / Attention / Mamba. Keep yaml `crop.size` in sync.
INPUT_SIZE = 256


def _center_crop(t: torch.Tensor, size: int) -> torch.Tensor:
    h, w = t.shape[-2:]
    if h == size and w == size:
        return t
    if h < size or w < size:
        ph, pw = max(0, size - h), max(0, size - w)
        t = torch.nn.functional.pad(
            t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        )
        h, w = t.shape[-2:]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return t[..., y0 : y0 + size, x0 : x0 + size]


class CropDataset(Dataset):
    def __init__(self, base: IRTDataset, size: int = INPUT_SIZE):
        self.base = base
        self.size = size

    def __len__(self) -> int:
        return len(self.base)

    @property
    def video_ids(self) -> list[str]:
        return list(self.base.video_ids)

    def __getitem__(self, index: int):
        s = self.base[index]
        img = s["image"]
        if img.ndim == 4:
            img = img.reshape(-1, *img.shape[-2:]) if img.shape[0] > 1 else img[0]
        mask = s["mask"]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = (mask > 0).float()
        return _center_crop(img.float(), self.size), _center_crop(mask, self.size)


def _restrict_videos(ds: IRTDataset, video_ids: list[str]) -> IRTDataset:
    keep = [v for v in video_ids if v in set(ds.video_ids)]
    if not keep:
        raise ValueError(f"No videos left after split filter (requested {len(video_ids)})")
    ds.video_ids = keep
    spv = max(1, ds.cfg.samples_per_video)
    ds._index = [vid for vid in keep for _ in range(spv)]
    return ds


def build_train_test_loaders(
    cfg: DatasetConfig,
    *,
    size: int = INPUT_SIZE,
    test_every: int = 4,
    batch_size: int | None = None,
    variant: str | None = None,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
    persistent_workers: bool | None = None,
) -> tuple[DataLoader, DataLoader, CropDataset, CropDataset]:
    """Return train_loader, test_loader, train_ds, test_ds."""
    if variant is not None:
        cfg = apply_variant_features(cfg, variant)

    train_cfg = copy.deepcopy(cfg)
    train_cfg.train = True

    # Eval must be deterministic and actually look at the specimen: no augs, one
    # sample per video, and a centred ROI crop instead of a random one.
    test_cfg = copy.deepcopy(cfg)
    test_cfg.train = False
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
    train_base = _restrict_videos(IRTDataset(train_cfg), train_ids)

    train_ds = CropDataset(train_base, size=size)
    test_ds = CropDataset(test_base, size=size)

    bs = batch_size if batch_size is not None else cfg.loader.batch_size
    nw = cfg.loader.num_workers if num_workers is None else num_workers
    persist = bool(nw) if persistent_workers is None else persistent_workers
    common: dict = {
        "num_workers": nw,
        "pin_memory": False,
        "persistent_workers": persist and nw > 0,
    }
    if nw > 0:
        common["prefetch_factor"] = 3 if prefetch_factor is None else prefetch_factor

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        drop_last=cfg.loader.drop_last,
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
