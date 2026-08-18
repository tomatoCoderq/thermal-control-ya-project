"""yaml → irt_data → center-crop 256×256 (thin wrapper over common.data)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEG = HERE.parent
ROOT = HERE.parents[1]
for p in (HERE, SEG, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.data import INPUT_SIZE, CropDataset, build_train_test_loaders  # noqa: F401
from irt_cfg import load_cfg
from irt_data import IRTDataset
from torch.utils.data import DataLoader

# back-compat alias
UNetCropDataset = CropDataset


def make_loader(yaml_path: str | Path, *, train: bool, size: int = INPUT_SIZE):
    """Single-split loader (legacy). Prefer build_train_test_loaders for training."""
    cfg = load_cfg(yaml_path, train=train)
    ds = CropDataset(IRTDataset(cfg), size=size)
    loader = DataLoader(
        ds,
        batch_size=cfg.loader.batch_size,
        shuffle=bool(cfg.loader.shuffle and train),
        num_workers=cfg.loader.num_workers,
        drop_last=cfg.loader.drop_last,
    )
    return cfg, ds, loader
