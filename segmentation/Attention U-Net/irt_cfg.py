"""dataset.yaml → irt_data.DatasetConfig (paths relative to repo root)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UNET_YAML = HERE.parent / "U-Net" / "dataset.yaml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from irt_data.config import DatasetConfig

DEFAULT_YAML = UNET_YAML if UNET_YAML.exists() else HERE / "dataset.yaml"


def load_cfg(yaml_path: str | Path | None = None, *, train: bool | None = None) -> DatasetConfig:
    path = Path(yaml_path) if yaml_path is not None else DEFAULT_YAML
    cfg = DatasetConfig.from_yaml(path)

    def abs_(p: str | None) -> str | None:
        if not p:
            return p
        pp = Path(p)
        return str(pp if pp.is_absolute() else (ROOT / pp).resolve())

    for src in cfg.sources:
        src.root = abs_(src.root)  # type: ignore[assignment]
        src.masks = abs_(src.masks)  # type: ignore[assignment]
    cfg.cache_dir = abs_(cfg.cache_dir)  # type: ignore[assignment]
    cfg.features.cache_dir = abs_(cfg.features.cache_dir)  # type: ignore[assignment]

    if train is not None:
        cfg.train = train
        if not train:
            cfg.loader.shuffle = False
            cfg.augs.spatial = []
    return cfg
