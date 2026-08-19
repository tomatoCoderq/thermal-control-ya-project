"""Turn `TermoDataset` videos into cached `(NUM_CHANNELS, H, W)` U-Net inputs.

Channel extraction is deterministic, so it runs once per video and the result is
cached on disk. Reading a 2000-frame `.mat` and collapsing it costs about half a
second; the cached channel stack loads instantly on the next run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from paths import CACHE_DIR, DATASETS_ROOT

from channels import DEFAULT_PARAMS, NUM_CHANNELS, ChannelParams, extract_channels
from common.split import split_videos
from datasets import TermoDataset


def video_id(mat_path: str) -> str:
    """`.../dataset_tpu/data/sample3.mat` → `sample3`."""
    return Path(mat_path).stem


def source_name(mat_path: str) -> str:
    """`.../dataset_tpu/data/sample3.mat` → `dataset_tpu`."""
    return Path(mat_path).parents[1].name


class ContrastDataset(Dataset):
    """A subset of `TermoDataset` served as extracted channels instead of raw video."""

    def __init__(
        self,
        source: TermoDataset,
        indices: list[int],
        *,
        params: ChannelParams = DEFAULT_PARAMS,
        cache_dir: Path | None = CACHE_DIR,
    ) -> None:
        self.source = source
        self.indices = list(indices)
        self.params = params
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def num_channels(self) -> int:
        return NUM_CHANNELS

    @property
    def video_ids(self) -> list[str]:
        return [video_id(self.source.items[i][0]) for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def _cache_path(self, index: int) -> Path | None:
        if self.cache_dir is None:
            return None
        mat_path = self.source.items[index][0]
        return self.cache_dir / f"{source_name(mat_path)}__{video_id(mat_path)}__{self.params.key}.npy"

    def channels_and_mask(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracted channels and binary mask for a source index."""
        cache_path = self._cache_path(index)
        if cache_path is not None and cache_path.is_file():
            stored = np.load(cache_path)
            return torch.from_numpy(stored[:NUM_CHANNELS]), torch.from_numpy(stored[NUM_CHANNELS:])

        video, mask = self.source[index]
        channels = extract_channels(video, self.params)
        mask = (mask > 0).float().unsqueeze(0)
        if cache_path is not None:
            np.save(cache_path, torch.cat([channels, mask], dim=0).numpy())
        return channels, mask

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        channels, mask = self.channels_and_mask(self.indices[position])
        return channels.contiguous(), mask.contiguous()

    def precompute(self) -> None:
        """Fill the cache for every video in this subset."""
        for position in tqdm(range(len(self)), desc="extract channels", leave=False):
            self.channels_and_mask(self.indices[position])


def build_datasets(
    *,
    root: Path | str = DATASETS_ROOT,
    include: list[str] | None = None,
    params: ChannelParams = DEFAULT_PARAMS,
    test_every: int = 4,
    cache_dir: Path | None = CACHE_DIR,
) -> tuple[ContrastDataset, ContrastDataset]:
    """Train/test split at video level, so no video contributes to both sides."""
    source = TermoDataset(root_dir=str(root), include=include)
    ids = [video_id(path) for path, _, _ in source.items]
    train_ids, test_ids = split_videos(ids, test_every=test_every)

    train_index = [i for i, vid in enumerate(ids) if vid in set(train_ids)]
    test_index = [i for i, vid in enumerate(ids) if vid in set(test_ids)]
    if not train_index or not test_index:
        raise ValueError(f"empty split: {len(train_index)} train / {len(test_index)} test videos")

    shared = dict(params=params, cache_dir=cache_dir)
    return (
        ContrastDataset(source, train_index, **shared),
        ContrastDataset(source, test_index, **shared),
    )


def build_loaders(
    *,
    root: Path | str = DATASETS_ROOT,
    include: list[str] | None = None,
    params: ChannelParams = DEFAULT_PARAMS,
    test_every: int = 4,
    batch_size: int = 4,
    num_workers: int = 0,
    cache_dir: Path | None = CACHE_DIR,
    precompute: bool = True,
) -> tuple[DataLoader, DataLoader, ContrastDataset, ContrastDataset]:
    train_ds, test_ds = build_datasets(
        root=root,
        include=include,
        params=params,
        test_every=test_every,
        cache_dir=cache_dir,
    )
    if precompute:
        train_ds.precompute()
        test_ds.precompute()

    loader_kwargs: dict = {
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False, **loader_kwargs
    )
    return train_loader, test_loader, train_ds, test_ds
