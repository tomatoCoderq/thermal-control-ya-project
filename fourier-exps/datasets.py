from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from config import CFG


class FourierDataset(Dataset):
    """Crops cached Fourier phasegrams using the same index as p5 training."""

    def __init__(self, index, mean, std, train: bool = True):
        self.index = list(index)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
        self.train = train
        videos = sorted({row[0] for row in self.index})
        self.feature_cache = {
            video: np.load(CFG.paths.feature_dir / f"{video}.npy", mmap_mode="r")
            for video in videos
        }

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        video, r0, c0, label = self.index[idx]
        size = CFG.crop.crop
        crop = np.asarray(
            self.feature_cache[video][:, r0:r0 + size, c0:c0 + size],
            dtype=np.float32,
        )
        crop = (crop - self.mean) / self.std
        return torch.from_numpy(crop.copy()).float(), torch.tensor(label).long()





