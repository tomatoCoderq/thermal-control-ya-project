import os
import numpy as np
from typing import Optional
from config import DatasetConfig
from PIL import Image

from dataclasses import astuple
from scipy.io import loadmat

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class TermoDataset(Dataset):
    def __init__(self, root_dir: str, include: Optional[list[str]] = None, transform: Optional[callable] = None) -> None:
        self.transform = transform
        self.items: list[tuple[str, str, DatasetConfig]] = []

        dataset_dirs = sorted(
            os.path.join(root_dir, name) for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))
            )

        if include:
            dataset_dirs = sorted(dir_path for dir_path in dataset_dirs if os.path.basename(dir_path) in include)

        for dataset_dir in dataset_dirs:
            config_path = os.path.join(dataset_dir, "manifest.yaml")
            config: DatasetConfig = DatasetConfig.from_yaml(config_path)

            mat_files: list[str] = sorted(
                os.path.join(dataset_dir, config.data.path, filename)
                for filename in os.listdir(os.path.join(dataset_dir, config.data.path))
                if filename.endswith(config.data.file_pattern)
                )

            for mat_path in mat_files:
                name: str = os.path.splitext(os.path.basename(mat_path))[0]
                mask_path: str = os.path.join(dataset_dir, config.masks.path, name + config.masks.file_pattern)
                self.items.append((mat_path, mask_path, config))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        mat_path, mask_path, config = self.items[idx % len(self.items)]
        data = loadmat(mat_path)[config.data.mat_key].astype(np.float32)
        mask = np.array(Image.open(mask_path)).astype(np.float32)

        data, mask = self._apply_crop(data, mask, config.crop)

        if self.transform is not None:
            data, mask = self.transform(data, mask)

        data = torch.from_numpy(data).float()
        mask = torch.from_numpy(mask).float()

        if data.ndim == 3:
            data = data.permute(2, 0, 1)
        else:
            data = data.unsqueeze(0)

        data = F.interpolate(
            data.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False
            ).squeeze(0)

        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)

        mean = data.mean()
        std = data.std()

        data = (data - mean) / (std + 1e-8)

        return data, mask



    @staticmethod
    def _apply_crop(data: np.ndarray, mask: np.ndarray, crop) -> tuple[np.ndarray, np.ndarray]:
        x0, x1, y0, y1 = astuple(crop)
        if x0 is None or x1 is None or y0 is None or y1 is None:
            return data, mask

        return data[y0:y1, x0:x1, :1500], mask[y0:y1, x0:x1, ...]