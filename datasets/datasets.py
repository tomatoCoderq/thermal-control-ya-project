import os
import numpy as np
from typing import Optional
from config import DatasetConfig
from PIL import Image

from dataclasses import astuple
from scipy.io import loadmat, whosmat

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class TermoDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            include: Optional[list[str]] = None,
            transform: Optional[callable] = None,
            standard_size: tuple[int, int] = (256, 256)
            ) -> None:
        
        self.transform = transform
        self.standard_size = standard_size
        self.items: list[tuple[str, str, DatasetConfig]] = []

        dataset_dirs = sorted(
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
            )

        if include:
            dataset_dirs = sorted(
                dir_path for dir_path in dataset_dirs
                if os.path.basename(dir_path) in include
                )

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
        mat_path, mask_path, config = self.items[idx]
        data = loadmat(mat_path)[config.data.mat_key].astype(np.float32)
        mask = np.array(Image.open(mask_path)).astype(np.float32)

        data, mask = self._apply_crop(data, mask, config.crop)

        data = np.transpose(data, (2, 0, 1))              # (H,W,T) -> (T,H,W), под трансформы
        if self.transform is not None:
            data, mask = self.transform(data, mask)

        data = torch.from_numpy(np.ascontiguousarray(data)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()

        data = F.interpolate(
            data.unsqueeze(0), size=self.standard_size, mode="bilinear", align_corners=False
            ).squeeze(0)
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0), size=self.standard_size, mode="nearest"
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


class TermoOversampledDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            include: Optional[list[str]] = None,
            transform: Optional[callable] = None,
            standard_size: tuple[int, int] = (256, 256),
            mag_coeff: float = 1.0
            ) -> None:
        
        self.transform = transform
        self.standard_size = standard_size
        self.mag_coeff = mag_coeff
        self.items: list[tuple[str, str, DatasetConfig]] = []

        dataset_dirs = sorted(
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
            )

        if include:
            dataset_dirs = sorted(
                dir_path for dir_path in dataset_dirs
                if os.path.basename(dir_path) in include
                )

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
        return len(self.items) * self.mag_coeff

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        mat_path, mask_path, config = self.items[idx % len(self.items)]
        data = loadmat(mat_path)[config.data.mat_key].astype(np.float32)
        mask = np.array(Image.open(mask_path)).astype(np.float32)

        data, mask = self._apply_crop(data, mask, config.crop)

        data = np.transpose(data, (2, 0, 1))              # (H,W,T) -> (T,H,W), под трансформы
        if self.transform is not None and idx // len(self.items) != 0:
            data, mask = self.transform(data, mask)

        data = torch.from_numpy(np.ascontiguousarray(data)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()

        data = F.interpolate(
            data.unsqueeze(0), size=self.standard_size, mode="bilinear", align_corners=False
            ).squeeze(0)
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0), size=self.standard_size, mode="nearest"
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


class TermoFrameDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            include: Optional[list[str]] = None,
            transform: Optional[callable] = None,
            standard_size: tuple[int, int] = (256, 256),
            ) -> None:
        
        self.transform = transform
        self.standard_size = standard_size
        self.items: list[tuple[str, str, DatasetConfig, int]] = []
        self._data_cache: dict[str, np.ndarray] = {}

        dataset_dirs = sorted(
            os.path.join(root_dir, name) for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        )

        if include:
            dataset_dirs = sorted(d for d in dataset_dirs if os.path.basename(d) in include)

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

                info = whosmat(mat_path)
                shape = next(s for var_name, s, _ in info if var_name == config.data.mat_key)
                n_frames = shape[-1]

                for frame_idx in range(n_frames):
                    self.items.append((mat_path, mask_path, config, frame_idx))

    def __len__(self) -> int:
        return len(self.items)

    def _load_full_data(self, mat_path: str, mat_key: str) -> np.ndarray:
        if mat_path not in self._data_cache:
            self._data_cache[mat_path] = loadmat(mat_path)[mat_key].astype(np.float32)
        return self._data_cache[mat_path]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        mat_path, mask_path, config, frame_idx = self.items[idx]

        full_data = self._load_full_data(mat_path, config.data.mat_key)
        frame = full_data[:, :, frame_idx]

        mask = np.array(Image.open(mask_path)).astype(np.float32)

        frame, mask = self._apply_crop(frame, mask, config.crop)

        frame = frame[None]                               # (H,W) -> (1,H,W), под трансформы
        if self.transform is not None:
            frame, mask = self.transform(frame, mask)

        frame = torch.from_numpy(np.ascontiguousarray(frame)).float()   # (1,H,W)
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()
        frame = F.interpolate(
            frame.unsqueeze(0), size=self.standard_size, mode="bilinear", align_corners=False
        ).squeeze(0)
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0), size=self.standard_size, mode="nearest"
        ).squeeze(0).squeeze(0)

        mean = frame.mean()
        std = frame.std()
        frame = (frame - mean) / (std + 1e-8)

        return frame, mask

    @staticmethod
    def _apply_crop(frame: np.ndarray, mask: np.ndarray, crop) -> tuple[np.ndarray, np.ndarray]:
        x0, x1, y0, y1 = astuple(crop)
        if x0 is None or x1 is None or y0 is None or y1 is None:
            return frame, mask

        return frame[y0:y1, x0:x1], mask[y0:y1, x0:x1, ...]