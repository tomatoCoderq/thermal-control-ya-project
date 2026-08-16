"""On-the-fly стандартная аугментация БЕЗ извлечения фич.

Берёт «стандартный» вход (уже готовый массив: кэш-фичи ИЛИ сырой кадр) и на лету
делает ровно стандартный геометрический пайплайн из irt_data:

    вход (H,W,C)  → ROI-crop рамки → resize → синхронная простр. аугментация → тензор

Никакого PPT/TSR-экстракта внутри, ничего не пишется на диск. Маска (если есть)
преобразуется синхронно с картинкой.

Работает и для kaggle, и для tpu — надо лишь подать массив/маску/ROI в ОДНОЙ
системе координат (фичи 256×320 + table_mask; либо сырой кадр + base_mask).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from irt_data.config import ObjectCropConfig, ROI, AugConfig, AugSpec
from irt_data.crops import ObjectCropper
from irt_data.transforms import TransformPipeline

# стандартный набор (как в augmentation, но без grid-shuffle по умолчанию)
DEFAULT_AUG = AugConfig(spatial=[
    AugSpec("HorizontalFlip", {"p": 0.5}),
    AugSpec("VerticalFlip", {"p": 0.5}),
    AugSpec("RandomRotate90", {"p": 0.5}),
])


class StandardAugDataset(Dataset):
    """items: список dict с ключами:
        image: np.ndarray (H,W) или (H,W,C)   — уже готовый вход, БЕЗ экстракта
        mask : np.ndarray (H,W) | None         — синхронно аугментируется
        label: любой скаляр (класс/глубина)    — не трогается
    """

    def __init__(self, items, roi: ROI | None, out_size=(256, 256),
                 aug_cfg: AugConfig | None = None, train=True, pad_mode="reflect"):
        self.items = items
        self.train = train
        self.cropper = ObjectCropper(ObjectCropConfig(
            enabled=roi is not None, roi=roi, output_size=out_size, pad_mode=pad_mode))
        self.aug = TransformPipeline(aug_cfg or DEFAULT_AUG)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = np.asarray(it["image"], dtype=np.float32)
        if img.ndim == 2:
            img = img[..., None]                       # (H,W)->(H,W,1)
        H, W = img.shape[:2]
        box = self.cropper.box_for(H, W, meta=None)

        img = self.cropper.apply_image(img, box)       # crop+resize (H',W',C)
        mask = it.get("mask")
        if mask is not None:
            mask = self.cropper.apply_mask(np.asarray(mask), box)

        if self.train:                                 # аугментация только на train
            img, mask = self.aug.apply_features(img, mask)

        img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
        out = {"image": img_t, "label": it.get("label")}
        if mask is not None:
            out["mask"] = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return out
