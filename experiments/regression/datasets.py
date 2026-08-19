from enum import Enum

import torch
from torch.utils.data import Dataset
import numpy as np

from config import CFG


class Modes(Enum):
    p5 = "p5"
    p5_d1 = "p5+d1"
    p5_d1_d2 = "p5+d1+d2"


def feature_dir_for(domain: str):
    """Каталог TSR-признаков по домену."""
    if domain == "kaggle":
        return CFG.paths.feature_dir
    if domain == "tpu":
        return CFG.paths.tpu_feature_dir
    raise ValueError(f"неизвестный домен: {domain}")


class RegressionDataset(Dataset):
    """Датасет для регрессии глубины (вариант B).

    Элемент индекса: (vid, r0, c0, depth_mm, domain).
      - depth_mm  : float — глубина залегания дефекта, мм (таргет);
      - domain    : 'kaggle' | 'tpu' — определяет и каталог признаков, и
                    поканальную нормировку (домены разные по масштабу).

    Признаки нормируются поканально статистикой СВОЕГО домена (`norm_by_domain`),
    таргет — z-score по train (`target_mean`/`target_std`); в __getitem__ отдаётся
    уже нормированный таргет, денормировка — на стороне метрик/инференса.
    """

    def __init__(self, index, norm_by_domain, target_mean, target_std,
                 mode: Modes, train=True):
        self.index = index
        self.mode = mode
        self.train = train
        self.target_mean = float(target_mean)
        self.target_std = float(target_std) + 1e-8
        # (C,1,1) для бродкаста по (C,H,W), отдельно на каждый домен
        self.norm = {
            d: (np.asarray(m, np.float32).reshape(-1, 1, 1),
                np.asarray(s, np.float32).reshape(-1, 1, 1))
            for d, (m, s) in norm_by_domain.items()
        }
        # признаки грузим один раз на (domain, vid)
        keys = {(r[4], r[0]) for r in index}
        self.feat_cache = {
            (dom, v): np.load(feature_dir_for(dom) / f"{v}.npy")
            for (dom, v) in keys
        }

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        vid, r0, c0, depth_mm, domain = self.index[idx]
        cropped = self._get_crop(domain, vid, r0, c0)
        mean, std = self.norm[domain]
        cropped = (cropped - mean) / std
        moded = self._build_channels(cropped, self.mode)

        y = (float(depth_mm) - self.target_mean) / self.target_std
        return torch.from_numpy(moded).float(), torch.tensor(y).float()

    def _get_crop(self, domain, vid, r0, c0):
        s = CFG.crop.crop
        return self.feat_cache[(domain, vid)][:, r0:r0 + s, c0:c0 + s]

    def _build_channels(self, t_dim, mode: Modes):
        if mode == Modes.p5_d1_d2:
            return np.concatenate(
                [t_dim, np.diff(t_dim, n=1, axis=0), np.diff(t_dim, n=2, axis=0)],
                axis=0)
        if mode == Modes.p5_d1:
            return np.concatenate([t_dim, np.diff(t_dim, n=1, axis=0)], axis=0)
        if mode == Modes.p5:
            return t_dim
        raise ValueError(f"Unknown mode: {mode}")
