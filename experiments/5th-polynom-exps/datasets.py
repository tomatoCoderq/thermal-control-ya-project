from enum import Enum
import os, glob

import torch
from torch.utils.data import Dataset
import numpy as np
from config import CFG


class Modes(Enum):
    p5 = "p5"
    p5_d1 = "p5+d1"
    p5_d1_d2 = "p5+d1+d2"


class DatasetAnyChannels(Dataset):
    def __init__(self, index, mean, std, mode: Modes, train=True):
        self.index = index
        # mean/std приводим к форме (C,1,1) для бродкаста по (C,H,W)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
        self.mode = mode
        self.train = train
        # признаки грузим один раз (а не на каждый __getitem__)
        vids = sorted({r[0] for r in index})
        self.feat_cache = {v: np.load(CFG.paths.feature_dir / f"{v}.npy") for v in vids}

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        vid, r0, c0, lab = self.index[idx]
        cropped = self._get_crop(vid, r0, c0)
        cropped = (cropped - self.mean) / self.std
        moded = self._build_channels(cropped, self.mode)
        
        '''добавить сюда аугментациии при трейне'''
        
        return torch.from_numpy(moded).float(), torch.tensor(lab).long()
        
        
    def _list_videos(self):
        ids = [os.path.splitext(os.path.basename(p))[0]
           for p in glob.glob(os.path.join(CFG.paths.data_dir, "*.mat"))]
        ids = sorted(ids)
        return ids 
    
    def _get_crop(self, vid, r0, c0):
        s = CFG.crop.crop
        return self.feat_cache[vid][:, r0:r0 + s, c0:c0 + s]
    
    def _build_channels(self, t_dim, mode: Modes):
        if mode == Modes.p5_d1_d2:
            return np.concatenate([t_dim, np.diff(t_dim, n=1, axis=0), np.diff(t_dim, n=2, axis=0)], axis=0)
        elif mode == Modes.p5_d1:
            return np.concatenate([t_dim, np.diff(t_dim, n=1, axis=0)], axis=0)
        elif mode == Modes.p5:
            return t_dim
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        
    
    
        