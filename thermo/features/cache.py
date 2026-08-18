"""Единый precompute+кэш признаков по (kind, domain).

kaggle-кадры уже kaggle-размера; tpu приводим тем же трансформом, что раньше
(обрезка фона сверху/снизу + resize к размеру kaggle), чтобы координаты совпадали.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

from ..config import CFG
from . import build_extractor

TPU_CROP_TB = 60


def _kaggle_hw():
    files = sorted(glob.glob(str(CFG.paths.tsr_kaggle_dir / "*.npy")))
    return tuple(np.load(files[0]).shape[1:]) if files else (256, 320)


def _tpu_crop_resize(a, Hk, Wk):
    import cv2
    a = a[..., TPU_CROP_TB:a.shape[-2] - TPU_CROP_TB, :]
    if a.ndim == 2:
        return cv2.resize(a, (Wk, Hk), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.stack([cv2.resize(a[i], (Wk, Hk), interpolation=cv2.INTER_LINEAR)
                     for i in range(a.shape[0])]).astype(np.float32)


def list_videos(domain: str):
    if domain == "kaggle":
        paths = glob.glob(str(CFG.paths.data_dir / "*.mat"))
        ids = [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]
    elif domain == "tpu":
        paths = glob.glob(str(CFG.paths.tpu_dir / "*.mat"))
        ids = [(os.path.splitext(os.path.basename(p))[0].replace(" ", "_"), p)
               for p in paths]
    else:
        raise ValueError(domain)
    ids = sorted(ids)
    return ids if CFG.train.max_videos is None else ids[:CFG.train.max_videos]


def precompute(domain: str, limit: int | None = None, verbose: bool = True):
    """Посчитать и закэшировать признаки CFG.features для домена."""
    kind = CFG.features.kind
    out_dir = CFG.paths.feature_dir(kind, domain)
    out_dir.mkdir(parents=True, exist_ok=True)
    extract = build_extractor(CFG.features)
    Hk, Wk = _kaggle_hw()

    vids = list_videos(domain)
    if limit:
        vids = vids[:limit]
    for vid, path in vids:
        out = out_dir / f"{vid}.npy"
        if out.exists():
            continue
        feats = extract(path)                      # (C,H,W)
        if domain == "tpu":
            feats = _tpu_crop_resize(feats, Hk, Wk)
        np.save(out, feats)
        if verbose:
            print(f"[{kind}/{domain}] {vid} {feats.shape}")
    return out_dir
