"""Высокоуровневый прогон: данные → модель → обучение → метрики.

Параметризуется CFG.train.task (classification|regression) и CFG.features.kind
(tsr|fourier). Единый вход для всех экспериментов на ветках.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CFG
from .data.index import (build_index_kaggle, build_index_tpu, split_by_video,
                         compute_norm_by_domain, compute_target_stats)
from .data.dataset import CropDataset
from .models import build_model
from .losses import Losses
from .optimizers import build_optimizer
from .metrics import RegressionMetrics, Metrics
from .engine import train_loop


def get_device():
    return ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")


def make_loss(task, train_idx):
    if task == "regression":
        return Losses.regression(CFG.regression.loss_name, CFG.regression.huber_beta)
    name = CFG.train.loss_name
    if name == "ce":
        return Losses.ce()
    if name == "label_smooth":
        return Losses.label_smooth()
    if name == "weighted_ce":
        freq = np.array([sum(1 for r in train_idx if r[4] == k)
                         for k in range(CFG.classes.n_classes)], dtype=np.float64)
        w = freq.sum() / (freq + 1e-6)
        return Losses.weighted_ce((w / w.mean()).tolist())
    raise ValueError(f"неизвестный loss: {name}")


def make_loaders(train_idx, test_idx, task):
    norm = compute_norm_by_domain(train_idx)
    for d in {r[5] for r in test_idx} - set(norm):
        norm[d] = compute_norm_by_domain([r for r in test_idx if r[5] == d])[d]
    t_mean, t_std = (compute_target_stats(train_idx) if task == "regression"
                     else (0.0, 1.0))
    train_ds = CropDataset(train_idx, norm, task, train=True,
                           target_mean=t_mean, target_std=t_std,
                           augment=CFG.regression.augment)
    test_ds = CropDataset(test_idx, norm, task, train=False,
                          target_mean=t_mean, target_std=t_std)
    train_ld = DataLoader(train_ds, batch_size=CFG.train.batch_size, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=CFG.train.batch_size, shuffle=False)
    return train_ld, test_ld, t_mean, t_std


def run(train_idx, test_idx, task=None, device=None, tag="run"):
    task = task or CFG.train.task
    device = device or get_device()
    torch.manual_seed(CFG.train.seed)
    np.random.seed(CFG.train.seed)

    train_ld, test_ld, t_mean, t_std = make_loaders(train_idx, test_idx, task)
    model = build_model(task, CFG.features.in_channels).to(device)
    loss_fn = make_loss(task, train_idx).to(device)
    optimizer = build_optimizer(model)

    print(f"[{tag}] task={task} kind={CFG.features.kind} in_ch={CFG.features.in_channels} "
          f"model={CFG.model.name} opt={CFG.optim.name} device={device}")
    train_loop(train_ld, model, loss_fn, optimizer, device, task,
               t_mean, t_std, val_loader=test_ld)

    if task == "regression":
        dom = np.array([r[5] for r in test_idx])
        m = RegressionMetrics.from_loader(model, test_ld, device, t_mean, t_std, domain=dom)
        print("\n=== RegressionMetrics ==="); m.summary()
    else:
        m = Metrics.from_loader(model, test_ld, device)
        print("\n=== Metrics ==="); m.summary()
    return dict(model=model, metrics=m, target_mean=t_mean, target_std=t_std)
