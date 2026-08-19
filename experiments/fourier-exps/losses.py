from __future__ import annotations

import torch
import torch.nn as nn


def make_loss(name: str, n_classes: int, train_labels: list[int]):
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "label_smooth":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    if name == "weighted_ce":
        counts = torch.bincount(torch.tensor(train_labels), minlength=n_classes).float()
        weights = counts.sum() / counts.clamp_min(1)
        weights = weights / weights.mean()
        return nn.CrossEntropyLoss(weight=weights)
    raise ValueError(f"unknown loss: {name}")





