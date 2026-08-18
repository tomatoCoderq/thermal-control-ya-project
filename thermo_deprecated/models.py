"""Модели: SmallCNN + фабрика build_model(task).

regression     → 1 выход, squeeze до (B,);
classification → n_classes выходов (логиты).
"""
from __future__ import annotations

import torch.nn as nn

from .config import CFG


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, n_out: int):
        super().__init__()
        self.features = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)),
            nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)),
            nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)),
        ])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, n_out))

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return self.classifier(x)


class _SqueezeHead(nn.Module):
    """Обёртка регрессии: (B,1) -> (B,)."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x).squeeze(-1)


def build_model(task: str, in_channels: int, name: str | None = None,
                pretrained: bool | None = None):
    n_out = 1 if task == "regression" else CFG.classes.n_classes
    name = name or CFG.model.name
    pretrained = CFG.model.pretrained if pretrained is None else pretrained

    if name == "small_cnn":
        backbone = SmallCNN(in_channels=in_channels, n_out=n_out)
    else:
        import timm
        backbone = timm.create_model(name, pretrained=pretrained,
                                     in_chans=in_channels, num_classes=n_out)
    return _SqueezeHead(backbone) if task == "regression" else backbone
