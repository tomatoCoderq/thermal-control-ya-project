from __future__ import annotations

import torch.nn as nn

from config import CFG


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = CFG.classes.n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(CFG.model.dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


def make_model(in_channels: int):
    if CFG.model.name == "small_cnn":
        return SmallCNN(in_channels)
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            f"model.name={CFG.model.name!r} requires the optional package 'timm'"
        ) from exc
    return timm.create_model(
        CFG.model.name,
        pretrained=CFG.model.pretrained,
        in_chans=in_channels,
        num_classes=CFG.classes.n_classes,
    )





