import torch
import torch.nn as nn

from config import CFG


'''
MODEL_ZOO = [
    "small_cnn",
    "resnet18",
    "resnet34",
    "convnext_nano",
]
тоже самое тут юзалось


'''


class SmallCNN(nn.Module):
    """Compact CNN. n_out=1 → регрессия глубины (вариант B)."""

    def __init__(self, in_channels: int, n_out: int = 1):
        super().__init__()
        x1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        x2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        x3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.features = nn.ModuleList([x1, x2, x3])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.classifier(x)
        return x


class RegressionModel(nn.Module):
    """Обёртка: бэкбон с 1-выходной головой → скаляр глубины (B,).

    Выход squeeze'ится до (B,), чтобы стыковаться с таргетом (B,) в лоссах
    вроде SmoothL1/L1/MSE. Работает и для timm-сетей, и для SmallCNN.

    Задел под вариант C (μ,σ): сделать n_out=2 и вернуть (mu, log_sigma) —
    здесь намеренно не реализуем (по договорённости — только B).
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone(x)          # (B, 1)
        return out.squeeze(-1)          # (B,)


def build_model(in_channels: int, name: str | None = None,
                pretrained: bool | None = None) -> nn.Module:
    """Собрать регрессионную модель (1 выход) по имени из конфига.

    name='small_cnn' → SmallCNN; иначе timm.create_model(num_classes=1).
    """
    name = name or CFG.model.name
    pretrained = CFG.model.pretrained if pretrained is None else pretrained
    print("Current model:", name, f"(pretrained={pretrained})")
    if name == "small_cnn":
        backbone = SmallCNN(in_channels=in_channels, n_out=1)
    else:
        import timm
        backbone = timm.create_model(name, pretrained=pretrained,
                                     in_chans=in_channels, num_classes=1)
    return RegressionModel(backbone)
