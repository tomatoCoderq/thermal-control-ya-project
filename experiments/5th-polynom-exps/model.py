import torch.nn as nn

from config import CFG


'''
вот эти модели юзаются в экспериментах
MODEL_ZOO = [
    "small_cnn",
    "resnet18",
    "resnet34",
    "convnext_nano",
]


'''


class SmallCNN(nn.Module):
    """Compact CNN"""

    def __init__(self, in_channels: int, n_cls: int = CFG.classes.n_classes):
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
            nn.Linear(64, n_cls),
        )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.classifier(x)
        return x
