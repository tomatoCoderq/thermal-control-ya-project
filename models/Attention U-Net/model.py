"""Attention U-Net for thermal IRT models (6ch features, 256×256)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_SEG = Path(__file__).resolve().parents[1]
if str(_SEG) not in sys.path:
    sys.path.insert(0, str(_SEG))

from common.metrics import SoftDiceLoss  # noqa: F401


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """Upsample + conv. ConvTranspose2d instead of nn.Upsample — MPS-safe backward."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.act = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.act(self.up(x))


class AttentionBlock(nn.Module):
    """Attention gate on skip connections."""

    def __init__(self, F_g, F_l, n_coefficients):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, 1, bias=True),
            nn.BatchNorm2d(n_coefficients),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, 1, bias=True),
            nn.BatchNorm2d(n_coefficients),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, gate, skip_connection):
        psi = self.psi(self.relu(self.W_gate(gate) + self.W_x(skip_connection)))
        return skip_connection * psi


class AttentionUNet(nn.Module):
    """Attention U-Net; ``img_ch=6`` = TSR coeffs (poly_degree=5) or Fourier/PPT bins 1..6."""

    def __init__(self, img_ch: int = 6, output_ch: int = 1):
        super().__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Conv2(self.MaxPool(e1))
        e3 = self.Conv3(self.MaxPool(e2))
        e4 = self.Conv4(self.MaxPool(e3))
        e5 = self.Conv5(self.MaxPool(e4))

        d5 = self.Up5(e5)
        d5 = self.UpConv5(torch.cat((self.Att5(gate=d5, skip_connection=e4), d5), dim=1))

        d4 = self.Up4(d5)
        d4 = self.UpConv4(torch.cat((self.Att4(gate=d4, skip_connection=e3), d4), dim=1))

        d3 = self.Up3(d4)
        d3 = self.UpConv3(torch.cat((self.Att3(gate=d3, skip_connection=e2), d3), dim=1))

        d2 = self.Up2(d3)
        d2 = self.UpConv2(torch.cat((self.Att2(gate=d2, skip_connection=e1), d2), dim=1))

        return self.Conv(d2)
