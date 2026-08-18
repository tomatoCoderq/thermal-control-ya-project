"""Mamba-UNet (VSSM) wrapper for thermal binary models."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_SEG = Path(__file__).resolve().parents[1]
if str(_SEG) not in sys.path:
    sys.path.insert(0, str(_SEG))

from common.metrics import SoftDiceLoss  # noqa: F401
from mamba_sys import VSSM


class MambaUNet(nn.Module):
    """VMamba / Mamba-UNet for 256×256 IRT features (TSR or Fourier, 6 channels)."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 1,
        *,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (2, 2, 2, 2),
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.net = VSSM(
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=num_classes,
            depths=list(depths),
            dims=embed_dim,
            drop_path_rate=drop_path_rate,
            final_upsample="expand_first",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
