"""ConvLSTM encoder–decoder for thermal frame sequences (1ch → binary mask).

Architecture (based on Shi et al. ConvLSTM + U-Net-style skips):
  - Shared CNN encoder per frame → spatial feature maps (1/4 resolution bottleneck)
  - ConvLSTM propagates temporal state at bottleneck resolution
  - Decoder upsamples with skip connections from the **last** frame encoder
  - Output: single mask logits (B, 1, H, W) — mask is static per video, temporal
    context improves contrast of heating/cooling defects.

Input:  (B, T, C, H, W)  — C=1 for raw thermal frames
Output: (B, 1, H, W)     — models logits (no sigmoid; use BCEWithLogits)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """One ConvLSTM step (Shi et al., 2015). Preserves (H, W) spatial structure."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=pad,
            bias=bias,
        )

    def _init_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        z = x.new_zeros(b, self.hidden_dim, h, w)
        return z, z

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, c = self._init_state(x) if state is None else state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, (h, c)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """ConvTranspose upsample + conv fusion with skip."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.conv(torch.cat([x, skip], dim=1))


class ConvLSTMSegNet(nn.Module):
    """Temporal models on raw thermal frame clips.

    Parameters
    ----------
    in_channels : int
        1 for grayscale IRT frames.
    base : int
        Base channel width of the encoder (32 → 64 → 128).
    hidden_dim : int
        ConvLSTM hidden state channels at 1/4 resolution.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2, stride=2)
        self.enc3 = ConvBlock(c2, c3, stride=2)

        self.convlstm = ConvLSTMCell(c3, hidden_dim, kernel_size=3)

        self.dec2 = UpBlock(hidden_dim, c2, c2)
        self.dec1 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def encode_frame(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        return e1, e2, e3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) → logits (B, 1, H, W)."""
        if x.dim() != 5:
            raise ValueError(f"expected (B,T,C,H,W), got {tuple(x.shape)}")

        b, t, _, _, _ = x.shape
        state: tuple[torch.Tensor, torch.Tensor] | None = None
        skip1 = skip2 = None

        for ti in range(t):
            e1, e2, e3 = self.encode_frame(x[:, ti])
            hidden, state = self.convlstm(e3, state)
            if ti == t - 1:
                skip1, skip2 = e1, e2

        assert skip1 is not None and skip2 is not None
        d2 = self.dec2(hidden, skip2)
        d1 = self.dec1(d2, skip1)
        return self.head(d1)
