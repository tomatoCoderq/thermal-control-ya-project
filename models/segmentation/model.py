import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x)
        pooled = self.max_pool(skip)
        return pooled, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.block = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.transpose(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNetModel(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 1) -> None:
        super().__init__()

        self.enc_block1 = EncoderBlock(in_channels, 64)
        self.enc_block2 = EncoderBlock(64, 128)
        self.enc_block3 = EncoderBlock(128, 256)
        self.enc_block4 = EncoderBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.dec_block1 = DecoderBlock(1024, 512)
        self.dec_block2 = DecoderBlock(512, 256)
        self.dec_block3 = DecoderBlock(256, 128)
        self.dec_block4 = DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip1 = self.enc_block1(x)
        x, skip2 = self.enc_block2(x)
        x, skip3 = self.enc_block3(x)
        x, skip4 = self.enc_block4(x)

        x = self.bottleneck(x)

        x = self.dec_block1(x, skip4)
        x = self.dec_block2(x, skip3)
        x = self.dec_block3(x, skip2)
        x = self.dec_block4(x, skip1)

        return self.out(x)