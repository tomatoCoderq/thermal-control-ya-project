import torch
import torch.nn as nn

from .model import patchify


class MAEReconstructionLoss(nn.Module):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, pred: torch.Tensor, imgs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = patchify(imgs, self.patch_size)
        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)
        return (loss_per_patch * mask).sum() / mask.sum()