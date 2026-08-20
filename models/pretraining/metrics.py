import torch


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    mse = ((pred - target) ** 2).mean()
    return 10 * torch.log10(max_val ** 2 / (mse + eps))


def compute_visible_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    visible = 1 - mask
    loss_per_patch = ((pred - target) ** 2).mean(dim=-1)
    return (loss_per_patch * visible).sum() / visible.sum()