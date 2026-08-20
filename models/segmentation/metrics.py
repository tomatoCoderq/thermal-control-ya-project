import torch


def compute_iou(
        logits: torch.Tensor, targets: torch.Tensor, 
        threshold: float = 0.5, eps: float = 1e-7,
        ) -> torch.Tensor:
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).clamp(0, 1).sum(dim=(1, 2, 3))

    return ((intersection + eps) / (union + eps)).mean()


def compute_dice(
        logits: torch.Tensor, targets: torch.Tensor, 
        threshold: float = 0.5, eps: float = 1e-7,
        ) -> torch.Tensor:
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    
    return ((2.0 * intersection + eps) / (union + eps)).mean()