import torch 
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.size(0), -1)
        targets_flat = targets.reshape(targets.size(0), -1).to(probs.dtype)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_score.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, 
                 pos_weight: float | None = None, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = SoftDiceLoss(smooth)

        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
        dice = self.dice(logits, targets)

        return self.bce_weight * bce + self.dice_weight * dice