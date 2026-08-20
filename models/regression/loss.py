import torch
import torch.nn as nn
import torch.nn.functional as F


def _distance_matrix(n_classes: int, kind: str = "quadratic") -> torch.Tensor:
    idx = torch.arange(n_classes, dtype=torch.float32)
    d = (idx[:, None] - idx[None, :]).abs()
    if kind == "quadratic":
        d = d ** 2
        denom = (n_classes - 1) ** 2
    else:
        denom = (n_classes - 1)
    return d / max(denom, 1)


class CostSensitiveCE(nn.Module):
    def __init__(self, n_classes: int, kind: str = "linear", alpha: float = 1.0,
                 class_weight: list[float] | None = None) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.register_buffer("cost", _distance_matrix(n_classes, kind))
        if class_weight is not None:
            self.register_buffer("cw", torch.as_tensor(class_weight, dtype=torch.float32))
        else:
            self.cw = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.cw, reduction="none")
        p = F.softmax(logits, dim=1)
        exp_cost = (p * self.cost[target]).sum(dim=1)
        return (ce + self.alpha * exp_cost).mean()


class QWKLoss(nn.Module):
    def __init__(self, n_classes: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.n = n_classes
        self.eps = eps
        self.register_buffer("W", _distance_matrix(n_classes, "quadratic"))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=1)
        oh = F.one_hot(target, self.n).float()
        N = logits.size(0)
        O = oh.t() @ p
        hist_t = oh.sum(dim=0)
        hist_p = p.sum(dim=0)
        E = torch.outer(hist_t, hist_p) / N
        num = (self.W * O).sum()
        den = (self.W * E).sum()
        return num / (den + self.eps)
