import torch
import torch.nn as nn
import torch.nn.functional as F


def _distance_matrix(n_classes: int, kind: str = "quadratic") -> torch.Tensor:
    """Матрица ординальных «стоимостей» C[i,j] по расстоянию между классами.

    kind='linear'    → |i-j|,       нормировка на (n-1)
    kind='quadratic' → (i-j)^2,     нормировка на (n-1)^2
    Значения в ~[0,1], диагональ = 0.
    """
    idx = torch.arange(n_classes, dtype=torch.float32)
    d = (idx[:, None] - idx[None, :]).abs()
    if kind == "quadratic":
        d = d ** 2
        denom = (n_classes - 1) ** 2
    else:
        denom = (n_classes - 1)
    return d / max(denom, 1)


class CostSensitiveCE(nn.Module):
    """Cost-sensitive (distance-weighted) cross-entropy.

    К обычному CE добавляется ожидаемая ординальная стоимость предсказания:
        L = CE(logits, y) + alpha * E_{j~softmax(logits)}[cost(y, j)],
    где cost(y, j) — |y-j| (kind='linear') или (y-j)^2 (kind='quadratic').
    Ошибка «через несколько классов» штрафуется сильнее, чем в соседний класс.
    """

    def __init__(self, n_classes: int, kind: str = "linear", alpha: float = 1.0,
                 class_weight=None):
        super().__init__()
        self.alpha = float(alpha)
        self.register_buffer("cost", _distance_matrix(n_classes, kind))
        if class_weight is not None:
            self.register_buffer("cw", torch.as_tensor(class_weight, dtype=torch.float32))
        else:
            self.cw = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.cw, reduction="none")   # (B,)
        p = F.softmax(logits, dim=1)                                             # (B,C)
        exp_cost = (p * self.cost[target]).sum(dim=1)                            # (B,)
        return (ce + self.alpha * exp_cost).mean()


class QWKLoss(nn.Module):
    """Quadratic Weighted Kappa loss — дифференцируемая аппроксимация 1 - QWK.

    Каппа Коэна с квадратичными весами W[i,j]=(i-j)^2 считается по «мягкой»
    матрице ошибок (softmax вместо argmax), поэтому дифференцируема:
        kappa = 1 - sum(W*O) / sum(W*E),   loss = sum(W*O)/sum(W*E) = 1 - kappa.
    Минимизация loss → kappa к 1. O — наблюдаемая (soft) матрица, E — ожидаемая.
    """

    def __init__(self, n_classes: int, eps: float = 1e-6):
        super().__init__()
        self.n = n_classes
        self.eps = eps
        self.register_buffer("W", _distance_matrix(n_classes, "quadratic"))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=1)                       # (B,C) мягкие вероятности
        oh = F.one_hot(target, self.n).float()             # (B,C)
        N = logits.size(0)
        O = oh.t() @ p                                     # (C,C) наблюдаемая
        hist_t = oh.sum(dim=0)                             # (C,) частоты истинных
        hist_p = p.sum(dim=0)                              # (C,) «частоты» предсказаний
        E = torch.outer(hist_t, hist_p) / N                # (C,C) ожидаемая
        num = (self.W * O).sum()
        den = (self.W * E).sum()
        return num / (den + self.eps)                      # = 1 - kappa


class Losses():
    """Фабрика лоссов для обучения."""

    @staticmethod
    def ce():
        return torch.nn.CrossEntropyLoss()

    @staticmethod
    def weighted_ce(weights: list[float]):
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        return torch.nn.CrossEntropyLoss(weight=weights_tensor)

    @staticmethod
    def label_smooth(epsilon: float = 0.1):
        return torch.nn.CrossEntropyLoss(label_smoothing=epsilon)

    @staticmethod
    def cost_sensitive_ce(n_classes: int, kind: str = "linear",
                          alpha: float = 1.0, weights=None):
        """Distance-weighted CE (штраф ∝ ординальному расстоянию ошибки)."""
        return CostSensitiveCE(n_classes, kind=kind, alpha=alpha, class_weight=weights)

    @staticmethod
    def qwk(n_classes: int):
        """Quadratic Weighted Kappa loss."""
        return QWKLoss(n_classes)

    # ── регрессия глубины (вариант B) ────────────────────────────────────────
    @staticmethod
    def smooth_l1(beta: float = 1.0):
        """SmoothL1 (Huber). beta — порог перехода L2→L1 (в нормир. единицах)."""
        return nn.SmoothL1Loss(beta=beta)

    @staticmethod
    def l1():
        """MAE-лосс (робастный к выбросам)."""
        return nn.L1Loss()

    @staticmethod
    def mse():
        """MSE-лосс (сильнее штрафует большие ошибки)."""
        return nn.MSELoss()

    @staticmethod
    def regression(name: str = "smooth_l1", beta: float = 1.0):
        """Фабрика регрессионного лосса по имени из конфига."""
        if name == "smooth_l1":
            return Losses.smooth_l1(beta)
        if name == "l1":
            return Losses.l1()
        if name == "mse":
            return Losses.mse()
        raise ValueError(f"неизвестный regression loss: {name}")
