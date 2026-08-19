"""Оптимизаторы: Adam/AdamW и Muon (гибрид Muon + Adam).

https://kellerjordan.github.io/posts/muon/
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer

from config import CFG


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Приближённая ортогонализация G (мягкая замена U V^T из SVD)."""
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    work_dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(work_dtype)
    if X.size(-2) > X.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon: SGD-momentum с ортогонализацией обновления (для 2D+ весов).

    Свёрточные ядра (ndim=4) решейпятся к (out_channels, -1). Обновление
    масштабируется на sqrt(max(1, rows/cols)), чтобы согласовать RMS с Adam.
    Одно-устройственная версия (без distributed).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 weight_decay=0.0, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            mom = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - mom)                       # EMA градиента
                g = g.lerp_(buf, mom) if group["nesterov"] else buf

                g2d = g.reshape(g.size(0), -1) if g.ndim > 2 else g
                u = zeropower_via_newtonschulz5(g2d, group["ns_steps"])
                u = u.reshape(p.shape)

                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(u, alpha=-group["lr"] * scale)
        return loss


class CombinedOptimizer:
    """Обёртка над несколькими оптимизаторами: единые step/zero_grad/state_dict.

    Нужна, чтобы train.py вызывал один `.zero_grad()`/`.step()` независимо от
    того, один это оптимизатор или гибрид (Muon + Adam).
    """

    def __init__(self, optimizers: list[Optimizer]):
        self.optimizers = [o for o in optimizers if o is not None]
        self.param_groups = [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return {"opts": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, sd):
        for o, s in zip(self.optimizers, sd["opts"]):
            o.load_state_dict(s)


# ── разбиение параметров: что в Muon, что в Adam ─────────────────────────────
_HEAD_HINTS = ("head", "classifier", "fc", "output", "backbone.classifier")


def split_params(model, head_hints=_HEAD_HINTS):
    """2D+ скрытые веса → Muon; bias/нормы/1D и выходная голова → Adam."""
    muon, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = any(h in name for h in head_hints)
        if p.ndim >= 2 and not is_head:
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


# ── фабрика ──────────────────────────────────────────────────────────────────
def build_optimizer(model, name: str | None = None, lr: float | None = None):
    """Собрать оптимизатор по имени: adam | adamw | muon (гибрид).

    Параметры Muon/Adam берутся из CFG.optim; lr переопределяет adam_lr/базовый lr.
    """
    oc = CFG.optim
    name = (name or oc.name).lower()
    base_lr = CFG.train.learning_rate if lr is None else lr

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=base_lr,
                                weight_decay=oc.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=base_lr,
                                 weight_decay=oc.weight_decay)
    if name == "muon":
        muon_p, adam_p = split_params(model)
        opts = []
        if muon_p:
            opts.append(Muon(muon_p, lr=oc.muon_lr, momentum=oc.momentum,
                             nesterov=oc.nesterov, weight_decay=oc.weight_decay,
                             ns_steps=oc.ns_steps))
        if adam_p:
            opts.append(torch.optim.AdamW(adam_p, lr=oc.adam_lr,
                                          weight_decay=oc.weight_decay))
        print(f"[muon] params: muon-группа={sum(p.numel() for p in muon_p)} "
              f"adam-группа={sum(p.numel() for p in adam_p)}")
        return CombinedOptimizer(opts)
    raise ValueError(f"неизвестный optimizer: {name}")
