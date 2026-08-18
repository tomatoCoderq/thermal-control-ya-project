"""Общий цикл обучения/оценки, ветвление по задаче.

regression     — лосс в нормир. пространстве, метрики (MAE/RMSE) в мм;
classification — CE-семейство, метрики accuracy/F1.
"""
from __future__ import annotations

import numpy as np
import torch

from .config import CFG


def train_loop(loader, model, loss_fn, optimizer, device, task,
               target_mean=0.0, target_std=1.0, val_loader=None, log=True):
    for epoch in range(1, CFG.train.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            seen += len(y)
        msg = f"epoch {epoch:3d}/{CFG.train.epochs} | loss {running/max(seen,1):.4f}"
        if val_loader is not None:
            m = evaluate(val_loader, model, device, task, target_mean, target_std)
            msg += " | " + " ".join(f"{k}={v:.4f}" for k, v in m.items())
        if log:
            print(msg)


@torch.no_grad()
def evaluate(loader, model, device, task, target_mean=0.0, target_std=1.0):
    model.eval()
    yt, yp = [], []
    for x, y in loader:
        out = model(x.to(device)).cpu().numpy()
        if task == "regression":
            yp.append(out.reshape(-1) * target_std + target_mean)
            yt.append(y.numpy().reshape(-1) * target_std + target_mean)
        else:
            yp += out.argmax(1).tolist()
            yt += y.tolist()
    if task == "regression":
        e = np.concatenate(yp) - np.concatenate(yt)
        return {"mae_mm": float(np.mean(np.abs(e))),
                "rmse_mm": float(np.sqrt(np.mean(e ** 2)))}
    from sklearn.metrics import accuracy_score, f1_score
    return {"acc": float(accuracy_score(yt, yp)),
            "f1": float(f1_score(yt, yp, average="macro", zero_division=0))}
