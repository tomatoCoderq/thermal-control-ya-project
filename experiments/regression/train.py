import numpy as np
import torch

from config import CFG


def train(dataLoader, model, loss, optimizer, device,
          target_mean, target_std, loss_history=None,
          val_loader=None, metrics_csv=None):
    """Per-epoch train для регрессии глубины.

    Таргет в loader'е уже нормирован (z-score), лосс считается в нормированном
    пространстве. Для логов val-метрики денормируются в мм через target_mean/std.
    """
    loss_history = [] if loss_history is None else loss_history

    for epoch in range(1, CFG.train.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in dataLoader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()
            loss_history.append(l.item())
            running += l.item() * len(y)
            seen += len(y)

        train_loss = running / max(seen, 1)
        row = {"epoch": epoch, "train_loss": round(train_loss, 6)}

        if val_loader is not None:
            mae, rmse = evaluate(val_loader, model, device, target_mean, target_std)
            row["val_mae"] = round(float(mae), 5)
            row["val_rmse"] = round(float(rmse), 5)

        if metrics_csv is not None:
            metrics_csv.append(row)
        msg = f"epoch {epoch:3d}/{CFG.train.epochs} | train_loss {train_loss:.5f}"
        if "val_mae" in row:
            msg += f" | val_mae {row['val_mae']:.4f} мм | val_rmse {row['val_rmse']:.4f} мм"
        print(msg)

    return loss_history


def evaluate(dataLoader, model, device, target_mean, target_std):
    """Возврат (mae_mm, rmse_mm) — денормированные в миллиметры."""
    model.eval()
    ypred, yt = [], []
    with torch.no_grad():
        for x, y in dataLoader:
            out = model(x.to(device)).cpu().numpy().reshape(-1)
            ypred.append(out * target_std + target_mean)
            yt.append(y.numpy().reshape(-1) * target_std + target_mean)
    ypred, yt = np.concatenate(ypred), np.concatenate(yt)
    e = ypred - yt
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))
