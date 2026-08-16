from __future__ import annotations

import torch

from metrics import classification_metrics


def evaluate(loader, model, device, n_classes: int) -> dict:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(targets.tolist())
    return classification_metrics(y_true, y_pred, n_classes)


def train_epoch(loader, model, loss_fn, optimizer, device) -> float:
    model.train()
    total_loss, count = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        count += len(targets)
    return total_loss / max(count, 1)





