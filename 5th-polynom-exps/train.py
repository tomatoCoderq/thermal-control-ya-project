from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch

from config import CFG


def train(dataLoader, model, loss, optimizer, device,
          loss_history=None, val_loader=None, metrics_csv=None):
    """Per-epoch train """
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
        row = {"epoch": epoch, "train_loss": round(train_loss, 5)}

        # validate each epoch
        if val_loader is not None:
            f1, acc, _ = evaluate(val_loader, model, device)
            row["val_f1"] = round(float(f1), 5)
            row["val_acc"] = round(float(acc), 5)

        if metrics_csv is not None:
            metrics_csv.append(row)
        msg = f"epoch {epoch:3d}/{CFG.train.epochs} | train_loss {train_loss:.4f}"
        if "val_f1" in row:
            msg += f" | val_f1 {row['val_f1']:.4f} | val_acc {row['val_acc']:.4f}"
        print(msg)

    return loss_history


def evaluate(dataLoader, model, device):
    model.eval()

    ypred = []
    yt = []

    with torch.no_grad():
        for x, y, in dataLoader:
            x, y = x.to(device), y.to(device)
            ypred += model(x).argmax(dim=1).tolist()
            yt += y.tolist()

    return (
        f1_score(yt, ypred, average="macro"),
        accuracy_score(yt, ypred),
        confusion_matrix(yt, ypred, labels=list(range(CFG.classes.n_classes))),
    )
