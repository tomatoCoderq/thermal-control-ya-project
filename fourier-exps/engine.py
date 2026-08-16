from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from config import CFG
from train import evaluate, train_epoch


def make_run_dir(tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = "".join(c if c.isalnum() or c in "+-_." else "_" for c in tag)
    path = CFG.paths.log_dir / f"{timestamp}_{safe_tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fit(train_loader, val_loader, model, loss_fn, optimizer, device, tag: str):
    run_dir = make_run_dir(tag)
    CFG.dump_yaml(run_dir / "config.yaml")
    metrics_path = run_dir / "metrics.csv"
    fields = ["epoch", "train_loss", "val_macro_f1", "val_accuracy"]
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()

    best_f1 = -1.0
    best_epoch = 0
    for epoch in range(1, CFG.train.epochs + 1):
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device)
        metrics = evaluate(val_loader, model, device, CFG.classes.n_classes)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": metrics["macro_f1"],
            "val_accuracy": metrics["accuracy"],
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow(row)
        print(
            f"epoch {epoch:3d}/{CFG.train.epochs} | loss {train_loss:.4f} | "
            f"val_f1 {metrics['macro_f1']:.4f} | val_acc {metrics['accuracy']:.4f}"
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            import torch
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    # Report the selected checkpoint, not the arbitrarily last epoch.
    import torch
    model.load_state_dict(
        torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    final = evaluate(val_loader, model, device, CFG.classes.n_classes)
    final["best_epoch"] = best_epoch
    final["selection_metric"] = "macro_f1"
    np.savetxt(run_dir / "confusion.csv", final["confusion"], fmt="%d", delimiter=",")
    (run_dir / "results.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir, final




