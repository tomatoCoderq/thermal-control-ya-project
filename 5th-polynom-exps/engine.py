from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from config import CFG
from train import train, evaluate


def make_run_dir(tag: str = "run") -> Path:
    """Creates unique dir for each run"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "+-_." else "_" for c in tag)
    run_dir = CFG.paths.log_dir / f"{ts}_{safe}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class CsvMetrics:
    """Writing to csv for further analysis"""

    def __init__(self, path: Path, fields: list[str]):
        self.path = Path(path)
        self.fields = fields
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(fields)

    def append(self, row: dict) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.fields])


def _save_config_snapshot(run_dir: Path) -> None:
    """Save snapshot of config.yaml for spec run"""
    try:
        CFG.dump_yaml(run_dir / "config.yaml")
    except Exception:
        pass


def fit(
    train_loader,
    model,
    loss_fn,
    optimizer,
    device,
    val_loader=None,
    tag: str = "run",
):
    """Обучение с логированием в файлы. Возврат (run_dir, loss_history, result)."""
    run_dir = make_run_dir(tag)
    _save_config_snapshot(run_dir)
    metrics = CsvMetrics(run_dir / "metrics.csv",
                         ["epoch", "train_loss", "val_f1", "val_acc"])

    n_params = sum(p.numel() for p in model.parameters())
    print("START tag=%s device=%s epochs=%d lr=%s params=%d",
          tag, device, CFG.train.epochs, CFG.train.learning_rate, n_params)

    loss_history = train(
        train_loader, model, loss_fn, optimizer, device,
        val_loader=val_loader, metrics_csv=metrics,
    )

    # Финальная оценка
    result: dict = {}
    if val_loader is not None:
        f1, acc, cm = evaluate(val_loader, model, device)
        result = {"final_f1": float(f1), "final_acc": float(acc)}
        np.savetxt(run_dir / "confusion.csv", cm, fmt="%d", delimiter=",")
        print("FINAL f1=%.4f acc=%.4f", f1, acc)

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"tag": tag,
                   "config": CFG.model_dump(mode="json"),
                   **result}, f, ensure_ascii=False, indent=2)

    print("DONE tag=%s device=%s epochs=%d lr=%s params=%d", tag, device, CFG.train.epochs, CFG.train.learning_rate, n_params)
    return run_dir, loss_history, result
