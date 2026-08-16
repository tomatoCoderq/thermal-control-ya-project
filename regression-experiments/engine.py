from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

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
    try:
        CFG.dump_yaml(run_dir / "config.yaml")
    except Exception:
        pass


def fit(train_loader, model, loss_fn, optimizer, device,
        target_mean, target_std, val_loader=None, tag: str = "run",
        extra_result: dict | None = None):
    """Обучение регрессора с логами. Возврат (run_dir, loss_history, result).

    target_mean/std (мм) нужны, чтобы денормировать val-метрики и записать
    статистики таргета в снапшот прогона.
    """
    run_dir = make_run_dir(tag)
    _save_config_snapshot(run_dir)
    metrics = CsvMetrics(run_dir / "metrics.csv",
                         ["epoch", "train_loss", "val_mae", "val_rmse"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"START tag={tag} device={device} epochs={CFG.train.epochs} "
          f"lr={CFG.train.learning_rate} params={n_params}")

    loss_history = train(
        train_loader, model, loss_fn, optimizer, device,
        target_mean, target_std, val_loader=val_loader, metrics_csv=metrics,
    )

    result: dict = {}
    if val_loader is not None:
        mae, rmse = evaluate(val_loader, model, device, target_mean, target_std)
        result = {"final_mae_mm": float(mae), "final_rmse_mm": float(rmse)}
        print(f"FINAL mae={mae:.4f} мм rmse={rmse:.4f} мм")

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"tag": tag,
                   "task": "regression",
                   "target_mean_mm": float(target_mean),
                   "target_std_mm": float(target_std),
                   "config": CFG.model_dump(mode="json"),
                   **result, **(extra_result or {})},
                  f, ensure_ascii=False, indent=2)

    print(f"DONE tag={tag} params={n_params} → {run_dir}")
    return run_dir, loss_history, result
