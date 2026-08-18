"""Uniform run history: history.json + metrics.csv (train_* / test_*)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class HistoryTracker:
    """Append one row per epoch; flush to history.json and metrics.csv."""

    FIELDS = [
        "epoch",
        "train_loss",
        "test_loss",
        "train_iou",
        "train_dice",
        "train_precision",
        "train_recall",
        "test_iou",
        "test_dice",
        "test_precision",
        "test_recall",
    ]

    def __init__(self, out_dir: Path | str, *, tag: str = "", resume: bool = False) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.json_path = self.out_dir / "history.json"
        self.csv_path = self.out_dir / "metrics.csv"
        self.history: list[dict[str, Any]] = []
        if resume and self.json_path.is_file():
            try:
                raw = json.loads(self.json_path.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    self.history = raw
                    print(f"history resume: {len(self.history)} epochs from {self.json_path}")
            except (json.JSONDecodeError, OSError) as exc:
                print(f"history resume skipped: {exc}")

    @property
    def next_epoch(self) -> int:
        if not self.history:
            return 1
        return int(max(int(r.get("epoch", 0)) for r in self.history)) + 1

    def best_test_iou(self) -> tuple[float, int]:
        if not self.history:
            return -1.0, -1
        best = max(self.history, key=lambda r: float(r.get("test_iou", -1.0)))
        return float(best.get("test_iou", -1.0)), int(best.get("epoch", -1))

    def log(
        self,
        epoch: int,
        train_loss: float,
        test_loss: float,
        train_metrics: dict[str, float],
        test_metrics: dict[str, float],
    ) -> dict[str, Any]:
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            **{f"train_{k}": float(v) for k, v in train_metrics.items()},
            **{f"test_{k}": float(v) for k, v in test_metrics.items()},
        }
        self.history.append(row)
        self.flush()
        return row

    def flush(self) -> None:
        self.json_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS, extrasaction="ignore")
            w.writeheader()
            for row in self.history:
                w.writerow({k: row.get(k, "") for k in self.FIELDS})

    def format_line(self, row: dict[str, Any], epochs: int) -> str:
        tag = f"[{self.tag}] " if self.tag else ""
        return (
            f"{tag}epoch {row['epoch']:3d}/{epochs}  "
            f"loss {row['train_loss']:.4f}/{row['test_loss']:.4f}  "
            f"IoU {row.get('train_iou', 0):.3f}/{row.get('test_iou', 0):.3f}  "
            f"test dice {row.get('test_dice', 0):.3f}  "
            f"P {row.get('test_precision', 0):.3f}  R {row.get('test_recall', 0):.3f}"
        )
