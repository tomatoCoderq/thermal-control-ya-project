#!/usr/bin/env python3
"""Plot train/test curves from history.json (or metrics.csv).

Usage:
  python -m models.common.plot_history models/U-Net/runs/tsr/history.json
  python models/common/plot_history.py runs/tsr --save curves.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path: Path) -> list[dict]:
    path = Path(path)
    if path.is_dir():
        js = path / "history.json"
        cs = path / "metrics.csv"
        if js.exists():
            path = js
        elif cs.exists():
            path = cs
        else:
            raise FileNotFoundError(f"no history.json / metrics.csv in {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    return float(row[key])


def plot_history(history: list[dict], *, title: str = "", save: Path | None = None, show: bool = True) -> None:
    if not history:
        raise ValueError("empty history")
    epochs = [_f(r, "epoch") for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle(title or "train / test")

    axes[0].plot(epochs, [_f(r, "train_loss") for r in history], label="train")
    axes[0].plot(epochs, [_f(r, "test_loss") for r in history], label="test")
    axes[0].set_title("loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [_f(r, "train_iou") for r in history], label="train")
    axes[1].plot(epochs, [_f(r, "test_iou") for r in history], label="test")
    axes[1].set_title("IoU")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [_f(r, "train_dice") for r in history], label="train")
    axes[2].plot(epochs, [_f(r, "test_dice") for r in history], label="test")
    axes[2].set_title("Dice")
    axes[2].set_xlabel("epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=140)
        print(f"saved {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="history.json, metrics.csv, or run directory")
    ap.add_argument("--save", type=Path, default=None, help="optional PNG path")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)

    hist = load_history(args.path)
    title = args.title or str(args.path)
    plot_history(hist, title=title, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
