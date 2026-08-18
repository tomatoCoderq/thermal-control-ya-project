"""Train static U-Net on temporal contrast maps (Variant A baseline)."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEG = HERE.parent
# Local `data.py` must win over segmentation/U-Net/data.py — do not put U-Net on sys.path.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
for p in (SEG, ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.append(sp)

from common.mps_train import setup_mps_env

setup_mps_env()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.device import get_device
from common.loop import run_epoch
from common.metrics import BCEDiceLoss
from common.mps_train import (
    load_model_weights,
    mps_empty_cache,
    optimize_model_mps,
    suggest_num_workers,
)
from common.tracking import HistoryTracker
from data import build_contrast_loaders
from features import PRESETS
from model import UNetModel

EpochCallback = Callable[
    [
        HistoryTracker,
        dict[str, Any],
        int,
        int,
        float,
        int,
        torch.optim.Optimizer,
        nn.Module,
    ],
    None,
]


def train_one(
    preset: str,
    yaml_path: Path,
    epochs: int,
    device: torch.device,
    *,
    test_every: int = 4,
    batch_size: int | None = None,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    pos_weight: float = 10.0,
    num_workers: int | None = None,
    resume: Path | str | None = None,
    compile_model: bool = False,
    on_epoch_end: EpochCallback | None = None,
) -> tuple[HistoryTracker, nn.Module, float, int]:
    nw = suggest_num_workers(device) if num_workers is None else num_workers
    train_loader, test_loader, train_ds, test_ds = build_contrast_loaders(
        yaml_path,
        preset=preset,
        test_every=test_every,
        batch_size=batch_size,
        num_workers=nw,
    )
    in_ch = train_ds.num_channels
    ch_names = train_ds.channels
    print(
        f"\n=== Thermal-Contrast U-Net [{preset}] | {device} | in_ch={in_ch} {ch_names} | "
        f"bs={train_loader.batch_size} workers={nw} | {epochs} ep ===\n"
        f"train: {len(train_ds.video_ids)} videos / {len(train_ds)} samples  "
        f"test: {len(test_ds.video_ids)} videos / {len(test_ds)} samples"
    )

    model = UNetModel(in_channels=in_ch, num_classes=1)
    ckpt_best = HERE / f"model_contrast_{preset}_best.tar"
    ckpt_last = HERE / f"model_contrast_{preset}_last.tar"

    resume_path: Path | None = None
    if resume is not None:
        resume_path = (
            ckpt_best if str(resume) == "best" else ckpt_last if str(resume) == "last" else Path(resume)
        )
        load_model_weights(model, resume_path)

    use_cl = device.type == "cuda"
    model = optimize_model_mps(
        model, device, channels_last=use_cl, compile_model=compile_model
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    criterion = BCEDiceLoss(pos_weight=pos_weight)

    run_dir = HERE / "runs" / preset
    tracker = HistoryTracker(run_dir, tag=f"contrast-{preset}", resume=resume_path is not None)
    start_epoch = tracker.next_epoch if resume_path is not None else 1
    end_epoch = start_epoch + epochs - 1
    best_iou, best_epoch = tracker.best_test_iou() if resume_path is not None else (-1.0, -1)

    for epoch in range(start_epoch, end_epoch + 1):
        train_loss, train_m = run_epoch(
            model,
            train_loader,
            device,
            opt,
            desc=f"train {epoch}",
            channels_last=use_cl,
            loss_fn=criterion,
            grad_clip=1.0,
        )
        test_loss, test_m = run_epoch(
            model,
            test_loader,
            device,
            desc=f"test {epoch}",
            channels_last=use_cl,
            loss_fn=criterion,
        )
        sched.step()
        row = tracker.log(epoch, train_loss, test_loss, train_m, test_m)
        print(tracker.format_line(row, end_epoch))

        if test_m["iou"] > best_iou:
            best_iou, best_epoch = test_m["iou"], epoch
            torch.save(model.state_dict(), ckpt_best)
            print(f"  >>> best test IoU {best_iou:.4f} @ epoch {best_epoch}")

        if on_epoch_end is not None:
            on_epoch_end(tracker, row, epoch, end_epoch, best_iou, best_epoch, opt, model)

        mps_empty_cache(device)

    torch.save(model.state_dict(), ckpt_last)
    print(f"done contrast-{preset}. best test IoU {best_iou:.4f} @ {best_epoch}")
    print(f"history -> {tracker.json_path}")
    return tracker, model, best_iou, best_epoch


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Thermal-Contrast U-Net (static baseline)")
    ap.add_argument("--yaml", type=Path, default=HERE / "dataset_contrast.yaml")
    ap.add_argument("--preset", default="combo", choices=sorted(PRESETS))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--test-every", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=10.0)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    train_one(
        args.preset,
        args.yaml,
        args.epochs,
        get_device(args.device),
        test_every=args.test_every,
        batch_size=args.batch_size,
        num_workers=args.workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        resume=args.resume,
        compile_model=args.compile,
    )
