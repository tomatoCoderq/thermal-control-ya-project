"""Train ConvLSTM models on temporal IRT frame clips."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEG = HERE.parent
for p in (HERE, SEG, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.mps_train import setup_mps_env

setup_mps_env()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.device import get_device
from common.metrics import BCEDiceLoss, SegMetrics
from common.mps_train import (
    load_model_weights,
    mps_empty_cache,
    optimize_model_mps,
    suggest_num_workers,
)
from common.tracking import HistoryTracker
from data import build_temporal_loaders
from model import ConvLSTMSegNet

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


def run_temporal_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    desc: str = "",
    loss_fn: BCEDiceLoss | None = None,
    grad_clip: float | None = 1.0,
) -> tuple[float, dict[str, float]]:
    train = optimizer is not None
    model.train(train)
    criterion = loss_fn or BCEDiceLoss(pos_weight=10.0)
    metrics = SegMetrics()
    total, seen = 0.0, 0

    bar = tqdm(loader, leave=False, desc=desc or ("train" if train else "test"))
    with torch.set_grad_enabled(train):
        for x, y in bar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bs = x.shape[0]
            total += float(loss.detach()) * bs
            seen += bs
            metrics.update(logits.detach(), y)
            bar.set_postfix(loss=total / max(seen, 1))

    return total / max(seen, 1), metrics.compute()


def train_one(
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
    train_loader, test_loader, train_ds, test_ds = build_temporal_loaders(
        yaml_path,
        test_every=test_every,
        batch_size=batch_size,
        num_workers=nw,
    )
    in_ch = int(train_ds[0][0].shape[1])
    num_frames = int(train_ds[0][0].shape[0])
    print(
        f"\n=== ConvLSTM | {device} | T={num_frames} in_ch={in_ch} | "
        f"bs={train_loader.batch_size} workers={nw} | {epochs} ep ===\n"
        f"train: {len(train_ds.video_ids)} videos / {len(train_ds)} samples  "
        f"test: {len(test_ds.video_ids)} videos / {len(test_ds)} samples"
    )
    if device.type == "mps" and nw > 0:
        print("WARNING: num_workers>0 on MPS often stalls; prefer --workers 0")

    model = ConvLSTMSegNet(in_channels=in_ch)
    ckpt_best = HERE / "model_convlstm_best.tar"
    ckpt_last = HERE / "model_convlstm_last.tar"

    resume_path: Path | None = None
    if resume is not None:
        resume_path = (
            ckpt_best if str(resume) == "best" else ckpt_last if str(resume) == "last" else Path(resume)
        )
        load_model_weights(model, resume_path)

    model = optimize_model_mps(
        model, device, channels_last=False, compile_model=compile_model
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    criterion = BCEDiceLoss(pos_weight=pos_weight)
    print(
        f"optimizer=AdamW lr={lr} loss=BCE+Dice pos_weight={pos_weight} "
        f"sched=cosine"
    )

    run_dir = HERE / "runs" / "convlstm"
    tracker = HistoryTracker(run_dir, tag="convlstm", resume=resume_path is not None)
    start_epoch = tracker.next_epoch if resume_path is not None else 1
    end_epoch = start_epoch + epochs - 1
    best_iou, best_epoch = tracker.best_test_iou() if resume_path is not None else (-1.0, -1)

    for epoch in range(start_epoch, end_epoch + 1):
        train_loss, train_m = run_temporal_epoch(
            model,
            train_loader,
            device,
            opt,
            desc=f"train {epoch}",
            loss_fn=criterion,
        )
        test_loss, test_m = run_temporal_epoch(
            model,
            test_loader,
            device,
            desc=f"test {epoch}",
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
    print(f"done convlstm. best test IoU {best_iou:.4f} @ {best_epoch}")
    print(f"history -> {tracker.json_path}")
    return tracker, model, best_iou, best_epoch


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train ConvLSTM on TPU frame sequences")
    ap.add_argument("--yaml", type=Path, default=HERE / "dataset_convlstm.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--test-every", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=10.0)
    ap.add_argument("--resume", type=str, default=None, help="best | last | path.tar")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    train_one(
        args.yaml,
        args.epochs,
        get_device(args.device),
        test_every=args.test_every,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        num_workers=args.workers,
        resume=args.resume,
        compile_model=args.compile,
    )
