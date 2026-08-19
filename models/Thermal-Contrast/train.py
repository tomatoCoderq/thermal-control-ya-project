"""Train the static U-Net on extracted thermal contrast channels."""
from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from paths import DATASETS_ROOT, RUNS_DIR

import torch
import torch.nn as nn

from channels import CHANNEL_NAMES, DEFAULT_PARAMS, NUM_CHANNELS, ChannelParams
from checkpoint import load_checkpoint, move_optimizer_state_to_device, save_checkpoint
from common.device import get_device
from common.loop import run_epoch
from common.metrics import BCEDiceLoss
from common.tracking import HistoryTracker
from data import build_loaders
from model import UNetModel

CHECKPOINT_BEST = RUNS_DIR / "contrast_best.pkl"
CHECKPOINT_LAST = RUNS_DIR / "contrast_last.pkl"

EpochCallback = Callable[[HistoryTracker, dict[str, Any], int, int, float, int], None]


def resolve_checkpoint(resume: str | Path) -> Path:
    """`"best"` / `"last"` name the standard run files; anything else is a path."""
    if str(resume) == "best":
        return CHECKPOINT_BEST
    if str(resume) == "last":
        return CHECKPOINT_LAST
    return Path(resume)


def train(
    *,
    epochs: int = 50,
    device: torch.device | None = None,
    params: ChannelParams = DEFAULT_PARAMS,
    root: Path | str = DATASETS_ROOT,
    include: list[str] | None = None,
    test_every: int = 4,
    batch_size: int = 4,
    num_workers: int = 0,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    pos_weight: float = 10.0,
    resume: str | Path | None = None,
    on_epoch_end: EpochCallback | None = None,
) -> tuple[HistoryTracker, nn.Module, float, int]:
    device = device or get_device()
    train_loader, test_loader, train_ds, test_ds = build_loaders(
        root=root,
        include=include,
        params=params,
        test_every=test_every,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = UNetModel(in_channels=NUM_CHANNELS, num_classes=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)
    criterion = BCEDiceLoss(pos_weight=pos_weight)

    start_epoch, best_iou, best_epoch, history = 1, -1.0, -1, []
    if resume is not None:
        state = load_checkpoint(resolve_checkpoint(resume))
        model.load_state_dict(state.model_state, strict=True)
        optimizer.load_state_dict(state.optimizer_state)
        scheduler.load_state_dict(state.scheduler_state)
        start_epoch = state.epoch + 1
        best_iou, best_epoch, history = state.best_iou, state.best_epoch, state.history
        print(f"resumed from epoch {state.epoch} (best IoU {best_iou:.4f} @ {best_epoch})")

    model = model.to(device)
    move_optimizer_state_to_device(optimizer, device)
    tracker = HistoryTracker(RUNS_DIR, tag="contrast", resume=False)
    tracker.history = list(history)

    end_epoch = start_epoch + epochs - 1
    print(
        f"\n=== Thermal-Contrast U-Net | {device} | in_ch={NUM_CHANNELS} {list(CHANNEL_NAMES)} "
        f"| bs={batch_size} workers={num_workers} | epochs {start_epoch}..{end_epoch} ===\n"
        f"train: {len(train_ds)} videos {train_ds.video_ids}\n"
        f"test:  {len(test_ds)} videos {test_ds.video_ids}"
    )

    for epoch in range(start_epoch, end_epoch + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, device, optimizer, desc=f"train {epoch}", loss_fn=criterion, grad_clip=1.0
        )
        test_loss, test_metrics = run_epoch(
            model, test_loader, device, desc=f"test {epoch}", loss_fn=criterion
        )
        scheduler.step()

        row = tracker.log(epoch, train_loss, test_loss, train_metrics, test_metrics)
        print(tracker.format_line(row, end_epoch))

        if test_metrics["iou"] > best_iou:
            best_iou, best_epoch = test_metrics["iou"], epoch
            save_checkpoint(
                CHECKPOINT_BEST,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_iou=best_iou,
                best_epoch=best_epoch,
                params=params,
                history=tracker.history,
            )
            print(f"  >>> best test IoU {best_iou:.4f} @ epoch {epoch}")

        save_checkpoint(
            CHECKPOINT_LAST,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_iou=best_iou,
            best_epoch=best_epoch,
            params=params,
            history=tracker.history,
        )

        if on_epoch_end is not None:
            on_epoch_end(tracker, row, epoch, end_epoch, best_iou, best_epoch)

        if device.type == "mps":
            torch.mps.empty_cache()

    print(f"done. best test IoU {best_iou:.4f} @ epoch {best_epoch}")
    print(f"checkpoints -> {CHECKPOINT_BEST.name}, {CHECKPOINT_LAST.name}")
    print(f"history -> {tracker.json_path}")
    return tracker, model, best_iou, best_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thermal-Contrast U-Net")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--root", type=Path, default=DATASETS_ROOT)
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="sub-dataset directories to use; default is all of them",
    )
    parser.add_argument("--test-every", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_PARAMS.num_frames)
    parser.add_argument("--resume", default=None, help='"best", "last", or a path to a .pkl')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        device=get_device(args.device),
        params=ChannelParams(num_frames=args.num_frames),
        root=args.root,
        include=args.include,
        test_every=args.test_every,
        batch_size=args.batch_size,
        num_workers=args.workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        resume=args.resume,
    )
