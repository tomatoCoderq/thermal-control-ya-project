"""Train Mamba-UNet — Adam + MPS/36GB-optimized."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEG = HERE.parent
UNET = SEG / "U-Net"
for p in (HERE, SEG, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.mps_train import setup_mps_env

setup_mps_env()

import torch

from common.data import build_train_test_loaders
from common.device import get_device
from common.loop import run_epoch
from common.mps_train import (
    load_model_weights,
    mps_empty_cache,
    optimize_model_mps,
    suggest_batch_size,
    suggest_num_workers,
)
from common.tracking import HistoryTracker
from common.variants import VARIANTS
from irt_cfg import load_cfg
from model import MambaUNet


def train_one(
    name: str,
    yaml_path: Path,
    epochs: int,
    device: torch.device,
    *,
    test_every: int = 4,
    batch_size: int | None = None,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    compile_model: bool = False,  # selective-scan + compile often fragile
    num_workers: int | None = None,
    resume: Path | None = None,
) -> None:
    from common.data import INPUT_SIZE

    cfg = load_cfg(yaml_path, train=True)
    bs = batch_size or suggest_batch_size(device, kind="mamba")
    nw = suggest_num_workers(device) if num_workers is None else num_workers

    train_loader, test_loader, train_ds, test_ds = build_train_test_loaders(
        cfg,
        size=INPUT_SIZE,
        test_every=test_every,
        batch_size=bs,
        variant=name,
        num_workers=nw,
    )
    in_ch = int(train_ds[0][0].shape[0])
    print(
        f"\n=== MambaUNet {name} | {device} | in_ch={in_ch} | bs={bs} workers={nw} | {epochs} ep ===\n"
        f"train: {len(train_ds.video_ids)} videos / {len(train_ds)} samples  "
        f"test: {len(test_ds.video_ids)} videos / {len(test_ds)} samples"
    )

    model = MambaUNet(in_channels=in_ch, num_classes=1)
    ckpt_best = HERE / f"model_mamba_{name}_best.tar"
    ckpt_last = HERE / f"model_mamba_{name}_last.tar"
    if resume is not None:
        load_model_weights(model, resume)
    model = optimize_model_mps(
        model, device, channels_last=False, compile_model=compile_model
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"optimizer=Adam lr={lr} weight_decay={weight_decay}")

    run_dir = HERE / "runs" / name
    tracker = HistoryTracker(run_dir, tag=f"mamba-{name}", resume=resume is not None)
    start_epoch = tracker.next_epoch if resume is not None else 1
    end_epoch = start_epoch + epochs - 1
    best_iou, best_epoch = tracker.best_test_iou() if resume is not None else (-1.0, -1)

    if resume is not None and best_iou < 0:
        _, test_m0 = run_epoch(model, test_loader, device, desc="resume-eval")
        best_iou, best_epoch = test_m0["iou"], start_epoch - 1
        print(f"resume baseline test IoU {best_iou:.4f}")

    print(
        f"starting epochs {start_epoch}..{end_epoch}"
        + (" (weights resumed; optimizer state is fresh)" if resume else "…"),
        flush=True,
    )
    for epoch in range(start_epoch, end_epoch + 1):
        train_loss, train_m = run_epoch(
            model, train_loader, device, opt, desc=f"train {epoch}"
        )
        test_loss, test_m = run_epoch(model, test_loader, device, desc=f"test {epoch}")
        row = tracker.log(epoch, train_loss, test_loss, train_m, test_m)
        print(tracker.format_line(row, end_epoch))

        if test_m["iou"] > best_iou:
            best_iou, best_epoch = test_m["iou"], epoch
            torch.save(model.state_dict(), ckpt_best)
            print(f"  >>> best test IoU {best_iou:.4f} @ epoch {best_epoch}")

        mps_empty_cache(device)

    torch.save(model.state_dict(), ckpt_last)
    print(f"done mamba-{name}. best test IoU {best_iou:.4f} @ {best_epoch}")
    print(f"history -> {tracker.json_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=Path, default=UNET / "dataset_tsr.yaml")
    ap.add_argument("--epochs", type=int, default=50, help="number of NEW epochs to run")
    ap.add_argument("--test-every", type=int, default=4)
    ap.add_argument("--variants", nargs="+", default=["tsr"], choices=list(VARIANTS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None, help="DataLoader workers (MPS: use 0)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--compile", action="store_true", help="try torch.compile (experimental for Mamba)")
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="path to .tar, or 'best' / 'last' for default model_mamba_<variant>_*.tar",
    )
    args = ap.parse_args()
    device = get_device(args.device)
    print(f"device={device} | mps={torch.backends.mps.is_available()}")
    for name in args.variants:
        resume = args.resume
        if resume is not None and str(resume) in ("best", "last"):
            resume = HERE / f"model_mamba_{name}_{resume}.tar"
        train_one(
            name,
            args.yaml,
            args.epochs,
            device,
            test_every=args.test_every,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            compile_model=args.compile,
            num_workers=args.workers,
            resume=resume,
        )
