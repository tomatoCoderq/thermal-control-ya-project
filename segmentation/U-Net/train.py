"""Train U-Net on TSR / Fourier with train+test tracking"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEG = HERE.parent
for p in (HERE, SEG, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch
import torch.optim as optim

from common.data import build_train_test_loaders
from common.device import get_device
from common.loop import run_epoch
from common.metrics import BCEDiceLoss
from common.mps_train import load_model_weights
from common.tracking import HistoryTracker
from common.variants import VARIANTS
from irt_cfg import load_cfg
from main import UNetModel


def optimize_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model = model.to(device=device, memory_format=torch.channels_last)
    if device.type not in ("mps", "cuda"):
        return model
    try:
        model = torch.compile(model, mode="default", fullgraph=False)
        print(f"torch.compile enabled on {device.type}")
    except Exception as exc:  # noqa: BLE001
        print(f"torch.compile skipped: {exc}")
    return model


def train_one(
    name: str,
    yaml_path: Path,
    epochs: int,
    device: torch.device,
    *,
    test_every: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    pos_weight: float | None = None,
    runs_root: Path | None = None,
    resume: Path | None = None,
) -> None:
    from common.data import INPUT_SIZE

    cfg = load_cfg(yaml_path, train=True)
    bs = cfg.loader.batch_size
    if device.type == "mps":
        bs = max(bs, 4)

    train_loader, test_loader, train_ds, test_ds = build_train_test_loaders(
        cfg, size=INPUT_SIZE, test_every=test_every, batch_size=bs, variant=name
    )
    in_ch = int(train_ds[0][0].shape[0])
    print(
        f"\n=== UNet {name} | {device} | in_ch={in_ch} | {epochs} ep ===\n"
        f"train: {len(train_ds.video_ids)} videos / {len(train_ds)} samples  "
        f"test: {len(test_ds.video_ids)} videos / {len(test_ds)} samples"
    )

    model = UNetModel(in_channels=in_ch, num_classes=1)
    ckpt_best = HERE / f"model_unet_{name}_best.tar"
    ckpt_last = HERE / f"model_unet_{name}_last.tar"
    if resume is not None:
        load_model_weights(model, resume)
    model = optimize_model(model, device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    criterion = BCEDiceLoss(pos_weight=pos_weight)
    print(
        f"optimizer=AdamW lr={lr} weight_decay={weight_decay} "
        f"sched=cosine loss=BCE+Dice pos_weight={pos_weight}"
    )

    run_dir = (runs_root or (HERE / "runs")) / name
    tracker = HistoryTracker(run_dir, tag=f"unet-{name}", resume=resume is not None)
    start_epoch = tracker.next_epoch if resume is not None else 1
    end_epoch = start_epoch + epochs - 1
    best_iou, best_epoch = tracker.best_test_iou() if resume is not None else (-1.0, -1)

    if resume is not None and best_iou < 0:
        _, test_m0 = run_epoch(
            model,
            test_loader,
            device,
            desc="resume-eval",
            channels_last=True,
            loss_fn=criterion,
        )
        best_iou, best_epoch = test_m0["iou"], start_epoch - 1
        print(f"resume baseline test IoU {best_iou:.4f}")

    for epoch in range(start_epoch, end_epoch + 1):
        train_loss, train_m = run_epoch(
            model,
            train_loader,
            device,
            opt,
            desc=f"train {epoch}",
            channels_last=True,
            loss_fn=criterion,
            grad_clip=1.0,
        )
        test_loss, test_m = run_epoch(
            model,
            test_loader,
            device,
            desc=f"test {epoch}",
            channels_last=True,
            loss_fn=criterion,
        )
        sched.step()
        row = tracker.log(epoch, train_loss, test_loss, train_m, test_m)
        print(tracker.format_line(row, end_epoch))

        if test_m["iou"] > best_iou:
            best_iou, best_epoch = test_m["iou"], epoch
            torch.save(model.state_dict(), ckpt_best)
            print(f"  >>> best test IoU {best_iou:.4f} @ epoch {best_epoch} -> {ckpt_best.name}")

        if device.type == "mps":
            torch.mps.empty_cache()

    torch.save(model.state_dict(), ckpt_last)
    print(
        f"done unet-{name}. best test IoU {best_iou:.4f} @ {best_epoch}\n"
        f"history -> {tracker.json_path} / {tracker.csv_path}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=Path, default=HERE / "dataset.yaml")
    ap.add_argument("--epochs", type=int, default=50, help="number of NEW epochs to run")
    ap.add_argument("--test-every", type=int, default=4)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="BCE positive-class weight; defects cover ~2%% of a frame, try 5-20",
    )
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="path to .tar, or 'best' / 'last' for model_unet_<variant>_*.tar",
    )
    args = ap.parse_args()
    device = get_device(args.device)
    print(f"device={device}")
    for name in args.variants:
        resume = args.resume
        if resume is not None and str(resume) in ("best", "last"):
            resume = HERE / f"model_unet_{name}_{resume}.tar"
        train_one(
            name,
            args.yaml,
            args.epochs,
            device,
            test_every=args.test_every,
            lr=args.lr,
            weight_decay=args.weight_decay,
            pos_weight=args.pos_weight,
            resume=resume,
        )
