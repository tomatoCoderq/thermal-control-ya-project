"""Train Attention U-Net — AdamW + MPS/36GB-optimized."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
import torch.nn as nn

from common.data import INPUT_SIZE, build_train_test_loaders
from common.device import get_device
from common.loop import run_epoch
from common.metrics import BCEDiceLoss
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
from model import AttentionUNet

EpochCallback = Callable[
    [
        HistoryTracker,
        dict[str, Any],
        int,
        int,
        float,
        int,
        torch.optim.Optimizer,
    ],
    None,
]


def _resolve_resume(resume: Path | str | None, ckpt_dir: Path, name: str) -> Path | None:
    if resume is None:
        return None
    if str(resume) in ("best", "last"):
        return ckpt_dir / f"model_attunet_{name}_{resume}.tar"
    return Path(resume)


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
    pos_weight: float | None = None,
    compile_model: bool = True,
    num_workers: int | None = None,
    resume: Path | str | None = None,
    model_factory: Callable[[int], nn.Module] | None = None,
    ckpt_dir: Path | None = None,
    runs_dir: Path | None = None,
    on_epoch_end: EpochCallback | None = None,
) -> tuple[HistoryTracker, nn.Module, float, int]:
    """Train one feature variant. Returns tracker, model, best_iou, best_epoch."""
    ckpt_dir = HERE if ckpt_dir is None else ckpt_dir
    runs_dir = HERE / "runs" if runs_dir is None else runs_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(yaml_path, train=True)
    bs = batch_size or suggest_batch_size(device, kind="attn")
    nw = suggest_num_workers(device) if num_workers is None else num_workers
    use_cl = device.type == "cuda"

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
        f"\n=== AttUNet {name} | {device} | in_ch={in_ch} | bs={bs} workers={nw} | {epochs} ep ===\n"
        f"train: {len(train_ds.video_ids)} videos / {len(train_ds)} samples  "
        f"test: {len(test_ds.video_ids)} videos / {len(test_ds)} samples"
    )
    if device.type == "mps" and nw > 0:
        print("WARNING: num_workers>0 on MPS often stalls; prefer --workers 0")

    factory = model_factory or (lambda ch: AttentionUNet(img_ch=ch, output_ch=1))
    model = factory(in_ch)
    ckpt_best = ckpt_dir / f"model_attunet_{name}_best.tar"
    ckpt_last = ckpt_dir / f"model_attunet_{name}_last.tar"

    resume_path = _resolve_resume(resume, ckpt_dir, name)
    if resume_path is not None:
        load_model_weights(model, resume_path)

    model = optimize_model_mps(
        model, device, channels_last=use_cl, compile_model=compile_model
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    criterion = BCEDiceLoss(pos_weight=pos_weight)
    print(
        f"optimizer=AdamW lr={lr} weight_decay={weight_decay} "
        f"sched=cosine loss=BCE+Dice pos_weight={pos_weight} channels_last={use_cl}"
    )

    run_dir = runs_dir / name
    tracker = HistoryTracker(run_dir, tag=f"att-{name}", resume=resume_path is not None)
    start_epoch = tracker.next_epoch if resume_path is not None else 1
    end_epoch = start_epoch + epochs - 1
    best_iou, best_epoch = tracker.best_test_iou() if resume_path is not None else (-1.0, -1)

    if resume_path is not None and best_iou < 0:
        _, test_m0 = run_epoch(
            model,
            test_loader,
            device,
            desc="resume-eval",
            channels_last=use_cl,
            loss_fn=criterion,
        )
        best_iou, best_epoch = test_m0["iou"], start_epoch - 1
        print(f"resume baseline test IoU {best_iou:.4f}")

    print(
        f"starting epochs {start_epoch}..{end_epoch}"
        + (" (weights resumed; optimizer state is fresh)" if resume_path else "")
        + (" — first batch + torch.compile can take 1–3 min…" if compile_model else "…"),
        flush=True,
    )
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
            on_epoch_end(tracker, row, epoch, end_epoch, best_iou, best_epoch, opt)

        mps_empty_cache(device)

    torch.save(model.state_dict(), ckpt_last)
    print(f"done att-{name}. best test IoU {best_iou:.4f} @ {best_epoch}")
    print(f"history -> {tracker.json_path}")
    return tracker, model, best_iou, best_epoch


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
    ap.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="BCE positive-class weight; defects cover ~2%% of a frame, try 5-20",
    )
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="path to .tar weights (e.g. model_attunet_tsr_best.tar). "
        "Use --resume best|last for default paths.",
    )
    args = ap.parse_args()
    device = get_device(args.device)
    print(f"device={device} | mps={torch.backends.mps.is_available()}")
    for name in args.variants:
        train_one(
            name,
            args.yaml,
            args.epochs,
            device,
            test_every=args.test_every,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            pos_weight=args.pos_weight,
            compile_model=not args.no_compile,
            num_workers=args.workers,
            resume=args.resume,
        )
