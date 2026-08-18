"""Train legacy Attention U-Net (nn.Upsample) — resume old checkpoints."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LEGACY = Path(__file__).resolve().parent
ATT = LEGACY.parent
SEG = ATT.parent
ROOT = ATT.parents[1]
UNET = SEG / "U-Net"

import importlib.util

_spec = importlib.util.spec_from_file_location("attunet_legacy", LEGACY / "model.py")
_legacy = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_legacy)
LegacyAttentionUNet = _legacy.AttentionUNet

for p in (ATT, SEG, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.mps_train import setup_mps_env

setup_mps_env()

from common.device import get_device
from common.variants import VARIANTS
from train import train_one


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yaml", type=Path, default=UNET / "dataset_tsr.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--test-every", type=int, default=4)
    ap.add_argument("--variants", nargs="+", default=["tsr"], choices=list(VARIANTS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=None)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--resume", default=None, help="best | last | path to .tar")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"legacy upsample model | device={device}")
    if device.type == "mps":
        print("WARNING: legacy Upsample + MPS backward often fails — use --device cpu")

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
            model_factory=lambda ch: LegacyAttentionUNet(img_ch=ch, output_ch=1),
            ckpt_dir=LEGACY / "checkpoints",
            runs_dir=LEGACY / "runs",
        )
