"""Eval Mamba-UNet on 256×256 crops."""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEG = HERE.parent
ROOT = HERE.parents[1]
UNET = SEG / "U-Net"
for p in (ROOT, SEG, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.device import get_device
from common.metrics import dice_score, iou_score
from data import make_loader
from model import MambaUNet, SoftDiceLoss

CKPT_BEST = HERE / "model_mamba_tsr_best.tar"


def load_weights(model: torch.nn.Module, path: Path) -> None:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)


if __name__ == "__main__":
    yaml = UNET / "dataset_tsr.yaml"
    _, ds, _ = make_loader(yaml, train=False, size=256)
    img, mask = ds[random.randrange(len(ds))]
    gt = mask[0].numpy()

    model = MambaUNet(in_channels=img.shape[0], num_classes=1)
    if CKPT_BEST.exists():
        load_weights(model, CKPT_BEST)
        print(f"loaded {CKPT_BEST.name}")
    else:
        print(f"WARNING: no {CKPT_BEST.name}")

    device = get_device()
    model.eval().to(device)
    bce, dice_l = torch.nn.BCEWithLogitsLoss(), SoftDiceLoss()
    y = mask.unsqueeze(0)
    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        loss = (bce(logits, y.to(device)) + dice_l(logits, y.to(device))).item()
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    pred = (prob > 0.5).astype(np.float32)
    d, iou = dice_score(pred, gt), iou_score(pred, gt)
    print(f"in_ch={img.shape[0]} size={tuple(img.shape[-2:])} out={logits.shape} Dice={d:.4f} IoU={iou:.4f}")

    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(img[0].numpy(), cmap="inferno")
    ax[0].set_title("input ch0")
    ax[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("GT")
    ax[2].imshow(prob, cmap="magma", vmin=0, vmax=1)
    ax[2].set_title("prob")
    ax[3].imshow(pred, cmap="gray", vmin=0, vmax=1)
    ax[3].set_title(f"Dice={d:.3f}")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()
