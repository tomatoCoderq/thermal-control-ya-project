"""Eval Thermal-Contrast U-Net on one sample."""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEG = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
for p in (SEG, ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.append(sp)

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.device import get_device
from common.metrics import SoftDiceLoss, dice_score, iou_score
from common.mps_train import load_model_weights
from data import ContrastCropDataset
from features import PRESETS
from irt_cfg import load_cfg
from irt_data import IRTDataset
from model import UNetModel

PRESET = "combo"
CKPT_BEST = HERE / f"model_contrast_{PRESET}_best.tar"
CHANNEL_TITLES = {
    "maxmin": "max−min",
    "maxfirst": "max−t0",
    "minfirst": "min−t0",
    "lastfirst": "last−t0",
    "std": "σ(t)",
    "mean": "mean",
    "pca1": "PCA₁(t)",
}


if __name__ == "__main__":
    cfg = load_cfg(HERE / "dataset_contrast.yaml", train=False)
    ds = ContrastCropDataset(IRTDataset(cfg), preset=PRESET)
    feat, mask = ds[random.randrange(len(ds))]
    gt = mask[0].numpy()

    model = UNetModel(in_channels=ds.num_channels, num_classes=1)
    if CKPT_BEST.exists():
        load_model_weights(model, CKPT_BEST)
        print(f"loaded {CKPT_BEST.name}")
    else:
        print(f"WARNING: no {CKPT_BEST.name}")

    device = get_device()
    model.eval().to(device)
    bce, dice_l = torch.nn.BCEWithLogitsLoss(), SoftDiceLoss()
    y = mask.unsqueeze(0)
    with torch.no_grad():
        logits = model(feat.unsqueeze(0).to(device))
        loss = (bce(logits, y.to(device)) + dice_l(logits, y.to(device))).item()
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    pred = (prob > 0.5).astype(np.float32)
    d, iou = dice_score(pred, gt), iou_score(pred, gt)
    print(
        f"preset={PRESET} ch={ds.channels} shape={tuple(feat.shape)} "
        f"Dice={d:.4f} IoU={iou:.4f} loss={loss:.4f}"
    )

    n_ch = feat.shape[0]
    fig, axes = plt.subplots(2, max(n_ch, 3), figsize=(3 * max(n_ch, 3), 6))
    for c in range(n_ch):
        name = ds.channels[c]
        axes[0, c].imshow(feat[c].numpy(), cmap="inferno")
        axes[0, c].set_title(CHANNEL_TITLES.get(name, name))
        axes[0, c].axis("off")
    for c in range(n_ch, axes.shape[1]):
        axes[0, c].axis("off")

    axes[1, 0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("GT")
    axes[1, 1].imshow(prob, cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title("prob")
    axes[1, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title(f"pred Dice={d:.3f}")
    for c in range(3, axes.shape[1]):
        axes[1, c].axis("off")
    for a in axes[1, :3]:
        a.axis("off")
    plt.tight_layout()
    plt.show()
