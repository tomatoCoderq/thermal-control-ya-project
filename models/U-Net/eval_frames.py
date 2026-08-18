"""Eval frames yaml — same as eval.py."""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import make_loader
from main import SoftDiceLoss, UNetModel
from metrics import dice_score, iou_score
from train import get_device

CKPT_BEST = HERE / "model_unet_frames_best.tar"

if __name__ == "__main__":
    _, ds, _ = make_loader(HERE / "dataset_frames.yaml", train=False, size=256)
    img, mask = ds[random.randrange(len(ds))]
    gt = mask[0].numpy()

    model = UNetModel(in_channels=img.shape[0], num_classes=1)
    if CKPT_BEST.exists():
        model.load_state_dict(torch.load(CKPT_BEST, map_location="cpu", weights_only=True))
        print(f"loaded {CKPT_BEST.name}")
    else:
        print(f"WARNING: no {CKPT_BEST.name}")

    device = get_device()
    model.eval().to(device)
    bce, dice_l = torch.nn.BCEWithLogitsLoss(), SoftDiceLoss()
    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        loss = (
            bce(logits, mask.unsqueeze(0).to(device))
            + dice_l(logits, mask.unsqueeze(0).to(device))
        ).item()
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    pred = (prob > 0.5).astype(np.float32)
    d, iou = dice_score(pred, gt), iou_score(pred, gt)
    print(f"Dice={d:.4f} IoU={iou:.4f} loss={loss:.4f}")

    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(img[0].numpy(), cmap="inferno")
    ax[0].set_title("frame")
    ax[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("GT")
    ax[2].imshow(prob, cmap="magma", vmin=0, vmax=1)
    ax[2].set_title("prob")
    ax[3].imshow(pred, cmap="gray", vmin=0, vmax=1)
    ax[3].set_title(f"Dice={d:.3f} IoU={iou:.3f}")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()
