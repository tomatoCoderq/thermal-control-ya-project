"""Eval ConvLSTM on one random temporal clip."""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEG = HERE.parent
for p in (HERE, SEG, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.device import get_device
from common.metrics import SoftDiceLoss, dice_score, iou_score
from common.mps_train import load_model_weights
from data import TemporalCropDataset
from irt_cfg import load_cfg
from irt_data import IRTDataset
from model import ConvLSTMSegNet

CKPT_BEST = HERE / "model_convlstm_best.tar"


if __name__ == "__main__":
    cfg = load_cfg(HERE / "dataset_convlstm.yaml", train=False)
    ds = TemporalCropDataset(IRTDataset(cfg), size=256)
    frames, mask = ds[random.randrange(len(ds))]
    gt = mask[0].numpy()

    in_ch = int(frames.shape[1])
    model = ConvLSTMSegNet(in_channels=in_ch)
    if CKPT_BEST.exists():
        load_model_weights(model, CKPT_BEST)
        print(f"loaded {CKPT_BEST.name}")
    else:
        print(f"WARNING: no {CKPT_BEST.name} — random weights")

    device = get_device()
    model.eval().to(device)
    bce, dice_l = torch.nn.BCEWithLogitsLoss(), SoftDiceLoss()
    y = mask.unsqueeze(0)
    x = frames.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        loss = (
            bce(logits, y.to(device)) + dice_l(logits, y.to(device))
        ).item()
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    pred = (prob > 0.5).astype(np.float32)
    d, iou = dice_score(pred, gt), iou_score(pred, gt)
    print(
        f"T={frames.shape[0]} in_ch={in_ch} size={tuple(frames.shape[-2:])} "
        f"Dice={d:.4f} IoU={iou:.4f} loss={loss:.4f}"
    )

    mid = frames.shape[0] // 2
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(frames[mid, 0].numpy(), cmap="inferno")
    ax[0].set_title(f"frame t={mid}")
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
