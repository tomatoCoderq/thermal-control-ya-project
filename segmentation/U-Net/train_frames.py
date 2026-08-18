"""Train on dataset_frames.yaml (same pipeline as train.py)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import make_loader
from main import SoftDiceLoss, UNetModel
from train import get_device

CKPT_BEST = HERE / "model_unet_frames_best.tar"
CKPT_LAST = HERE / "model_unet_frames_last.tar"

if __name__ == "__main__":
    _, ds, loader = make_loader(HERE / "dataset_frames.yaml", train=True, size=256)
    in_ch = int(ds[0][0].shape[0])
    device = get_device()
    print(f"device={device} | n={len(ds)} | in_ch={in_ch}")

    model = UNetModel(in_channels=in_ch, num_classes=1).to(device)
    opt = optim.RMSprop(model.parameters(), lr=0.001)
    bce, dice = nn.BCEWithLogitsLoss(), SoftDiceLoss()
    best_loss, best_epoch = float("inf"), -1
    model.train()
    for epoch in range(20):
        loss_mean, n = 0.0, 0
        bar = tqdm(loader, leave=True)
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = bce(pred, y) + dice(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            n += 1
            loss_mean = loss.item() / n + (1 - 1 / n) * loss_mean
            bar.set_description(f"Epoch [{epoch + 1}/20], loss={loss_mean:.3f}")
        if loss_mean < best_loss:
            best_loss, best_epoch = loss_mean, epoch + 1
            torch.save(model.state_dict(), CKPT_BEST)
            print(f"\n>>> best epoch {best_epoch}, loss={best_loss:.4f}")
    torch.save(model.state_dict(), CKPT_LAST)
