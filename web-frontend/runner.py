"""Inference glue between Thermal-Contrast and the Gradio app.

One `.mat` in, everything the UI shows out: mask overlay, defect table, the four
channels, the probability map and per-defect crops. The heavy lifting stays in
`Thermal-Contrast` (channels, weights) and in `pipeline` (`Prediction`); this file
only wires them together and turns tensors into displayable `uint8` images.
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from scipy import ndimage

# make Thermal-Contrast and repo root importable (same trick as pipeline/models.py)
_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "models" / "Thermal-Contrast", _REPO / "models"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from channels import CHANNEL_NAMES, CHANNEL_TITLES, NUM_CHANNELS, build_channels, select_frames  # noqa: E402
from checkpoint import load_model                                                                # noqa: E402
from common.device import get_device                                                             # noqa: E402
from model import UNetModel                                                                       # noqa: E402
from video_io import load_video_from_mat                                                          # noqa: E402
from pipeline.models import Defect, Prediction                                                    # noqa: E402
from datasets.transforms import (Compose, Stack, TSR, PerChannelZNorm, AppendDerivatives)         # noqa: E402

CKPT_ENV = "THERMAL_SEG_CKPT"
REG_ENV = "THERMAL_REG_CKPT"
CROP = 48

_model: UNetModel | None = None
_device: torch.device | None = None
_reg: torch.nn.Module | None = None
_depth_extract: Compose | None = None
_depth_norm: Compose | None = None


def _get_model() -> tuple[UNetModel, torch.device]:
    """Load the U-Net once from `$THERMAL_SEG_CKPT`, then reuse it."""
    global _model, _device
    if _model is None:
        ckpt = os.environ.get(CKPT_ENV)
        if not ckpt:
            raise RuntimeError(f"set {CKPT_ENV} to a Thermal-Contrast checkpoint (.pkl)")
        _device = get_device()
        model = UNetModel(in_channels=NUM_CHANNELS, num_classes=1)
        state = load_model(ckpt, model)
        print(f"loaded {Path(ckpt).name}: epoch {state.epoch}, best IoU {state.best_iou:.4f}")
        _model = model.eval().to(_device)
    return _model, _device


class _Reg(torch.nn.Module):
    """timm-бэкбон с 1-выходной головой (совместимо с backbone.* из чекпойнта)."""

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def _get_depth() -> tuple[torch.nn.Module, Compose, Compose]:
    """Load the depth regressor once from `$THERMAL_REG_CKPT` (default reg_web_...pkl)."""
    global _reg, _depth_extract, _depth_norm
    if _reg is None:
        _, device = _get_model()
        ckpt = os.environ.get(REG_ENV) or str(_REPO / "reg_web_resnet34_p5d1.pkl")
        ck = pickle.load(open(ckpt, "rb"))
        backbone = timm.create_model(ck["model_name"], pretrained=False,
                                     in_chans=ck["in_channels"], num_classes=1)
        reg = _Reg(backbone)
        reg.load_state_dict(ck["model_state"], strict=True)
        _reg = reg.eval().to(device)
        _depth_extract = Compose([Stack([TSR(ck["poly_degree"])])])
        _depth_norm = Compose([PerChannelZNorm(), AppendDerivatives(1)])
        print(f"loaded {Path(ckpt).name}: depth {ck['model_name']} in_ch={ck['in_channels']}")
    return _reg, _depth_extract, _depth_norm


@torch.no_grad()
def _defects(mask: np.ndarray, video: np.ndarray) -> list[Defect]:
    """Connected components → a Defect at each centroid, depth (mm) from the regressor."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return []
    reg, extract, norm = _get_depth()
    _, device = _get_model()
    feats, _ = extract(video)                                # (C,H,W) TSR на полном кадре
    h, w = mask.shape
    meta, crops = [], []
    for k in range(1, n + 1):
        rr, cc = ndimage.center_of_mass(labels == k)
        x, y = int(cc), int(rr)
        r0 = int(np.clip(y - CROP // 2, 0, max(h - CROP, 0)))
        c0 = int(np.clip(x - CROP // 2, 0, max(w - CROP, 0)))
        meta.append((k, x, y))
        crops.append(norm(feats[:, r0:r0 + CROP, c0:c0 + CROP])[0])   # per-crop norm+deriv
    batch = torch.from_numpy(np.ascontiguousarray(np.stack(crops))).float().to(device)
    depths = reg(batch).reshape(-1).cpu().numpy().tolist()
    return [Defect(x=x, y=y, depth_mm=float(d), region_id=k)
            for (k, x, y), d in zip(meta, depths)]


@torch.no_grad()
def predict(mat_path: str | Path, mat_key: str | None = None,
            threshold: float = 0.5) -> tuple[Prediction, np.ndarray]:
    """`.mat` → (`Prediction`, channels `(4,H,W)`). Same preprocessing as training."""
    model, device = _get_model()
    video = load_video_from_mat(mat_path, mat_key)
    channels = build_channels(video, select_frames(video))
    logits = model(channels.unsqueeze(0).to(device))
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    pred = Prediction(mask=mask, defects=_defects(mask, video.numpy()), prob=prob,
                      size=tuple(video.shape[1:]))
    return pred, channels.numpy()


# --- rendering -------------------------------------------------------------

def _u8(img: np.ndarray) -> np.ndarray:
    """Min-max stretch a 2-D map to uint8."""
    lo, hi = float(img.min()), float(img.max())
    return ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)


def _gray_rgb(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=-1)


def _window(d: Defect, crop: int, shape: tuple[int, int]) -> tuple[int, int]:
    """Top-left of a `crop`-sized box centred on the defect (clamped in-frame)."""
    s, (h, w) = crop, shape
    r0 = int(np.clip(d.y - s // 2, 0, max(h - s, 0)))
    c0 = int(np.clip(d.x - s // 2, 0, max(w - s, 0)))
    return r0, c0


def overlay(channels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """maxmin channel in gray with the mask painted red."""
    rgb = _gray_rgb(_u8(channels[0]))
    rgb[mask.astype(bool)] = (255, 0, 0)
    return rgb


def gallery(channels: np.ndarray, prob: np.ndarray,
            defects: list[Defect], crop: int = CROP) -> list[tuple[np.ndarray, str]]:
    """The four channels, the probability heatmap and one crop per defect."""
    items = [(_gray_rgb(_u8(ch)), CHANNEL_TITLES.get(name, name))
             for name, ch in zip(CHANNEL_NAMES, channels)]
    items.append((cv2.applyColorMap(_u8(prob), cv2.COLORMAP_INFERNO)[..., ::-1], "prob"))
    for d in defects:
        r0, c0 = _window(d, crop, channels[0].shape)
        items.append((_gray_rgb(_u8(channels[0, r0:r0 + crop, c0:c0 + crop])),
                      f"defect {d.region_id}"))
    return items


def run(file, mat_key: str, threshold: float):
    """Gradio callback: uploaded file → (overlay, table rows, gallery, download files)."""
    if file is None:
        return None, [], [], []
    pred, channels = predict(file.name, mat_key or None, threshold)
    rows = [[d.region_id, d.x, d.y, "—" if d.depth_mm is None else round(d.depth_mm, 3)]
            for d in pred.defects]
    out = Path(tempfile.mkdtemp())
    pred.save(out)
    files = [str(out / "mask.npy"), str(out / "depth.txt"), str(out / "meta.json")]
    return overlay(channels, pred.mask), rows, gallery(channels, pred.prob, pred.defects), files
