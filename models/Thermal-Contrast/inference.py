"""Predict a defect mask for any thermal `.mat` video."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from paths import DATASETS_ROOT

import numpy as np
import torch

from channels import DEFAULT_PARAMS, NUM_CHANNELS, ChannelParams, build_channels, select_frames
from checkpoint import load_model
from common.metrics import dice_score, iou_score
from datasets import TermoDataset
from model import UNetModel
from video_io import load_video_from_mat


def load_trained_model(checkpoint_path: Path | str, device: torch.device) -> UNetModel:
    model = UNetModel(in_channels=NUM_CHANNELS, num_classes=1)
    state = load_model(checkpoint_path, model)
    print(f"loaded {Path(checkpoint_path).name}: epoch {state.epoch}, best IoU {state.best_iou:.4f}")
    return model.eval().to(device)


def find_mask_for_mat(mat_path: Path | str, root: Path | str = DATASETS_ROOT) -> Path | None:
    """Locate the mask paired with a `.mat` inside `datasets_list`, if it lives there."""
    stem = Path(mat_path).stem
    root = Path(root)
    if not root.is_dir():
        return None
    for sub in sorted(root.iterdir()):
        candidate = sub / "masks" / f"{stem}.png"
        if candidate.is_file():
            return candidate
    return None


@torch.no_grad()
def predict_channels(
    model: torch.nn.Module,
    channels: torch.Tensor,
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """`(4, H, W)` input → probability map and thresholded mask."""
    model.eval()
    logits = model(channels.unsqueeze(0).to(device))
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    return {"prob": prob, "pred": (prob > threshold).astype(np.float32)}


@torch.no_grad()
def predict_mat(
    model: torch.nn.Module,
    mat_path: Path | str,
    device: torch.device,
    *,
    mat_key: str | None = None,
    params: ChannelParams = DEFAULT_PARAMS,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Full path from a `.mat` file on disk to a predicted mask.

    Uses the same reader and the same channel extraction as training, so an
    unfamiliar video is preprocessed exactly like a training sample.
    """
    video = load_video_from_mat(mat_path, mat_key)
    selection = select_frames(video, params)
    channels = build_channels(video, selection, params)
    result = predict_channels(model, channels, device, threshold=threshold)
    result.update(
        {
            "channels": channels,
            "selection": selection,
            "video": video,
            "mat_path": str(mat_path),
        }
    )
    return result


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> list[dict[str, float | str]]:
    """Per-video Dice/IoU over a `ContrastDataset`."""
    model.eval()
    rows: list[dict[str, float | str]] = []
    for position, index in enumerate(dataset.indices):
        channels, mask = dataset.channels_and_mask(index)
        out = predict_channels(model, channels, device, threshold=threshold)
        truth = mask.squeeze().numpy()
        rows.append(
            {
                "video_id": dataset.video_ids[position],
                "dice": dice_score(out["pred"], truth),
                "iou": iou_score(out["pred"], truth),
                "gt_positive": float(truth.mean()),
                "pred_positive": float(out["pred"].mean()),
            }
        )
    return rows


def mask_for_video(mask_path: Path | str, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Read a mask PNG the way `TermoDataset` does: nearest resize, then binarize."""
    import torch.nn.functional as F
    from PIL import Image

    raw = np.asarray(Image.open(mask_path)).astype(np.float32)
    if raw.ndim == 3:
        raw = raw[..., 0]
    tensor = torch.from_numpy(raw)[None, None]
    resized = F.interpolate(tensor, size=size, mode="nearest")[0, 0]
    return (resized > 0).float().numpy()
