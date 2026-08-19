"""Inference: .mat thermal video → contrast maps → segmentation mask."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from irt_data.cache import _load_mat_array
from irt_data.config import DatasetConfig, FileMeta, ROI
from irt_data.crops import CropBox, ObjectCropper, RoiCropper
from irt_data.formatter import TensorFormatter
from irt_data.samplers import UniformSampler

from features import channel_names, collapse_temporal, normalize_contrast


def load_mat_video(
    mat_path: str | Path,
    *,
    time_axis: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load (T, H, W) float32 video from .mat."""
    return _load_mat_array(Path(mat_path), time_axis=time_axis)


def _resolve_meta(cfg: DatasetConfig, video_id: str) -> FileMeta | None:
    return cfg.files_meta.get(video_id)


def _source_object_roi(cfg: DatasetConfig, video_id: str) -> ROI | None:
    stem = video_id
    for src in cfg.sources:
        if src.object_roi is not None:
            if stem.startswith("sample") or "sample" in (src.pattern or ""):
                return src.object_roi
    return None


def guess_time_axis(video_id: str, cfg: DatasetConfig) -> int | None:
    """Infer time_axis from yaml sources (sample*.mat → 2)."""
    import fnmatch

    for src in cfg.sources:
        pat = src.pattern or "*.mat"
        if fnmatch.fnmatch(video_id + ".mat", pat) or fnmatch.fnmatch(video_id, pat.replace(".mat", "*")):
            return src.time_axis
    return 2 if video_id.startswith("sample") else None


def resolve_gt_mask_path(
    stem: str,
    cfg: DatasetConfig,
    root: Path,
) -> Path | None:
    """Find GT mask using the same dirs as dataset yaml (training masks)."""
    import fnmatch

    ordered: list[Path] = []
    for src in cfg.sources:
        pat = src.pattern or "*.mat"
        if fnmatch.fnmatch(stem + ".mat", pat) or fnmatch.fnmatch(stem, pat.replace(".mat", "*")):
            if src.masks:
                ordered.append(Path(src.masks))
    for extra in (
        root / "labels" / "table_mask_raw",
        root / "labels" / "tpu_binary_masks",
        root / "labels" / "kaggle_binary_masks",
    ):
        if extra not in ordered:
            ordered.append(extra)

    aliases = [stem, stem.replace("_Static", "")]
    if stem.startswith("sample") and stem[6:].isdigit():
        aliases.append(f"Sample_{stem[6:]}_Static")

    for mask_dir in ordered:
        if not mask_dir.is_absolute():
            mask_dir = root / mask_dir
        for name in aliases:
            for ext in (".png", ".npy"):
                p = mask_dir / f"{name}{ext}"
                if p.exists():
                    return p
    return None


def load_gt_mask(path: Path, out_shape: tuple[int, int]) -> np.ndarray:
    """Load binary GT mask resized to shape (naive — may misalign with model pred)."""
    import cv2
    from PIL import Image

    if path.suffix == ".npy":
        arr = np.load(path)
        if arr.dtype == np.float32 and np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)
        gt = (arr > 0).astype(np.float32)
    else:
        gt = (np.array(Image.open(path)) > 0).astype(np.float32)
    if gt.shape != out_shape:
        gt = cv2.resize(gt, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_NEAREST)
    return gt


def load_gt_mask_for_mat(
    mat_path: str | Path,
    cfg: DatasetConfig,
    out_shape: tuple[int, int],
    *,
    mask_path: Path | None = None,
    root: Path | None = None,
) -> np.ndarray | None:
    """Load GT mask through the same object_crop + center/roi crop as inference."""
    import copy

    from PIL import Image

    path = Path(mat_path)
    vid = path.stem.replace(" ", "_")
    if mask_path is None:
        if root is None:
            root = path.resolve().parents[1] if path.parent.name == "data" else path.parent
        mask_path = resolve_gt_mask_path(vid, cfg, Path(root))
    if mask_path is None or not mask_path.exists():
        return None

    raw = np.array(Image.open(mask_path))
    mask = (raw > 0).astype(np.float32)
    if mask_path.suffix == ".npy":
        arr = np.load(mask_path)
        mask = (np.nan_to_num(arr, nan=0.0) > 0).astype(np.float32)

    video, _ = load_mat_video(path, time_axis=guess_time_axis(vid, cfg))
    T, H, W = video.shape
    meta = _resolve_meta(cfg, vid)
    source_object_roi = _source_object_roi(cfg, vid)

    object_cropper = ObjectCropper(cfg.object_crop)
    object_box = object_cropper.box_for(H, W, meta, source_object_roi)
    mask = object_cropper.apply_mask(mask.astype(np.uint8), object_box)
    mask = object_cropper.apply_image(
        mask.astype(np.float32),
        CropBox(0, 0, mask.shape[0], mask.shape[1], mask.shape[0], mask.shape[1]),
    )

    eval_cfg = copy.deepcopy(cfg)
    if eval_cfg.crop.strategy == "random":
        eval_cfg.crop.strategy = "center"
    elif eval_cfg.crop.strategy.startswith("roi_"):
        eval_cfg.crop.strategy = "roi_center" if meta and meta.rois else "center"
    cropper = RoiCropper(eval_cfg.crop)
    rng = np.random.default_rng(0)
    box = cropper.plan(mask.shape[0], mask.shape[1], rng, meta)
    mask = cropper.apply_mask((mask > 0).astype(np.uint8), box)
    mask = (mask > 0).astype(np.float32)

    if mask.shape != out_shape:
        import cv2

        mask = cv2.resize(mask, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def sample_frame_indices(
    T_total: int,
    cfg: DatasetConfig,
    meta: FileMeta | None,
    *,
    frame_range: tuple[int, int | None] | None = None,
) -> np.ndarray:
    """Deterministic uniform indices (eval-style, no jitter)."""
    fr = frame_range
    if fr is None and meta is not None and meta.frame_range is not None:
        fr = meta.frame_range
    elif fr is None and cfg.temporal.frame_range is not None:
        fr = tuple(cfg.temporal.frame_range)

    if fr is not None:
        meta = meta or FileMeta()
        meta = replace(meta, frame_range=fr)

    sampler = UniformSampler(num_frames=cfg.temporal.num_frames, time_pad=cfg.temporal.time_pad)
    rng = np.random.default_rng(0)
    return sampler.sample(T_total, rng, meta)


def preprocess_temporal_clip(
    video_thw: np.ndarray,
    cfg: DatasetConfig,
    *,
    video_id: str | None = None,
    frame_indices: np.ndarray | None = None,
    meta: FileMeta | None = None,
    source_object_roi: ROI | None = None,
) -> np.ndarray:
    """Match IRTDataset temporal path → normalized (T, H, W) before contrast collapse."""
    T, H, W = video_thw.shape
    if meta is None and video_id:
        meta = _resolve_meta(cfg, video_id)
    if source_object_roi is None and video_id:
        source_object_roi = _source_object_roi(cfg, video_id)

    if frame_indices is None:
        frame_indices = sample_frame_indices(T, cfg, meta)
    frames = video_thw[frame_indices].astype(np.float32)

    object_cropper = ObjectCropper(cfg.object_crop)
    cropper = RoiCropper(cfg.crop)
    rng = np.random.default_rng(0)

    object_box = object_cropper.box_for(H, W, meta, source_object_roi)
    frames = object_cropper.apply_frames(video_thw[frame_indices], object_box)
    frames = np.stack(
        [
            object_cropper.apply_image(
                frames[t],
                CropBox(0, 0, frames.shape[1], frames.shape[2], frames.shape[1], frames.shape[2]),
            )
            for t in range(frames.shape[0])
        ],
        axis=0,
    )

    eval_cfg = copy.deepcopy(cfg)
    if eval_cfg.crop.strategy == "random":
        eval_cfg.crop.strategy = "center"
    elif eval_cfg.crop.strategy.startswith("roi_"):
        eval_cfg.crop.strategy = "roi_center"
    cropper = RoiCropper(eval_cfg.crop)

    H2, W2 = frames.shape[1:]
    box = cropper.plan(H2, W2, rng, meta)
    frames = cropper.apply_frames(frames, box)

    formatter = TensorFormatter(cfg.norm, cfg.mask)
    return formatter.normalize(frames, stats=None)


def video_to_contrast(
    video_thw: np.ndarray,
    cfg: DatasetConfig,
    *,
    preset: str = "combo",
    channels: list[str] | None = None,
    **preprocess_kw,
) -> tuple[np.ndarray, np.ndarray]:
    """Full pipeline: raw (T,H,W) → contrast (C,H,W) float32 in [0,1]."""
    clip = preprocess_temporal_clip(video_thw, cfg, **preprocess_kw)
    ch = channel_names(preset, channels=channels)
    feat = normalize_contrast(collapse_temporal(clip, ch))
    return feat, clip


def mat_to_contrast(
    mat_path: str | Path,
    cfg: DatasetConfig,
    *,
    preset: str | None = None,
    channels: list[str] | None = None,
    time_axis: int | None = None,
    video_id: str | None = None,
    frame_range: tuple[int, int | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load .mat → contrast maps (C,H,W) and normalized clip (T,H,W)."""
    path = Path(mat_path)
    vid = video_id or path.stem.replace(" ", "_")
    video, _meta = load_mat_video(path, time_axis=time_axis)
    p = preset or "combo"
    ch = channel_names(p, channels=channels)
    meta = _resolve_meta(cfg, vid)
    idx = sample_frame_indices(video.shape[0], cfg, meta, frame_range=frame_range)
    clip = preprocess_temporal_clip(
        video, cfg, video_id=vid, frame_indices=idx, meta=meta
    )
    feat = normalize_contrast(collapse_temporal(clip, ch))
    return feat, clip, video


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    feat_chw: np.ndarray,
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Contrast tensor (C,H,W) → prob / binary mask."""
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(feat_chw)).float().unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    pred = (prob > threshold).astype(np.float32)
    return {"prob": prob, "pred": pred, "logits": logits.squeeze().cpu().numpy()}


@torch.no_grad()
def predict_mat(
    model: torch.nn.Module,
    mat_path: str | Path,
    cfg: DatasetConfig,
    device: torch.device,
    *,
    preset: str | None = "combo",
    channels: list[str] | None = None,
    time_axis: int | None = None,
    frame_range: tuple[int, int | None] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """End-to-end: .mat file → segmentation + contrast maps."""
    p = preset or "combo"
    ch = channel_names(p, channels=channels)
    feat, clip, raw = mat_to_contrast(
        mat_path,
        cfg,
        preset=p,
        channels=ch,
        time_axis=time_axis,
        frame_range=frame_range,
    )
    out = predict_mask(model, feat, device, threshold=threshold)
    out.update(
        {
            "feat": feat,
            "clip": clip,
            "raw_video": raw,
            "channels": ch,
            "mat_path": str(mat_path),
        }
    )
    return out


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    loss_fn=None,
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    """Run full eval split; return mean loss/metrics + per-sample rows."""
    from common.loop import run_epoch

    if loss_fn is None:
        from common.metrics import BCEDiceLoss

        loss_fn = BCEDiceLoss()

    loss, metrics = run_epoch(
        model, loader, device, optimizer=None, loss_fn=loss_fn, desc="eval"
    )

    from common.metrics import dice_score, iou_score

    per_sample: list[dict[str, float]] = []
    ds = loader.dataset
    model.eval()
    for i in range(len(ds)):
        feat, mask = ds[i]
        prob = torch.sigmoid(model(feat.unsqueeze(0).to(device))).squeeze().cpu().numpy()
        pred = (prob > 0.5).astype(np.float32)
        gt = mask.squeeze().numpy()
        if hasattr(ds, "base"):
            vid = ds.base[i]["video_id"]
        elif hasattr(ds, "video_ids") and i < len(ds.video_ids):
            vid = ds.video_ids[i]
        else:
            vid = str(i)
        per_sample.append(
            {
                "idx": i,
                "video_id": vid,
                "dice": dice_score(pred, gt),
                "iou": iou_score(pred, gt),
                "gt_pos": float(gt.mean()),
                "pred_pos": float(pred.mean()),
            }
        )
    return loss, metrics, per_sample
