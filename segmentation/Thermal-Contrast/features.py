"""Collapse a temporal IRT clip (T, H, W) into static 2D contrast maps."""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Named presets → list of channel extractors
PRESETS: dict[str, list[str]] = {
    "minimal": ["maxmin"],
    "delta": ["maxmin", "maxfirst"],
    "combo": ["maxmin", "maxfirst", "std"],
    "pca": ["maxmin", "pca1"],
    "full": ["maxmin", "maxfirst", "std", "pca1"],
}


def channel_names(preset: str | None = None, *, channels: Sequence[str] | None = None) -> list[str]:
    if channels is not None:
        return list(channels)
    if preset is None:
        preset = "combo"
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    return list(PRESETS[preset])


def _temporal_pca1(frames: np.ndarray) -> np.ndarray:
    """First temporal PCA score map (dominant heating/cooling mode)."""
    t, h, w = frames.shape
    if t < 2:
        return np.zeros((h, w), dtype=np.float32)
    x = frames.reshape(t, -1).astype(np.float64)
    x -= x.mean(axis=0, keepdims=True)
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    score = (u[:, :1].T @ x).reshape(h, w)
    ref = frames.max(axis=0) - frames.min(axis=0)
    if float(np.std(score)) > 1e-8 and float(np.std(ref)) > 1e-8:
        corr = np.corrcoef(score.ravel(), ref.ravel())[0, 1]
        if corr < 0:
            score = -score
    return score.astype(np.float32)


def collapse_temporal(frames: np.ndarray, channels: Sequence[str]) -> np.ndarray:
    """(T, H, W) float → (C, H, W) contrast stack."""
    if frames.ndim != 3:
        raise ValueError(f"expected (T,H,W), got {frames.shape}")
    t = frames.shape[0]
    f0 = frames[0]
    fmax = frames.max(axis=0)
    fmin = frames.min(axis=0)

    out: list[np.ndarray] = []
    for name in channels:
        if name == "maxmin":
            out.append(fmax - fmin)
        elif name == "maxfirst":
            out.append(fmax - f0)
        elif name == "minfirst":
            out.append(fmin - f0)
        elif name == "lastfirst":
            out.append(frames[-1] - f0)
        elif name == "std":
            out.append(frames.std(axis=0) if t > 1 else np.zeros_like(f0))
        elif name == "mean":
            out.append(frames.mean(axis=0))
        elif name == "pca1":
            out.append(_temporal_pca1(frames))
        else:
            raise ValueError(f"unknown contrast channel: {name!r}")
    return np.stack(out, axis=0).astype(np.float32)


def normalize_contrast(feat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-sample min-max over all channels (matches yaml norm: per_sample)."""
    mn = float(feat.min())
    mx = float(feat.max())
    return ((feat - mn) / (mx - mn + eps)).astype(np.float32)
