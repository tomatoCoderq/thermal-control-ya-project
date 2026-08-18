"""Fourier phase-feature extractor (portированный из fourier-exps) для регрессии.
Выдаёт (C,H,W) фазограммы низких частот — тот же формат, что TSR-фичи, поэтому
подключается в RegressionDataset как ещё один feature-dir (mode=p5, каналы как есть).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import fft, signal
from scipy.io import loadmat


def _parse_float(value, default: float) -> float:
    try:
        parsed = float(np.asarray(value).squeeze())
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def load_thermal_video(path: str | Path) -> tuple[np.ndarray, float]:
    """Load H x W x frames thermal video and its sampling frequency."""
    mat = loadmat(path, squeeze_me=True)
    key = next(
        (name for name in ("imageArray", "data", "IMAGES")
         if name in mat and np.asarray(mat[name]).ndim == 3),
        None,
    )
    if key is None:
        available = [name for name in mat if not name.startswith("__")]
        raise KeyError(f"3-D thermal array not found in {path}; variables: {available}")
    video = np.asarray(mat[key])
    if video.ndim != 3:
        raise ValueError(f"expected H x W x frames, got {video.shape}")
    return video, _parse_float(mat.get("Fs"), default=1.0)


def peak_frame(video: np.ndarray, row_chunk: int = 32) -> int:
    """Find the maximum of the spatially averaged temperature curve."""
    total = np.zeros(video.shape[-1], dtype=np.float64)
    pixels = 0
    for r0 in range(0, video.shape[0], row_chunk):
        block = np.asarray(video[r0:r0 + row_chunk], dtype=np.float32)
        total += block.sum(axis=(0, 1), dtype=np.float64)
        pixels += block.shape[0] * block.shape[1]
    return int(np.argmax(total / max(pixels, 1)))


def _prepare_block(
    block: np.ndarray,
    window: Literal["none", "hann"],
    detrend: Literal["none", "constant", "linear"],
) -> np.ndarray:
    block = np.asarray(block, dtype=np.float32)
    if detrend != "none":
        block = signal.detrend(block, axis=-1, type=detrend).astype(np.float32, copy=False)
    if window == "hann":
        weights = signal.windows.hann(block.shape[-1], sym=False).astype(np.float32)
        block = block * weights
    return block


def fourier_phase_features(
    video: np.ndarray,
    *,
    first_bin: int = 1,
    n_frequencies: int = 8,
    phase_encoding: Literal["raw", "sin_cos"] = "raw",
    window: Literal["none", "hann"] = "none",
    detrend: Literal["none", "constant", "linear"] = "none",
    start_at_peak: bool = False,
    row_chunk: int = 32,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build low-frequency phasegrams from an H x W x frames video.

    The returned tensor is C x H x W. With ``raw`` encoding, each selected
    FFT bin contributes one phase channel in [-pi, pi]. With ``sin_cos``, each
    bin contributes sin(phase) and cos(phase), avoiding the -pi/pi jump.
    """
    if video.ndim != 3:
        raise ValueError(f"expected H x W x frames, got {video.shape}")
    if first_bin < 1:
        raise ValueError("first_bin must be >= 1 because bin 0 is DC")
    if n_frequencies < 1 or row_chunk < 1:
        raise ValueError("n_frequencies and row_chunk must be positive")

    start = peak_frame(video, row_chunk) if start_at_peak else 0
    n_frames = video.shape[-1] - start
    bins = np.arange(first_bin, first_bin + n_frequencies, dtype=np.int64)
    if bins[-1] >= n_frames // 2 + 1:
        raise ValueError(
            f"requested FFT bin {bins[-1]}, but only {n_frames} frames are available"
        )

    channels_per_bin = 2 if phase_encoding == "sin_cos" else 1
    features = np.empty(
        (n_frequencies * channels_per_bin, video.shape[0], video.shape[1]),
        dtype=np.float32,
    )

    for r0 in range(0, video.shape[0], row_chunk):
        r1 = min(r0 + row_chunk, video.shape[0])
        block = _prepare_block(video[r0:r1, :, start:], window, detrend)
        spectrum = fft.rfft(block, axis=-1, workers=1)
        phase = np.angle(spectrum[..., bins]).astype(np.float32)
        phase = np.moveaxis(phase, -1, 0)  # frequencies x rows x columns
        if phase_encoding == "raw":
            features[:, r0:r1] = phase
        elif phase_encoding == "sin_cos":
            features[0::2, r0:r1] = np.sin(phase)
            features[1::2, r0:r1] = np.cos(phase)
        else:
            raise ValueError(f"unknown phase_encoding: {phase_encoding}")

    return features, bins, start


