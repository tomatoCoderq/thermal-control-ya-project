"""Video-level transforms (T,H,W) -> (T',H,W): temporal frame selection."""

import numpy as np

from .base import Transform


class SelectFrames(Transform):
    """Drop flash and leading calibration frames, then evenly subsample (as channels.py)."""

    def __init__(self, num_frames: int = 64, flash_brightness: float = 0.5,
                 flash_flatness: float = 0.5, onset_brightness: float = 0.2,
                 leading_spread: float = 3.0, eps: float = 1e-6):
        self.num_frames = num_frames
        self.flash_brightness = flash_brightness
        self.flash_flatness = flash_flatness
        self.onset_brightness = onset_brightness
        self.leading_spread = leading_spread
        self.eps = eps

    def __call__(self, data, mask=None):
        v = data.astype(np.float32)
        fmean = v.mean(axis=(1, 2))
        fstd = v.std(axis=(1, 2))

        baseline = np.median(fmean)
        amplitude = max(np.quantile(fmean, 0.99) - baseline, self.eps)
        flash = (fmean > baseline + self.flash_brightness * amplitude) & \
                (fstd < self.flash_flatness * np.median(fstd))

        onset_level = baseline + self.onset_brightness * amplitude
        hot = np.flatnonzero(fmean > onset_level)
        onset = int(hot[0]) if hot.size else fmean.size

        leading = np.zeros_like(flash)
        if onset > 1:
            cold_spread = np.median(fstd[:onset])
            for t in range(onset):
                if fstd[t] <= self.leading_spread * cold_spread:
                    break
                leading[t] = True

        kept = np.flatnonzero(~flash & ~leading)
        if kept.size < 2:
            raise ValueError("frame rejection left < 2 frames; loosen params")
        if kept.size <= self.num_frames:
            sampled = kept
        else:
            picks = np.unique(np.linspace(0, kept.size - 1, self.num_frames).round().astype(int))
            sampled = kept[picks]
        return v[sampled], mask
