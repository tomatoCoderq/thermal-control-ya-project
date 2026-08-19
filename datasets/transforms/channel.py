"""Channel-wise transforms (z-norm, percentile, derivatives). Layout (C,H,W)."""

import numpy as np

from .base import Transform


class PerChannelZNorm(Transform):
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, data, mask=None):
        mean = data.mean(axis=(1, 2), keepdims=True)  # (C,1,1)
        std = data.std(axis=(1, 2), keepdims=True)
        z_score = (data - mean) / (std + self.eps)
        return z_score, mask


class PercentileNorm(Transform):
    def __init__(self, lo: float = 0.01, hi: float = 0.99, eps: float = 1e-6):
        self.lo, self.hi, self.eps = lo, hi, eps

    def __call__(self, data, mask=None):
        low = np.quantile(data, self.lo, axis=(1, 2), keepdims=True)
        high = np.quantile(data, self.hi, axis=(1, 2), keepdims=True)
        norm = (data - low) / np.maximum(high - low, self.eps)
        return np.clip(norm, 0.0, 1.0), mask


class AppendDerivatives(Transform):
    """
    Adds dereviatives along time axis. Thought to add to extract.py,
    but function works above changed channels so technically a transform.
    """

    def __init__(self, order: int = 1):
        self.order = order

    def __call__(self, data, mask=None):
        outs = [data]
        cur = data
        for _ in range(self.order):
            cur = np.diff(cur, axis=0)
            outs.append(cur)
        return np.concatenate(outs, axis=0), mask

    def out_channels(self, c_in: int) -> int:
        c, total = c_in, c_in
        for _ in range(self.order):
            c -= 1
            total += c
        return total
