"""Extractors convert T channels to custom C channels"""

import numpy as np

from .base import Transform


class _Reducer(Transform):
    """Base class for reducers"""
    out = 1

    def out_channels(self, c_in: int = 0) -> int:
        return self.out


class MaxMin(_Reducer):
    def __call__(self, data, mask=None):
        return (data.max(axis=0) - data.min(axis=0))[None], mask


class MaxFirst(_Reducer):
    def __call__(self, data, mask=None):
        return (data.max(axis=0) - data[0])[None], mask


class Std(_Reducer):
    def __call__(self, data, mask=None):
        return data.std(axis=0)[None], mask


class PCA1(_Reducer):
    """First temporal mode (power iteration), sign aligned with max-min."""

    def __init__(self, iters: int = 8, eps: float = 1e-6):
        self.iters = iters
        self.eps = eps

    def __call__(self, data, mask=None):
        T, H, W = data.shape
        x = data.reshape(T, -1).astype(np.float32)
        x = x - x.mean(axis=0, keepdims=True)                # центрируем по времени

        ref = (data.max(axis=0) - data.min(axis=0)).reshape(-1)   # max-min как сид
        v = ref - ref.mean()
        v = v / max(np.linalg.norm(v), self.eps)
        for _ in range(self.iters):
            u = x @ v
            u = u / max(np.linalg.norm(u), self.eps)
            v = x.T @ u
            v = v / max(np.linalg.norm(v), self.eps)
        score = (x.T @ (x @ v)).reshape(H, W)

        centred = score.reshape(-1) - score.mean()
        if np.dot(centred, ref - ref.mean()) < 0:            # знак: корреляция с max-min
            score = -score
        return score[None], mask


class TSR(_Reducer):
    """Coefficients of log-time polynomial regression (TSR)"""

    def __init__(self, deg: int = 5):
        self.deg = deg
        self.out = deg + 1

    def __call__(self, data, mask=None):
        T, H, W = data.shape
        peak = int(data.reshape(T, -1).mean(1).argmax())
        base = data[:max(1, peak // 4)].mean(0)

        dT = np.maximum(data[peak:] - base, 1e-3)
        n = dT.shape[0]

        logt = np.log(np.arange(1, n + 1, dtype=data.dtype))
        A = np.stack([logt ** j for j in range(self.deg + 1)], axis=1)  # (n, deg+1)
        Y = np.log(dT).reshape(n, -1)                                   # (n, H*W)

        coef = np.linalg.lstsq(A, Y, rcond=None)[0]                     # (deg+1, H*W)
        return coef.reshape(self.deg + 1, H, W), mask
