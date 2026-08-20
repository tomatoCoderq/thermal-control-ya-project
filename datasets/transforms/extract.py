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


def _tsr_fit(data, deg: int):
    """log ΔT ≈ Σ c_k (log t)^k on the cooling tail. Returns (deg+1, H, W), n."""
    T, H, W = data.shape
    peak = int(data.reshape(T, -1).mean(1).argmax())
    base = data[:max(1, peak // 4)].mean(0)
    dT = np.maximum(data[peak:] - base, 1e-3)
    n = dT.shape[0]
    logt = np.log(np.arange(1, n + 1, dtype=data.dtype))
    A = np.stack([logt ** j for j in range(deg + 1)], axis=1)
    Y = np.log(dT).reshape(n, -1)
    coef = np.linalg.lstsq(A, Y, rcond=None)[0]
    return coef.reshape(deg + 1, H, W), n


def _poly_deriv(coef, u: float, order: int):
    """d^order/du^order of Σ c_k u^k. coef is (deg+1, H, W), low-to-high."""
    deg = coef.shape[0] - 1
    out = np.zeros(coef.shape[1:], dtype=coef.dtype)
    for k in range(order, deg + 1):
        fall = 1.0
        for i in range(order):
            fall *= k - i
        out += fall * coef[k] * (u ** (k - order))
    return out


class TSR(_Reducer):
    """Coefficients of log-time polynomial regression (TSR)"""

    def __init__(self, deg: int = 5):
        self.deg = deg
        self.out = deg + 1

    def __call__(self, data, mask=None):
        coef, _ = _tsr_fit(data, self.deg)
        return coef, mask


class TSRDeriv(_Reducer):
    """1st/2nd derivative of the TSR polynomial (Chulkov Seq 5 / Seq 6).

    p(u)=Σ c_k u^k, u=log t on cooling.

    * ``at=None`` (default) — one map: max over cooling of |p^{(order)}|
    * ``at=(f1, f2, ...)`` — one map per fraction of the cooling length
    """

    def __init__(self, order: int = 1, deg: int = 5, at=None):
        self.deg = deg
        self.order = order
        self.at = at
        self.out = 1 if at is None else len(at)

    def __call__(self, data, mask=None):
        coef, n = _tsr_fit(data, self.deg)
        if self.at is None:
            peak = None
            for k in range(1, n + 1):
                d = _poly_deriv(coef, float(np.log(k)), self.order)
                peak = np.abs(d) if peak is None else np.maximum(peak, np.abs(d))
            return peak[None], mask
        maps = []
        for f in self.at:
            k = min(max(2, int(round(n * f))), n)
            maps.append(_poly_deriv(coef, float(np.log(k)), self.order))
        return np.stack(maps, axis=0), mask
