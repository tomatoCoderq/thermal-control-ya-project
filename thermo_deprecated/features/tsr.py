"""TSR-признаки: коэффициенты полинома степени N по log ΔT ~ log t (фаза остывания).

Выход — (poly_degree+1, H, W). Перенесено из 5th-polynom-exps.
"""
from __future__ import annotations

import numpy as np
import scipy.io as sio


def load_raw_video(path) -> np.ndarray:
    """(T,H,W) float32 из .mat (ищем 3-мерный массив по известным ключам)."""
    m = sio.loadmat(path)
    key = next(k for k in ("imageArray", "data", "IMAGES")
               if k in m and np.asarray(m[k]).ndim == 3)
    return np.transpose(np.asarray(m[key]).astype(np.float32), (2, 0, 1))


def tsr_coeffs(path, deg: int = 5) -> np.ndarray:
    """(deg+1, H, W) — попиксельная аппроксимация log ΔT по log t на остывании."""
    X = load_raw_video(path)                                 # (n,H,W)
    n, H, W = X.shape
    peak = int(np.argmax(X.reshape(n, -1).mean(1)))
    base = X[:max(1, peak // 4)].mean(0)
    dT = np.clip(X[peak:] - base[None], 1e-3, None)
    tc = (np.arange(dT.shape[0]) + 1).astype(np.float32)
    logt = np.log(tc)
    coef = np.polyfit(logt, np.log(dT).reshape(len(tc), -1), deg)
    return coef.reshape(deg + 1, H, W).astype(np.float32)
