"""Feature extractors: collapse temporal axis into channel maps (TSR / PCA)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.decomposition import PCA

from irt_data.config import FeatureConfig

logger = logging.getLogger(__name__)


class FeatureExtractor(Protocol):
    """Extract spatial feature maps from a video (T, H, W)."""

    name: str

    def __call__(self, video: np.ndarray) -> np.ndarray:
        """Return (H, W, C) float32 feature stack."""
        ...

    def num_channels(self) -> int: ...


def _preprocess(
    video: np.ndarray,
    frame_step: int,
    max_frames: int | None,
    thermal_diff: bool,
) -> np.ndarray:
    """video (T,H,W) -> subsampled (T',H,W), optionally excess temperature."""
    v = video.astype(np.float32)
    if thermal_diff:
        v = v - v[0:1]
    if max_frames is not None:
        v = v[:max_frames]
    if frame_step > 1:
        v = v[::frame_step]
    return v


class TSRFeatureExtractor:
    """Thermographic Signal Reconstruction style features.

    Channels (5):
      0: max excess temperature
      1: time-to-peak (normalized)
      2: 1st derivative at peak (approx)
      3: mean 1st derivative over cooling half
      4: mean 2nd derivative over cooling half
    """

    name = "tsr"

    def __init__(
        self,
        frame_step: int = 4,
        max_frames: int | None = None,
        poly_degree: int = 5,
        thermal_diff: bool = True,
    ) -> None:
        self.frame_step = frame_step
        self.max_frames = max_frames
        self.poly_degree = poly_degree
        self.thermal_diff = thermal_diff

    def num_channels(self) -> int:
        return 5

    def __call__(self, video: np.ndarray) -> np.ndarray:
        v = _preprocess(video, self.frame_step, self.max_frames, self.thermal_diff)
        T, H, W = v.shape
        # reshape to (N, T)
        X = v.reshape(T, -1).T.astype(np.float64)  # (N, T)
        N = X.shape[0]

        # time axis for poly fit in log-log (avoid non-positive)
        t = np.arange(1, T + 1, dtype=np.float64)
        log_t = np.log(t)
        # ensure positive excess for log; shift per-pixel
        X_pos = X - X.min(axis=1, keepdims=True) + 1e-3
        log_x = np.log(X_pos)

        deg = min(self.poly_degree, T - 1)
        # Vandermonde (T, deg+1)
        A = np.vander(log_t, N=deg + 1, increasing=True)
        # solve A @ coef.T ≈ log_x.T  -> coef (N, deg+1)
        coef, *_ = np.linalg.lstsq(A, log_x.T, rcond=None)
        coef = coef.T  # (N, deg+1)

        # reconstruct and derivatives on log_t grid
        # d/d(log_t) of poly, then convert approx to d/dt via chain rule later if needed
        fitted = A @ coef.T  # (T, N)
        fitted = fitted.T  # (N, T)

        # polynomial derivative coeffs
        # p = c0 + c1 z + c2 z^2 + ... ; z = log_t
        dcoef = np.zeros_like(coef)
        for k in range(1, deg + 1):
            dcoef[:, k - 1] = coef[:, k] * k
        dA = np.vander(log_t, N=deg, increasing=True) if deg >= 1 else np.ones((T, 1))
        if deg >= 1:
            d1 = (dA @ dcoef[:, :deg].T).T  # (N, T)
        else:
            d1 = np.zeros((N, T))

        d2coef = np.zeros_like(dcoef)
        for k in range(1, deg):
            d2coef[:, k - 1] = dcoef[:, k] * k
        if deg >= 2:
            d2A = np.vander(log_t, N=deg - 1, increasing=True)
            d2 = (d2A @ d2coef[:, : deg - 1].T).T
        else:
            d2 = np.zeros((N, T))

        # also use raw excess stats (more stable)
        max_dt = X.max(axis=1)
        ttp = X.argmax(axis=1).astype(np.float64) / max(T - 1, 1)

        half = T // 2
        mean_d1 = d1[:, half:].mean(axis=1) if half < T else d1.mean(axis=1)
        mean_d2 = d2[:, half:].mean(axis=1) if half < T else d2.mean(axis=1)
        # derivative at peak index
        peak_idx = X.argmax(axis=1)
        d1_at_peak = d1[np.arange(N), peak_idx]

        feats = np.stack(
            [
                max_dt,
                ttp,
                d1_at_peak,
                mean_d1,
                mean_d2,
            ],
            axis=-1,
        ).astype(np.float32)
        return feats.reshape(H, W, 5)


class PCAFeatureExtractor:
    """Principal Component Thermography: first k score maps."""

    name = "pca"

    def __init__(
        self,
        n_components: int = 3,
        frame_step: int = 4,
        max_frames: int | None = None,
        thermal_diff: bool = True,
        center_pixels: bool = True,
    ) -> None:
        self.n_components = n_components
        self.frame_step = frame_step
        self.max_frames = max_frames
        self.thermal_diff = thermal_diff
        self.center_pixels = center_pixels

    def num_channels(self) -> int:
        return self.n_components

    def __call__(self, video: np.ndarray) -> np.ndarray:
        v = _preprocess(video, self.frame_step, self.max_frames, self.thermal_diff)
        T, H, W = v.shape
        X = v.reshape(T, -1).T.astype(np.float32)  # (N, T)
        if self.center_pixels:
            X = X - X.mean(axis=1, keepdims=True)
        k = min(self.n_components, X.shape[0], X.shape[1])
        pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
        scores = pca.fit_transform(X)  # (N, k)
        if k < self.n_components:
            pad = np.zeros((scores.shape[0], self.n_components - k), dtype=np.float32)
            scores = np.concatenate([scores, pad], axis=1)
        return scores.reshape(H, W, self.n_components).astype(np.float32)


class CompositeFeatureExtractor:
    """Concatenate several extractors along channel axis."""

    name = "composite"

    def __init__(self, extractors: list[FeatureExtractor]) -> None:
        if not extractors:
            raise ValueError("CompositeFeatureExtractor needs at least one extractor")
        self.extractors = extractors

    def num_channels(self) -> int:
        return sum(e.num_channels() for e in self.extractors)

    def __call__(self, video: np.ndarray) -> np.ndarray:
        parts = [e(video) for e in self.extractors]
        return np.concatenate(parts, axis=-1).astype(np.float32)


FEATURE_REGISTRY: dict[str, type] = {
    "tsr": TSRFeatureExtractor,
    "pca": PCAFeatureExtractor,
}


def build_feature_extractor(cfg: FeatureConfig) -> FeatureExtractor:
    extractors: list[FeatureExtractor] = []
    for name in cfg.extractors:
        if name not in FEATURE_REGISTRY:
            raise KeyError(
                f"Unknown feature extractor '{name}'. Available: {list(FEATURE_REGISTRY)}"
            )
        cls = FEATURE_REGISTRY[name]
        if name == "tsr":
            extractors.append(
                cls(
                    frame_step=cfg.frame_step,
                    max_frames=cfg.max_frames,
                    poly_degree=cfg.poly_degree,
                    thermal_diff=cfg.thermal_diff,
                )
            )
        elif name == "pca":
            extractors.append(
                cls(
                    n_components=cfg.pca_components,
                    frame_step=cfg.frame_step,
                    max_frames=cfg.max_frames,
                    thermal_diff=cfg.thermal_diff,
                )
            )
        else:
            extractors.append(cls())
    if len(extractors) == 1:
        return extractors[0]
    return CompositeFeatureExtractor(extractors)


def _cfg_hash(cfg: FeatureConfig) -> str:
    payload = json.dumps(
        {
            "extractors": cfg.extractors,
            "frame_step": cfg.frame_step,
            "max_frames": cfg.max_frames,
            "poly_degree": cfg.poly_degree,
            "pca_components": cfg.pca_components,
            "thermal_diff": cfg.thermal_diff,
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:10]


class CachedFeatureExtractor:
    """Disk-cached wrapper around a FeatureExtractor."""

    def __init__(
        self,
        base: FeatureExtractor,
        cfg: FeatureConfig,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.base = base
        self.cfg = cfg
        self.cache_dir = Path(cache_dir or cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hash = _cfg_hash(cfg)
        self.name = f"cached_{base.name}"

    def num_channels(self) -> int:
        return self.base.num_channels()

    def path_for(self, video_id: str) -> Path:
        return self.cache_dir / f"{video_id}__{self._hash}.npy"

    def __call__(self, video: np.ndarray, video_id: str | None = None) -> np.ndarray:
        if video_id is not None:
            path = self.path_for(video_id)
            if path.exists():
                return np.load(path)
            feats = self.base(video)
            np.save(path, feats.astype(np.float32))
            logger.info("cached features %s", path.name)
            return feats
        return self.base(video)
