"""Frame sampling strategies for temporal mode."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from irt_data.config import FileMeta, TemporalConfig, TimePad


class FrameSampler(Protocol):
    """Select frame indices from a video of length T_total."""

    def sample(
        self,
        T_total: int,
        rng: np.random.Generator,
        meta: FileMeta | None = None,
    ) -> np.ndarray:
        """Return int64 indices of shape (T,) possibly after temporal padding."""
        ...


def _clip_range(
    T_total: int,
    meta: FileMeta | None = None,
    default_range: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Resolve frame_range → [start, end) clipped to video length.

    Priority: per-file meta.frame_range → temporal.frame_range → full video.
    """
    fr = None
    if meta is not None and meta.frame_range is not None:
        fr = meta.frame_range
    elif default_range is not None:
        fr = default_range

    if fr is None:
        return 0, T_total

    start = max(0, int(fr[0]))
    end = min(T_total, int(fr[1])) if fr[1] is not None else T_total
    if end <= start:
        return 0, T_total
    return start, end


def pad_indices(
    indices: np.ndarray,
    num_frames: int,
    mode: TimePad = "repeat_last",
) -> np.ndarray:
    """Pad or truncate indices to exactly num_frames."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == num_frames:
        return indices
    if len(indices) > num_frames:
        return indices[:num_frames]
    if len(indices) == 0:
        return np.zeros(num_frames, dtype=np.int64)
    need = num_frames - len(indices)
    if mode == "repeat_last":
        pad = np.full(need, indices[-1], dtype=np.int64)
        return np.concatenate([indices, pad])
    # reflect
    mirrored = indices[::-1]
    parts = [indices]
    while sum(len(p) for p in parts) < num_frames:
        parts.append(mirrored if len(parts) % 2 == 1 else indices)
    out = np.concatenate(parts)[:num_frames]
    return out.astype(np.int64)


def apply_temporal_jitter(
    indices: np.ndarray,
    T_total: int,
    jitter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if jitter <= 0:
        return indices
    noise = rng.integers(-jitter, jitter + 1, size=len(indices))
    return np.clip(indices.astype(np.int64) + noise, 0, T_total - 1)


def apply_frame_drop(
    indices: np.ndarray,
    drop_p: float,
    rng: np.random.Generator,
    time_pad: TimePad = "repeat_last",
) -> np.ndarray:
    """Randomly drop frames and pad back to original length."""
    if drop_p <= 0 or len(indices) == 0:
        return indices
    keep = rng.random(len(indices)) > drop_p
    if not keep.any():
        keep[rng.integers(0, len(indices))] = True
    kept = indices[keep]
    return pad_indices(kept, len(indices), mode=time_pad)


class UniformSampler:
    """Evenly spaced frames across the (optional) frame_range."""

    def __init__(self, num_frames: int = 20, time_pad: TimePad = "repeat_last") -> None:
        self.num_frames = num_frames
        self.time_pad = time_pad

    def sample(
        self,
        T_total: int,
        rng: np.random.Generator,
        meta: FileMeta | None = None,
    ) -> np.ndarray:
        start, end = _clip_range(T_total, meta)
        length = end - start
        if length <= 0:
            return np.zeros(self.num_frames, dtype=np.int64)
        if length >= self.num_frames:
            idx = np.linspace(start, end - 1, self.num_frames)
            return np.round(idx).astype(np.int64)
        raw = np.arange(start, end, dtype=np.int64)
        return pad_indices(raw, self.num_frames, self.time_pad)


class WindowSampler:
    """Random contiguous window of K frames (stride applied)."""

    def __init__(
        self,
        num_frames: int = 30,
        stride: int = 1,
        window_size: int | None = None,
        time_pad: TimePad = "repeat_last",
    ) -> None:
        self.num_frames = num_frames
        self.stride = max(1, stride)
        self.window_size = window_size or num_frames
        self.time_pad = time_pad

    def sample(
        self,
        T_total: int,
        rng: np.random.Generator,
        meta: FileMeta | None = None,
    ) -> np.ndarray:
        start, end = _clip_range(T_total, meta)
        length = end - start
        span = (self.window_size - 1) * self.stride + 1
        if length <= 0:
            return np.zeros(self.num_frames, dtype=np.int64)
        if length < span:
            raw = np.arange(start, end, dtype=np.int64)
            # subsample / pad to num_frames
            if len(raw) >= self.num_frames:
                pick = np.linspace(0, len(raw) - 1, self.num_frames)
                return np.round(pick).astype(np.int64)
            return pad_indices(raw, self.num_frames, self.time_pad)

        max_start = end - span
        w0 = int(rng.integers(start, max_start + 1))
        idx = w0 + np.arange(self.window_size) * self.stride
        if self.window_size == self.num_frames:
            return idx.astype(np.int64)
        # resample window to num_frames
        pick = np.linspace(0, len(idx) - 1, self.num_frames)
        return idx[np.round(pick).astype(np.int64)]


class KeypointSampler:
    """Sample around heat_start / cool_start / peak_contrast from FileMeta."""

    def __init__(self, num_frames: int = 20, time_pad: TimePad = "repeat_last") -> None:
        self.num_frames = num_frames
        self.time_pad = time_pad

    def sample(
        self,
        T_total: int,
        rng: np.random.Generator,
        meta: FileMeta | None = None,
    ) -> np.ndarray:
        start, end = _clip_range(T_total, meta)
        keypoints: list[int] = []
        if meta is not None:
            for attr in ("heat_start", "cool_start", "peak_contrast"):
                val = getattr(meta, attr, None)
                if val is not None:
                    keypoints.append(int(np.clip(val, start, end - 1)))
        if not keypoints:
            # fallback: early / mid / late
            keypoints = [
                start,
                start + (end - start) // 4,
                start + (end - start) // 2,
                end - 1,
            ]
        keypoints = sorted(set(int(np.clip(k, 0, T_total - 1)) for k in keypoints))

        # expand each keypoint with a small neighborhood, then uniform across unique
        neighborhood: list[int] = []
        radius = max(1, (end - start) // max(self.num_frames, 1))
        for k in keypoints:
            lo = max(start, k - radius)
            hi = min(end, k + radius + 1)
            neighborhood.extend(range(lo, hi))
        neighborhood = sorted(set(neighborhood))
        if len(neighborhood) >= self.num_frames:
            pick = np.linspace(0, len(neighborhood) - 1, self.num_frames)
            return np.array(neighborhood, dtype=np.int64)[np.round(pick).astype(np.int64)]
        return pad_indices(np.asarray(neighborhood, dtype=np.int64), self.num_frames, self.time_pad)


SAMPLER_REGISTRY: dict[str, type] = {
    "uniform": UniformSampler,
    "window": WindowSampler,
    "keypoints": KeypointSampler,
}


class SamplingPipeline:
    """Sampler + optional jitter / frame-drop (train only)."""

    def __init__(
        self,
        sampler: FrameSampler,
        cfg: TemporalConfig,
        train: bool = True,
    ) -> None:
        self.sampler = sampler
        self.cfg = cfg
        self.train = train

    def _meta_with_defaults(self, meta: FileMeta | None) -> FileMeta | None:
        """Fill frame_range from TemporalConfig when per-file unset."""
        from dataclasses import replace

        if meta is not None and meta.frame_range is not None:
            return meta
        if self.cfg.frame_range is None:
            return meta
        base = meta if meta is not None else FileMeta()
        return replace(base, frame_range=tuple(self.cfg.frame_range))

    def __call__(
        self,
        T_total: int,
        rng: np.random.Generator,
        meta: FileMeta | None = None,
    ) -> np.ndarray:
        meta = self._meta_with_defaults(meta)
        idx = self.sampler.sample(T_total, rng, meta)
        if self.train:
            idx = apply_temporal_jitter(idx, T_total, self.cfg.jitter, rng)
            idx = apply_frame_drop(idx, self.cfg.frame_drop_p, rng, self.cfg.time_pad)
        return np.asarray(idx, dtype=np.int64)

def build_frame_sampler(cfg: TemporalConfig, train: bool = True) -> SamplingPipeline:
    name = cfg.sampler
    if name not in SAMPLER_REGISTRY:
        raise KeyError(f"Unknown sampler '{name}'. Available: {list(SAMPLER_REGISTRY)}")
    cls = SAMPLER_REGISTRY[name]
    if name == "window":
        base = cls(
            num_frames=cfg.num_frames,
            stride=cfg.stride,
            window_size=cfg.window_size,
            time_pad=cfg.time_pad,
        )
    else:
        base = cls(num_frames=cfg.num_frames, time_pad=cfg.time_pad)
    return SamplingPipeline(base, cfg, train=train)
