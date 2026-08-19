"""Collapse a thermal video into static maps that feed the U-Net.

Based on Chulkov et al. (2019) "Optimizing input data for training an ANN...":
The best results for defect depth evaluation were achieved using:
- Seq 5: First logarithmic derivative of polynomial fit (TSR)
- Seq 6: Second logarithmic derivative of polynomial fit (TSR)
- Seq 7: PCA (1st component)
All these methods require NO reference points, solving uneven heating issues.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import torch

CHANNEL_NAMES: tuple[str, ...] = ("maxmin", "std", "pca1", "tsr_d1", "tsr_d2")
NUM_CHANNELS: int = len(CHANNEL_NAMES)

CHANNEL_TITLES: dict[str, str] = {
    "maxmin": "max - min",
    "std": "\u03c3(t)",
    "pca1": "PCA\u2081(t)",
    "tsr_d1": "TSR d/dln(t)",
    "tsr_d2": "TSR d\u00b2/dln(t)\u00b2",
}


@dataclass(frozen=True)
class ChannelParams:
    """Parameters optimized based on Chulkov et al. (2019)."""

    num_frames: int = 128
    flash_brightness: float = 0.5
    flash_flatness: float = 0.5
    onset_brightness: float = 0.2
    leading_spread: float = 3.0
    pca_iters: int = 8
    tsr_poly_degree: int = 5
    norm_percentile: float = 0.01
    eps: float = 1e-6

    @property
    def key(self) -> str:
        payload = repr(sorted(asdict(self).items())) + repr(CHANNEL_NAMES)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


DEFAULT_PARAMS = ChannelParams()


@dataclass(frozen=True)
class FrameSelection:
    mean: torch.Tensor = field(repr=False)
    std: torch.Tensor = field(repr=False)
    onset: int
    flash: torch.Tensor = field(repr=False)
    leading: torch.Tensor = field(repr=False)
    keep: torch.Tensor = field(repr=False)
    sampled: torch.Tensor = field(repr=False)

    @property
    def num_kept(self) -> int:
        return int(self.keep.sum())

    @property
    def num_flash(self) -> int:
        return int(self.flash.sum())

    @property
    def num_leading(self) -> int:
        return int(self.leading.sum())

    def summary(self) -> str:
        return (
            f"T={self.keep.numel()} onset={self.onset} "
            f"flash={self.num_flash} leading={self.num_leading} "
            f"kept={self.num_kept} sampled={self.sampled.numel()}"
        )


def _as_video(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 3:
        raise ValueError(f"expected a (T, H, W) video, got {tuple(video.shape)}")
    if video.shape[0] < 2:
        raise ValueError(f"need at least 2 frames, got {video.shape[0]}")
    return video.float()


def select_frames(video: torch.Tensor, params: ChannelParams = DEFAULT_PARAMS) -> FrameSelection:
    video = _as_video(video)
    frame_mean = video.mean(dim=(1, 2))
    frame_std = video.std(dim=(1, 2))

    baseline = frame_mean.median()
    peak = torch.quantile(frame_mean, 0.99)
    amplitude = (peak - baseline).clamp_min(params.eps)

    flash = (frame_mean > baseline + params.flash_brightness * amplitude) & (
        frame_std < params.flash_flatness * frame_std.median()
    )

    onset_level = baseline + params.onset_brightness * amplitude
    hot = (frame_mean > onset_level).nonzero().flatten()
    onset = int(hot[0]) if hot.numel() else frame_mean.numel()

    leading = torch.zeros_like(flash)
    if onset > 1:
        cold_spread = frame_std[:onset].median()
        for t in range(onset):
            if frame_std[t] <= params.leading_spread * cold_spread:
                break
            leading[t] = True

    keep = ~flash & ~leading
    kept = keep.nonzero().flatten()
    if kept.numel() < 2:
        raise ValueError(f"frame rejection left {kept.numel()} frames; loosen ChannelParams")

    if kept.numel() <= params.num_frames:
        sampled = kept
    else:
        picks = torch.linspace(0, kept.numel() - 1, params.num_frames).round().long()
        sampled = kept[picks.unique()]

    return FrameSelection(
        mean=frame_mean,
        std=frame_std,
        onset=onset,
        flash=flash,
        leading=leading,
        keep=keep,
        sampled=sampled,
    )


def temporal_pca1(frames: torch.Tensor, reference: torch.Tensor, params: ChannelParams) -> torch.Tensor:
    t, h, w = frames.shape
    x = frames.reshape(t, -1).clone()
    x -= x.mean(dim=0, keepdim=True)

    v = reference.reshape(-1) - reference.mean()
    v = v / v.norm().clamp_min(params.eps)
    for _ in range(params.pca_iters):
        u = x @ v
        u = u / u.norm().clamp_min(params.eps)
        v = x.transpose(0, 1) @ u
        v = v / v.norm().clamp_min(params.eps)

    u = x @ v
    u = u / u.norm().clamp_min(params.eps)
    score = (x.transpose(0, 1) @ u).reshape(h, w)

    flat_score = score.reshape(-1)
    flat_ref = reference.reshape(-1)
    if flat_score.std() > params.eps and flat_ref.std() > params.eps:
        centred = flat_score - flat_score.mean()
        if torch.dot(centred, flat_ref - flat_ref.mean()) < 0:
            score = -score
    return score


def _tsr_excess_on_cooling(frames: torch.Tensor, params: ChannelParams) -> torch.Tensor:
    """Excess temperature on the cooling half, as in `irt_data.features.TSRCoeffsFeatureExtractor`."""
    t = frames.shape[0]
    frame_mean = frames.reshape(t, -1).mean(dim=1)
    peak = int(frame_mean.argmax())
    cool = frames[peak:]
    if cool.shape[0] < params.tsr_poly_degree + 2:
        cool = frames
        peak = 0
    base_end = max(1, peak // 4)
    base = frames[:base_end].mean(dim=0) if peak > 0 else frames[0]
    return (cool - base.unsqueeze(0)).clamp_min(params.eps)


def compute_tsr_derivatives(
    frames: torch.Tensor,
    params: ChannelParams = DEFAULT_PARAMS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1st/2nd ln(t)-derivatives of a deg-5 log-log polynomial fit (Chulkov et al., 2019).

    Polynomial fit follows `irt_data.features.TSRFeatureExtractor` / `thermo.features.tsr`:
    log(excess T) ~ poly(log t) on the cooling phase. Returns per-pixel max |derivative|
    over time, where defect contrast is usually strongest.
    """
    dT = _tsr_excess_on_cooling(frames, params)
    t_count, h, w = dT.shape
    device = dT.device
    dtype = dT.dtype

    log_t = torch.log(torch.arange(1, t_count + 1, device=device, dtype=dtype))
    log_x = torch.log(dT)
    degree = min(params.tsr_poly_degree, t_count - 1)

    basis = torch.stack([log_t**i for i in range(degree + 1)], dim=1)
    targets = log_x.reshape(t_count, -1)
    coeffs = torch.linalg.lstsq(basis, targets).solution.reshape(degree + 1, h, w)

    deriv1 = torch.zeros(t_count, h, w, device=device, dtype=dtype)
    for power in range(1, degree + 1):
        deriv1 += power * coeffs[power] * (log_t[:, None, None] ** (power - 1))

    deriv2 = torch.zeros(t_count, h, w, device=device, dtype=dtype)
    for power in range(2, degree + 1):
        deriv2 += power * (power - 1) * coeffs[power] * (log_t[:, None, None] ** (power - 2))

    return deriv1.abs().max(dim=0).values, deriv2.abs().max(dim=0).values


def normalize_channels(channels: torch.Tensor, params: ChannelParams = DEFAULT_PARAMS) -> torch.Tensor:
    flat = channels.reshape(channels.shape[0], -1)
    low = torch.quantile(flat, params.norm_percentile, dim=1)[:, None, None]
    high = torch.quantile(flat, 1.0 - params.norm_percentile, dim=1)[:, None, None]
    return ((channels - low) / (high - low).clamp_min(params.eps)).clamp(0.0, 1.0)


def build_channels(
    video: torch.Tensor,
    selection: FrameSelection,
    params: ChannelParams = DEFAULT_PARAMS,
) -> torch.Tensor:
    video = _as_video(video)
    sampled = video[selection.sampled]
    kept = video[selection.keep.nonzero().flatten()]

    maxmin = sampled.amax(dim=0) - sampled.amin(dim=0)
    std = sampled.std(dim=0)
    pca1 = temporal_pca1(kept, maxmin, params)
    tsr_d1, tsr_d2 = compute_tsr_derivatives(kept, params)

    return normalize_channels(
        torch.stack([maxmin, std, pca1, tsr_d1, tsr_d2], dim=0),
        params,
    )


def extract_channels(video: torch.Tensor, params: ChannelParams = DEFAULT_PARAMS) -> torch.Tensor:
    return build_channels(video, select_frames(video, params), params)
