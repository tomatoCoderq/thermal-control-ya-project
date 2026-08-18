"""Реестр экстракторов признаков: единый интерфейс extract(mat_path) -> (C,H,W)."""
from __future__ import annotations

from ..config import FeaturesConfig
from . import tsr as _tsr
from . import fourier as _fourier


def build_extractor(cfg: FeaturesConfig):
    """Вернуть callable(mat_path) -> np.ndarray (C,H,W) по cfg.features.kind."""
    if cfg.kind == "tsr":
        def _extract(path):
            return _tsr.tsr_coeffs(path, deg=cfg.poly_degree)
        return _extract

    if cfg.kind == "fourier":
        def _extract(path):
            video, _fs = _fourier.load_thermal_video(path)     # (H,W,frames)
            feats, _bins, _start = _fourier.fourier_phase_features(
                video, first_bin=cfg.first_bin, n_frequencies=cfg.n_frequencies,
                phase_encoding=cfg.phase_encoding, window=cfg.window,
                detrend=cfg.detrend, start_at_peak=cfg.start_at_peak)
            return feats                                       # (C,H,W)
        return _extract

    raise ValueError(f"неизвестный features.kind: {cfg.kind}")
