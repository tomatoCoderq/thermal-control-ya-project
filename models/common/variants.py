"""Feature variants shared by U-Net / Attention / Mamba train scripts."""
from __future__ import annotations

from pathlib import Path

from irt_data.config import DatasetConfig, FeatureConfig

VARIANTS = {
    "tsr": {
        "extractors": ["tsr_coeffs"],
        "poly_degree": 5,
        "pca_components": 6,
        "ppt_bins": (1, 2, 3),
        "cache_subdir": "tsr_coeffs",
    },
    "fourier": {
        "extractors": ["ppt"],
        "poly_degree": 5,
        "pca_components": 6,
        "ppt_bins": (1, 2, 3, 4, 5, 6),
        "cache_subdir": "ppt",
    },
    "pca": {
        "extractors": ["pca"],
        "poly_degree": 5,
        "pca_components": 6,
        "ppt_bins": (1, 2, 3, 4, 5, 6),
        "cache_subdir": "pca",
    },
}


def apply_variant_features(cfg: DatasetConfig, name: str) -> DatasetConfig:
    v = VARIANTS[name]
    # nest under features.cache_dir from yaml (e.g. artifacts/features) → …/pca
    base = Path(cfg.features.cache_dir)
    # if yaml already points at …/tsr_coeffs, use parent so variants don't nest wrongly
    if base.name in {x["cache_subdir"] for x in VARIANTS.values()}:
        base = base.parent
    cache = base / v["cache_subdir"]
    cache.mkdir(parents=True, exist_ok=True)
    cfg.features = FeatureConfig(
        extractors=list(v["extractors"]),
        frame_step=cfg.features.frame_step,
        max_frames=cfg.features.max_frames,
        poly_degree=int(v["poly_degree"]),
        pca_components=int(v.get("pca_components", cfg.features.pca_components)),
        ppt_bins=tuple(v["ppt_bins"]),
        ppt_frames=cfg.features.ppt_frames,
        ppt_auto_cooling=cfg.features.ppt_auto_cooling,
        thermal_diff=cfg.features.thermal_diff,
        cache_dir=str(cache),
    )
    return cfg
