"""Albumentations-based transform pipeline for features and temporal modes."""

from __future__ import annotations

import logging
from typing import Any

import albumentations as A
import numpy as np

from irt_data.config import AugConfig, AugSpec

logger = logging.getLogger(__name__)


def _coerce_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert YAML lists to tuples where albumentations expects ranges."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            out[k] = tuple(v)
        else:
            out[k] = v
    return out


def build_albumentations(specs: list[AugSpec]) -> A.Compose | None:
    if not specs:
        return None
    transforms = []
    for spec in specs:
        if spec.name == "DiscreteRotate7":
            params = {"border_mode": 4, "fill": 0, "fill_mask": 0}
            params.update(_coerce_params(spec.params))
            p = float(params.pop("p", 1.0))
            transforms.append(
                A.OneOf(
                    [
                        A.Rotate(limit=(-7, -7), p=0.25, **params),
                        A.NoOp(p=0.50),
                        A.Rotate(limit=(7, 7), p=0.25, **params),
                    ],
                    p=p,
                )
            )
            continue
        if not hasattr(A, spec.name):
            raise AttributeError(
                f"albumentations has no transform '{spec.name}'. "
                f"Check AugSpec.name in config."
            )
        cls = getattr(A, spec.name)
        transforms.append(cls(**_coerce_params(spec.params)))
    return A.Compose(transforms)


class TransformPipeline:
    """Spatial augs applied identically to all frames in a clip (+ mask)."""

    def __init__(self, cfg: AugConfig) -> None:
        self.cfg = cfg
        self.spatial = build_albumentations(cfg.spatial)

    @classmethod
    def from_config(cls, cfg: AugConfig) -> TransformPipeline:
        return cls(cfg)

    def apply_features(
        self,
        image_hwc: np.ndarray,
        mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """image (H,W,C) float, mask (H,W) uint8/int."""
        if self.spatial is None:
            return image_hwc, mask
        kwargs: dict[str, Any] = {"image": image_hwc}
        if mask is not None:
            kwargs["mask"] = mask
        out = self.spatial(**kwargs)
        return out["image"], (out["mask"] if mask is not None else None)

    def apply_temporal(
        self,
        frames_thw: np.ndarray,
        mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """frames (T,H,W) — same geometry for every frame and the mask."""
        if self.spatial is None:
            return frames_thw, mask

        frames_list = [frames_thw[t] for t in range(frames_thw.shape[0])]

        if self.cfg.use_replay_fallback:
            return self._replay_temporal(frames_list, mask)

        kwargs: dict[str, Any] = {"images": frames_list}
        if mask is not None:
            kwargs["mask"] = mask
        out = self.spatial(**kwargs)
        images = np.asarray(out["images"], dtype=frames_thw.dtype)
        return images, (out["mask"] if mask is not None else None)

    def _replay_temporal(
        self,
        frames_list: list[np.ndarray],
        mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Fallback: ReplayCompose — apply recorded params to every frame."""
        specs = self.cfg.spatial
        transforms = []
        for spec in specs:
            cls = getattr(A, spec.name)
            transforms.append(cls(**_coerce_params(spec.params)))
        replay = A.ReplayCompose(transforms)

        first_kwargs: dict[str, Any] = {"image": frames_list[0]}
        if mask is not None:
            first_kwargs["mask"] = mask
        first = replay(**first_kwargs)
        replay_data = first["replay"]
        out_frames = [first["image"]]
        out_mask = first.get("mask", mask)
        for frame in frames_list[1:]:
            r = A.ReplayCompose.replay(replay_data, image=frame)
            out_frames.append(r["image"])
        return np.asarray(out_frames), out_mask
