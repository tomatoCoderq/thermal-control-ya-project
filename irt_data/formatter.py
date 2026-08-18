"""Convert numpy arrays to torch tensors with normalization and mask encoding."""

from __future__ import annotations

import numpy as np
import torch

from irt_data.config import MaskConfig, NormConfig


class TensorFormatter:
    """Normalize and pack arrays into model-ready tensors."""

    def __init__(self, norm: NormConfig, mask_cfg: MaskConfig) -> None:
        self.norm = norm
        self.mask_cfg = mask_cfg

    def normalize(
        self,
        x: np.ndarray,
        stats: dict[str, float] | None = None,
        channel_axis: int | None = None,
    ) -> np.ndarray:
        """Normalize array of any shape using configured mode.

        ``channel_axis`` must be given by the caller for ``per_channel``; guessing it
        from the shape silently mislabels a short temporal clip as channels.
        """
        x = x.astype(np.float32)
        mode = self.norm.mode
        eps = self.norm.eps
        if mode == "none":
            return x
        if mode == "per_channel" and channel_axis is not None:
            axes = tuple(i for i in range(x.ndim) if i != channel_axis % x.ndim)
            mean = x.mean(axis=axes, keepdims=True)
            std = x.std(axis=axes, keepdims=True)
            return (x - mean) / np.maximum(std, eps)
        if mode == "per_video" and stats and "mean" in stats and "std" in stats:
            mean = float(stats["mean"])
            std = float(stats["std"])
            if std < eps:
                std = 1.0
            return (x - mean) / (std + eps)
        if mode == "per_video" and stats and "min" in stats and "max" in stats:
            mn, mx = float(stats["min"]), float(stats["max"])
            return (x - mn) / (mx - mn + eps)
        # per_sample min-max over all elements
        mn = float(x.min())
        mx = float(x.max())
        return (x - mn) / (mx - mn + eps)

    def encode_mask(self, mask: np.ndarray | None, has_mask: bool) -> torch.Tensor:
        cfg = self.mask_cfg
        if mask is None or not has_mask:
            if cfg.missing == "error":
                raise FileNotFoundError("Mask is required but missing")
            if cfg.missing == "none":
                # return empty-like sentinel; dataset should handle
                return torch.full((1, 1), cfg.ignore_index, dtype=torch.long)
            # zeros
            raise RuntimeError("Caller must pass a zeros mask for missing='zeros'")

        mask = np.asarray(mask)
        if cfg.kind == "binary":
            out = (mask > 0).astype(np.int64)
        else:
            out = np.zeros(mask.shape, dtype=np.int64)
            mapping = cfg.pixel_to_class
            # fast path: multiples of 51
            if set(mapping.keys()) <= {0, 51, 102, 153, 204, 255}:
                out = (mask.astype(np.int64) // 51).clip(0, cfg.num_classes - 1)
                # remap if custom
                for pv, cls in mapping.items():
                    out[mask == pv] = cls
            else:
                for pv, cls in mapping.items():
                    out[mask == pv] = cls
        return torch.from_numpy(out.astype(np.int64))

    def format_features(
        self,
        image_hwc: np.ndarray,
        mask: np.ndarray,
        has_mask: bool,
        stats: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(H,W,C) -> (C,H,W) float32; mask -> (H,W) long."""
        img = self.normalize(image_hwc, stats, channel_axis=-1)
        tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        mask_t = self.encode_mask(mask, has_mask)
        return tensor.float(), mask_t

    def format_temporal(
        self,
        frames_thw: np.ndarray,
        mask: np.ndarray,
        has_mask: bool,
        stats: dict[str, float] | None = None,
        add_dt_channel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(T,H,W) -> (T,C,H,W); mask -> (H,W) long."""
        frames = self.normalize(frames_thw, stats)
        if add_dt_channel:
            dt = np.gradient(frames.astype(np.float32), axis=0)
            stacked = np.stack([frames, dt], axis=1)  # (T, 2, H, W)
        else:
            stacked = frames[:, None, :, :]  # (T, 1, H, W)
        tensor = torch.from_numpy(np.ascontiguousarray(stacked)).float()
        mask_t = self.encode_mask(mask, has_mask)
        return tensor, mask_t
