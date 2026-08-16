"""ROI-aware cropping with consistent coordinates across a temporal clip."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from irt_data.config import CropConfig, FileMeta, ObjectCropConfig, PadMode, ROI


@dataclass(frozen=True)
class CropBox:
    """Crop window that may extend outside the image (padding required)."""

    y0: int
    x0: int
    y1: int
    x1: int  # exclusive
    out_h: int
    out_w: int

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.y0, self.x0, self.y1, self.x1


def _pad_mode_numpy(mode: PadMode) -> str:
    return "reflect" if mode == "reflect" else "constant"


def apply_crop_2d(
    image: np.ndarray,
    box: CropBox,
    pad_mode: PadMode = "reflect",
) -> np.ndarray:
    """Crop HxW or HxWxC with padding if box is out of bounds."""
    H, W = image.shape[:2]
    y0, x0, y1, x1 = box.y0, box.x0, box.y1, box.x1

    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, y1 - H)
    pad_right = max(0, x1 - W)

    if any(v > 0 for v in (pad_top, pad_left, pad_bottom, pad_right)):
        if image.ndim == 2:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
        else:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        kwargs = {}
        if pad_mode == "zeros":
            kwargs["constant_values"] = 0
        image = np.pad(image, pad_width, mode=_pad_mode_numpy(pad_mode), **kwargs)
        y0 += pad_top
        y1 += pad_top
        x0 += pad_left
        x1 += pad_left

    cropped = image[y0:y1, x0:x1]
    # exact size (safety)
    if cropped.shape[0] != box.out_h or cropped.shape[1] != box.out_w:
        # hard truncate / pad to exact
        out = np.zeros((box.out_h, box.out_w) + cropped.shape[2:], dtype=cropped.dtype)
        hh = min(box.out_h, cropped.shape[0])
        ww = min(box.out_w, cropped.shape[1])
        out[:hh, :ww] = cropped[:hh, :ww]
        return out
    return cropped


def apply_crop_thw(
    frames: np.ndarray,
    box: CropBox,
    pad_mode: PadMode = "reflect",
) -> np.ndarray:
    """Apply the same crop to every frame in (T, H, W)."""
    out = [apply_crop_2d(frames[t], box, pad_mode) for t in range(frames.shape[0])]
    return np.stack(out, axis=0)


def resize_2d(
    image: np.ndarray, size: tuple[int, int], is_mask: bool = False
) -> np.ndarray:
    """Resize HxW or HxWxC; masks always use nearest-neighbour interpolation."""
    import cv2

    out_h, out_w = size
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (out_w, out_h), interpolation=interpolation)
    if image.ndim == 3 and resized.ndim == 2:
        resized = resized[..., None]
    return resized.astype(image.dtype, copy=False)


class ObjectCropper:
    """Remove the fixture/frame while preserving one coordinate system per video."""

    def __init__(self, cfg: ObjectCropConfig) -> None:
        self.cfg = cfg

    def roi_for(self, meta: FileMeta | None, source_roi: ROI | None = None) -> ROI | None:
        if meta is not None and meta.object_roi is not None:
            return meta.object_roi
        if source_roi is not None:
            return source_roi
        return self.cfg.roi

    def box_for(
        self, H: int, W: int, meta: FileMeta | None, source_roi: ROI | None = None
    ) -> CropBox:
        roi = self.roi_for(meta, source_roi)
        if not self.cfg.enabled or roi is None:
            return CropBox(0, 0, H, W, H, W)

        x0 = max(0, int(roi.x))
        y0 = max(0, int(roi.y))
        x1 = min(W, int(roi.x + roi.w))
        y1 = min(H, int(roi.y + roi.h))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"object_roi {roi} is outside image shape {(H, W)}")

        h, w = y1 - y0, x1 - x0
        return CropBox(y0, x0, y1, x1, h, w)

    def _pad_square(self, image: np.ndarray, is_mask: bool) -> np.ndarray:
        if not self.cfg.square_pad or image.shape[0] == image.shape[1]:
            return image
        h, w = image.shape[:2]
        side = max(h, w)
        top = (side - h) // 2
        bottom = side - h - top
        left = (side - w) // 2
        right = side - w - left
        pad_width = ((top, bottom), (left, right))
        if image.ndim == 3:
            pad_width += ((0, 0),)
        if is_mask:
            return np.pad(image, pad_width, mode="constant", constant_values=0)
        return np.pad(image, pad_width, mode=_pad_mode_numpy(self.cfg.pad_mode))

    def apply_frames(self, frames: np.ndarray, box: CropBox) -> np.ndarray:
        cropped = apply_crop_thw(frames, box, self.cfg.pad_mode)
        return np.stack([self._pad_square(frame, is_mask=False) for frame in cropped])

    def apply_image(self, image: np.ndarray, box: CropBox) -> np.ndarray:
        cropped = apply_crop_2d(image, box, self.cfg.pad_mode)
        cropped = self._pad_square(cropped, is_mask=False)
        if not self.cfg.enabled:
            return cropped
        return resize_2d(cropped, self.cfg.output_size, is_mask=False)

    def apply_mask(self, mask: np.ndarray, box: CropBox) -> np.ndarray:
        cropped = apply_crop_2d(mask, box, pad_mode="zeros")
        cropped = self._pad_square(cropped, is_mask=True)
        if not self.cfg.enabled:
            return cropped
        return resize_2d(cropped, self.cfg.output_size, is_mask=True)


class RoiCropper:
    """Plan a CropBox once per __getitem__, then apply to frames + mask."""

    def __init__(self, cfg: CropConfig) -> None:
        self.cfg = cfg
        self.out_h, self.out_w = int(cfg.size[0]), int(cfg.size[1])

    def plan(
        self,
        H: int,
        W: int,
        rng: np.random.Generator,
        meta: FileMeta | None = None,
    ) -> CropBox:
        strategy = self.cfg.strategy
        rois = meta.rois if meta is not None else []

        if strategy in ("roi_random", "roi_center") and rois:
            roi = rois[int(rng.integers(0, len(rois)))] if strategy == "roi_random" else rois[0]
            return self._plan_in_roi(H, W, roi, rng, center=(strategy == "roi_center"))

        if strategy == "center" or (strategy.startswith("roi_") and not rois):
            return self._center_box(H, W)
        if strategy == "full":
            # crop box covering full image; output size may pad
            return CropBox(0, 0, H, W, self.out_h, self.out_w)
        # random
        return self._random_box(H, W, rng)

    def _center_box(self, H: int, W: int) -> CropBox:
        y0 = (H - self.out_h) // 2
        x0 = (W - self.out_w) // 2
        return CropBox(y0, x0, y0 + self.out_h, x0 + self.out_w, self.out_h, self.out_w)

    def _random_box(self, H: int, W: int, rng: np.random.Generator) -> CropBox:
        # if image smaller than crop, allow negative origin (padding fills the rest)
        y0 = int(rng.integers(0, H - self.out_h + 1)) if H >= self.out_h else int(
            rng.integers(H - self.out_h, 1)
        )
        x0 = int(rng.integers(0, W - self.out_w + 1)) if W >= self.out_w else int(
            rng.integers(W - self.out_w, 1)
        )
        return CropBox(y0, x0, y0 + self.out_h, x0 + self.out_w, self.out_h, self.out_w)

    def _plan_in_roi(
        self,
        H: int,
        W: int,
        roi: ROI,
        rng: np.random.Generator,
        center: bool,
    ) -> CropBox:
        pad = self.cfg.roi_padding
        rx0 = max(0, roi.x - pad)
        ry0 = max(0, roi.y - pad)
        rx1 = min(W, roi.x + roi.w + pad)
        ry1 = min(H, roi.y + roi.h + pad)

        if center:
            cy = (ry0 + ry1) // 2
            cx = (rx0 + rx1) // 2
            y0 = cy - self.out_h // 2
            x0 = cx - self.out_w // 2
        else:
            # random crop whose center lies inside ROI (or window intersects ROI)
            # sample top-left so crop overlaps ROI
            min_y = ry0 - self.out_h + 1
            max_y = ry1
            min_x = rx0 - self.out_w + 1
            max_x = rx1
            if min_y > max_y:
                min_y, max_y = ry0, ry0
            if min_x > max_x:
                min_x, max_x = rx0, rx0
            y0 = int(rng.integers(min_y, max_y + 1))
            x0 = int(rng.integers(min_x, max_x + 1))

        return CropBox(y0, x0, y0 + self.out_h, x0 + self.out_w, self.out_h, self.out_w)

    def apply_image(
        self,
        image: np.ndarray,
        box: CropBox,
    ) -> np.ndarray:
        return apply_crop_2d(image, box, self.cfg.pad_mode)

    def apply_frames(
        self,
        frames: np.ndarray,
        box: CropBox,
    ) -> np.ndarray:
        return apply_crop_thw(frames, box, self.cfg.pad_mode)

    def apply_mask(
        self,
        mask: np.ndarray,
        box: CropBox,
    ) -> np.ndarray:
        # masks always zero-pad outside (labels shouldn't reflect)
        return apply_crop_2d(mask, box, pad_mode="zeros")
