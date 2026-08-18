"""Albumentations spatial augs + swap of same-size defect/empty boxes."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import albumentations as A
import numpy as np

from irt_data.config import AugConfig, AugSpec, DefectBox, SwapDefectBoxesConfig

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


def _build_one_transform(spec: AugSpec) -> Any:
    """Build one albumentations op; supports custom DiscreteRotate7."""
    if spec.name == "DiscreteRotate7":
        # −7° / 0° / +7° with probs 0.25 / 0.50 / 0.25
        params = {"border_mode": 4, "fill": 0, "fill_mask": 0}
        params.update(_coerce_params(spec.params))
        p = float(params.pop("p", 1.0))
        return A.OneOf(
            [
                A.Rotate(limit=(-7, -7), p=0.25, **params),
                A.NoOp(p=0.50),
                A.Rotate(limit=(7, 7), p=0.25, **params),
            ],
            p=p,
        )
    if not hasattr(A, spec.name):
        raise AttributeError(
            f"albumentations has no transform '{spec.name}'. "
            f"Check AugSpec.name in config."
        )
    cls = getattr(A, spec.name)
    return cls(**_coerce_params(spec.params))


def build_albumentations(
    specs: list[AugSpec],
    *,
    apply: str = "one_of",
) -> A.Compose | None:
    """Build spatial pipeline.

    apply=one_of — ровно один transform из списка (веса = params.p).
    apply=chain  — все подряд (каждый со своим params.p).
    """
    if not specs:
        return None
    transforms = [_build_one_transform(spec) for spec in specs]
    if apply == "chain":
        return A.Compose(transforms)
    return A.Compose([A.OneOf(transforms, p=1.0)])


def choose_aug_op(
    cfg: AugConfig,
    rng: np.random.Generator,
) -> tuple[str, int | None]:
    """Выбрать ровно один тип аугментации для sample.

    Returns
    -------
    ("none", None) | ("swap", None) | ("spatial", index_in_cfg.spatial)
    """
    candidates: list[tuple[str, int | None, float]] = []
    for i, spec in enumerate(cfg.spatial):
        w = float(spec.params.get("p", 1.0))
        if w > 0:
            candidates.append(("spatial", i, w))
    if cfg.swap_defect_boxes is not None and cfg.swap_defect_boxes.p > 0:
        candidates.append(("swap", None, float(cfg.swap_defect_boxes.p)))
    if not candidates:
        return "none", None
    weights = np.asarray([c[2] for c in candidates], dtype=np.float64)
    weights = weights / weights.sum()
    pick = int(rng.choice(len(candidates), p=weights))
    kind, idx, _ = candidates[pick]
    return kind, idx


def build_single_spatial(spec: AugSpec) -> A.Compose:
    """Один spatial-transform с p=1 (уже выбран снаружи)."""
    forced = AugSpec(name=spec.name, params={**spec.params, "p": 1.0})
    return A.Compose([_build_one_transform(forced)])


def remap_boxes_to_object_crop(
    boxes: list[DefectBox],
    object_box: Any,
    *,
    enabled: bool,
    square_pad: bool,
    output_size: tuple[int, int],
) -> list[DefectBox]:
    """Переводит defect/empty boxes из сырых координат кадра в пространство после object_crop.

    Совпадает с ``ObjectCropper``: crop → square_pad → resize to ``output_size``.
    ``object_box`` — ``CropBox`` из ``ObjectCropper.box_for``.
    """
    if not boxes:
        return []
    if not enabled:
        return list(boxes)

    crop_h, crop_w = object_box.out_h, object_box.out_w
    pad_top = pad_left = 0
    if square_pad and crop_h != crop_w:
        side = max(crop_h, crop_w)
        pad_top = (side - crop_h) // 2
        pad_left = (side - crop_w) // 2
        sq_h = sq_w = side
    else:
        sq_h, sq_w = crop_h, crop_w

    out_h, out_w = int(output_size[0]), int(output_size[1])
    sy = out_h / float(sq_h)
    sx = out_w / float(sq_w)

    remapped: list[DefectBox] = []
    for b in boxes:
        x0 = b.x - object_box.x0
        y0 = b.y - object_box.y0
        x1 = x0 + b.w
        y1 = y0 + b.h
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(crop_w, x1), min(crop_h, y1)
        w, h = x1c - x0c, y1c - y0c
        if w <= 0 or h <= 0:
            continue
        x0c += pad_left
        y0c += pad_top
        remapped.append(
            DefectBox(
                x=int(round(x0c * sx)),
                y=int(round(y0c * sy)),
                w=max(1, int(round(w * sx))),
                h=max(1, int(round(h * sy))),
                label=b.label,
            )
        )
    return remapped


def _clip_box(box: DefectBox, H: int, W: int) -> DefectBox | None:
    x0 = max(0, box.x)
    y0 = max(0, box.y)
    x1 = min(W, box.x + box.w)
    y1 = min(H, box.y + box.h)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    return DefectBox(x=x0, y=y0, w=w, h=h, label=box.label)


def swap_same_size_defect_boxes(
    image: np.ndarray,
    mask: np.ndarray | None,
    boxes: list[DefectBox],
    rng: np.random.Generator,
    cfg: SwapDefectBoxesConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Permute patches among boxes that share the same (w, h).

    Works on ``(H, W)`` or ``(H, W, C)``. Mask (if given) uses the same permutation.
    """
    if not boxes or cfg.p <= 0 or rng.random() > cfg.p:
        return image, mask

    H, W = image.shape[:2]
    valid = []
    for b in boxes:
        c = _clip_box(b, H, W)
        if c is not None:
            valid.append(c)
    if len(valid) < 2:
        return image, mask

    groups: dict[tuple[int, int], list[DefectBox]] = defaultdict(list)
    for b in valid:
        groups[b.size_key(cfg.size_tol)].append(b)

    image = image.copy()
    mask_out = None if mask is None else mask.copy()

    for _, group in groups.items():
        if len(group) < 2:
            continue
        wh = {(b.w, b.h) for b in group}
        if len(wh) != 1:
            continue
        order = np.arange(len(group))
        rng.shuffle(order)
        src_img = [image[b.y : b.y + b.h, b.x : b.x + b.w].copy() for b in group]
        src_msk = None
        if mask_out is not None:
            src_msk = [mask_out[b.y : b.y + b.h, b.x : b.x + b.w].copy() for b in group]
        for dst_i, src_i in enumerate(order):
            b = group[dst_i]
            image[b.y : b.y + b.h, b.x : b.x + b.w] = src_img[src_i]
            if src_msk is not None:
                mask_out[b.y : b.y + b.h, b.x : b.x + b.w] = src_msk[src_i]

    return image, mask_out


def swap_same_size_defect_boxes_temporal(
    frames: np.ndarray,
    mask: np.ndarray | None,
    boxes: list[DefectBox],
    rng: np.random.Generator,
    cfg: SwapDefectBoxesConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Same swap on every frame of ``(T, H, W)`` with one shared permutation."""
    if not boxes or cfg.p <= 0 or rng.random() > cfg.p:
        return frames, mask

    H, W = frames.shape[1], frames.shape[2]
    valid = []
    for b in boxes:
        c = _clip_box(b, H, W)
        if c is not None:
            valid.append(c)
    if len(valid) < 2:
        return frames, mask

    groups: dict[tuple[int, int], list[DefectBox]] = defaultdict(list)
    for b in valid:
        groups[b.size_key(cfg.size_tol)].append(b)

    frames = frames.copy()
    mask_out = None if mask is None else mask.copy()

    for _, group in groups.items():
        if len(group) < 2:
            continue
        if len({(b.w, b.h) for b in group}) != 1:
            continue
        order = np.arange(len(group))
        rng.shuffle(order)

        for t in range(frames.shape[0]):
            src = [frames[t, b.y : b.y + b.h, b.x : b.x + b.w].copy() for b in group]
            for dst_i, src_i in enumerate(order):
                b = group[dst_i]
                frames[t, b.y : b.y + b.h, b.x : b.x + b.w] = src[src_i]

        if mask_out is not None:
            src_m = [mask_out[b.y : b.y + b.h, b.x : b.x + b.w].copy() for b in group]
            for dst_i, src_i in enumerate(order):
                b = group[dst_i]
                mask_out[b.y : b.y + b.h, b.x : b.x + b.w] = src_m[src_i]

    return frames, mask_out


class TransformPipeline:
    """Spatial augs applied identically to all frames in a clip (+ mask)."""

    def __init__(self, cfg: AugConfig, spatial_compose: A.Compose | None = None) -> None:
        self.cfg = cfg
        self.spatial = (
            spatial_compose
            if spatial_compose is not None
            else build_albumentations(cfg.spatial, apply=cfg.apply)
        )

    @classmethod
    def from_config(cls, cfg: AugConfig) -> TransformPipeline:
        return cls(cfg)

    @classmethod
    def single_spatial(cls, cfg: AugConfig, spec_index: int) -> TransformPipeline:
        """Пайплайн ровно из одного spatial-op (для apply=one_of)."""
        spec = cfg.spatial[spec_index]
        return cls(cfg, spatial_compose=build_single_spatial(spec))

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
        if self.spatial is None:
            return np.asarray(frames_list), mask
        transforms = [_build_one_transform(spec) for spec in self.cfg.spatial]
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
