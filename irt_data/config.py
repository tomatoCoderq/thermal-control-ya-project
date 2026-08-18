"""Config-driven dataclasses for the IRT segmentation dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Mode = Literal["features", "temporal"]
CropStrategy = Literal["roi_random", "roi_center", "roi_exact", "random", "center", "full"]
PadMode = Literal["reflect", "zeros", "resize"]
MaskKind = Literal["multiclass", "binary"]
MissingMask = Literal["zeros", "none", "error"]
NormMode = Literal["per_video", "per_sample", "per_channel", "none"]
CollateMode = Literal["stack", "pad"]
TimePad = Literal["repeat_last", "reflect"]
SamplerName = Literal["uniform", "window", "keypoints"]
AugApply = Literal["one_of", "chain"]


@dataclass
class ROI:
    """Axis-aligned bounding box in pixel coordinates (x, y, w, h)."""

    x: int
    y: int
    w: int
    h: int

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass
class DefectBox:
    """BBox дефекта / empty-патча (для swap/shuffle аугментации)."""

    x: int
    y: int
    w: int
    h: int
    label: int | None = None  # значение пикселя маски (0, 51, 102, ...)

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    def size_key(self, tol: int = 0) -> tuple[int, int]:
        if tol <= 0:
            return self.w, self.h
        return (self.w // (tol + 1)) * (tol + 1), (self.h // (tol + 1)) * (tol + 1)


@dataclass
class FileMeta:
    """Per-file metadata: ROI, defect boxes, temporal keypoints.

    frame_range: [start, end) — interesting frames for this video.
    defect_boxes / empty_boxes: bbox для swap_defect_boxes (одинаковый size).
    """

    rois: list[ROI] = field(default_factory=list)
    object_roi: ROI | None = None
    defect_boxes: list[DefectBox] = field(default_factory=list)
    empty_boxes: list[DefectBox] = field(default_factory=list)
    frame_range: tuple[int, int] | None = None
    heat_start: int | None = None
    cool_start: int | None = None
    peak_contrast: int | None = None
    fps: float | None = None
    notes: str = ""

    def swap_boxes(self) -> list[DefectBox]:
        """defect + empty — общий пул для shuffle одинакового размера."""
        return list(self.defect_boxes) + list(self.empty_boxes)


@dataclass
class SourceConfig:
    """One folder of .mat videos + optional mask folder."""

    root: str
    masks: str | None = None
    pattern: str = "*.mat"
    time_axis: int | None = None  # None = auto-detect
    object_roi: ROI | None = None


@dataclass
class FeatureConfig:
    extractors: list[str] = field(default_factory=lambda: ["tsr"])
    frame_step: int = 4
    max_frames: int | None = None
    poly_degree: int = 5
    pca_components: int = 3
    ppt_bins: tuple[int, ...] = (1, 2, 3)
    ppt_frames: int = 512
    ppt_auto_cooling: bool = True
    thermal_diff: bool = True
    cache_dir: str = "artifacts/features"


@dataclass
class TemporalConfig:
    sampler: SamplerName = "uniform"
    num_frames: int = 20
    stride: int = 1
    window_size: int | None = None  # defaults to num_frames
    jitter: int = 0
    frame_drop_p: float = 0.0
    time_pad: TimePad = "repeat_last"
    add_dt_channel: bool = False
    # Global default window [start, end); per-file files_meta.*.frame_range overrides.
    frame_range: tuple[int, int] | None = None


@dataclass
class CropConfig:
    size: tuple[int, int] = (256, 256)  # (H, W)
    strategy: CropStrategy = "random"
    pad_mode: PadMode = "resize"
    roi_padding: int = 0


@dataclass
class ObjectCropConfig:
    """Fixed specimen crop applied before temporal feature extraction.

    ``roi`` is a global fallback. ``files_meta.<id>.object_roi`` overrides it.
    The rectangle is cropped first and optionally reflection-padded to a square.
    Feature maps and masks are resized to ``output_size`` afterwards.
    """

    enabled: bool = False
    roi: ROI | None = None
    square_pad: bool = True
    pad_mode: PadMode = "reflect"
    output_size: tuple[int, int] = (256, 256)


@dataclass
class AugSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SwapDefectBoxesConfig:
    """Рандомно меняет местами патчи defect_boxes(+empty) одинакового (w,h)."""

    p: float = 0.5
    size_tol: int = 0  # 0 = точное совпадение размера


@dataclass
class AugConfig:
    """Аугментации.

    apply:
      one_of — ровно один тип из spatial ∪ swap (по умолчанию);
      chain  — все spatial подряд + swap отдельно (каждый со своим p).
    """

    spatial: list[AugSpec] = field(default_factory=list)
    swap_defect_boxes: SwapDefectBoxesConfig | None = None
    apply: AugApply = "one_of"
    use_replay_fallback: bool = False


@dataclass
class NormConfig:
    mode: NormMode = "per_video"
    eps: float = 1e-6


@dataclass
class MaskConfig:
    kind: MaskKind = "multiclass"
    num_classes: int = 6
    missing: MissingMask = "zeros"
    ignore_index: int = 255
    pixel_to_class: dict[int, int] = field(
        default_factory=lambda: {0: 0, 51: 1, 102: 2, 153: 3, 204: 4, 255: 5}
    )


@dataclass
class LoaderConfig:
    batch_size: int = 4
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = False
    drop_last: bool = False
    collate: CollateMode = "stack"


@dataclass
class DatasetConfig:
    """Top-level dataset configuration."""

    mode: Mode = "features"
    sources: list[SourceConfig] = field(default_factory=list)
    cache_dir: str = "artifacts/cache"
    files_meta: dict[str, FileMeta] = field(default_factory=dict)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    object_crop: ObjectCropConfig = field(default_factory=ObjectCropConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    augs: AugConfig = field(default_factory=AugConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    samples_per_video: int = 1
    seed: int = 0
    train: bool = True

    @staticmethod
    def from_dict(data: dict[str, Any]) -> DatasetConfig:
        data = dict(data)
        sources = []
        for source_raw in data.pop("sources", []):
            source_raw = dict(source_raw)
            source_roi_raw = source_raw.pop("object_roi", None)
            sources.append(
                SourceConfig(
                    object_roi=ROI(**source_roi_raw) if source_roi_raw is not None else None,
                    **source_raw,
                )
            )
        files_meta_raw = data.pop("files_meta", {}) or {}
        files_meta: dict[str, FileMeta] = {}
        _file_meta_keys = {
            "frame_range", "heat_start", "cool_start", "peak_contrast", "fps", "notes",
        }
        for vid, meta in files_meta_raw.items():
            meta = dict(meta)
            rois = [ROI(**r) for r in meta.pop("rois", []) or []]
            object_roi_raw = meta.pop("object_roi", None)
            object_roi = ROI(**object_roi_raw) if object_roi_raw is not None else None
            defect_boxes = [
                DefectBox(**b) for b in meta.pop("defect_boxes", []) or []
            ]
            empty_boxes = [
                DefectBox(**b) for b in meta.pop("empty_boxes", []) or []
            ]
            # ignore helper fields from box-extraction yaml
            meta.pop("image_size", None)
            meta.pop("defect_hull", None)
            # migrate renamed keys → frame_range
            if "frame_range" not in meta and (
                "interesting_start" in meta or "interesting_end" in meta
            ):
                meta["frame_range"] = (
                    meta.pop("interesting_start", 0),
                    meta.pop("interesting_end", None),
                )
            meta.pop("interesting_start", None)
            meta.pop("interesting_end", None)
            meta.pop("start", None)
            meta.pop("end", None)
            fr = meta.pop("frame_range", None)
            if fr is not None:
                meta["frame_range"] = tuple(fr)
            if object_roi is None and rois:
                object_roi = rois[0]
            meta = {k: v for k, v in meta.items() if k in _file_meta_keys}
            files_meta[vid] = FileMeta(
                rois=rois,
                object_roi=object_roi,
                defect_boxes=defect_boxes,
                empty_boxes=empty_boxes,
                **meta,
            )

        features_raw = data.pop("features", {}) or {}
        if "ppt_bins" in features_raw:
            features_raw["ppt_bins"] = tuple(features_raw["ppt_bins"])
        features = FeatureConfig(**features_raw)
        temporal_raw = data.pop("temporal", {}) or {}
        temporal_raw = {k: v for k, v in temporal_raw.items() if not str(k).startswith("_")}
        # migrate old interesting_* → frame_range
        if "frame_range" not in temporal_raw and (
            "interesting_start" in temporal_raw or "interesting_end" in temporal_raw
        ):
            temporal_raw["frame_range"] = (
                temporal_raw.pop("interesting_start", 0),
                temporal_raw.pop("interesting_end", None),
            )
        temporal_raw.pop("interesting_start", None)
        temporal_raw.pop("interesting_end", None)
        if "frame_range" in temporal_raw and temporal_raw["frame_range"] is not None:
            fr = temporal_raw["frame_range"]
            temporal_raw["frame_range"] = tuple(fr) if not isinstance(fr, tuple) else fr
        temporal = TemporalConfig(**temporal_raw)

        object_crop_raw = data.pop("object_crop", {}) or {}
        object_roi_raw = object_crop_raw.pop("roi", None)
        if "output_size" in object_crop_raw:
            object_crop_raw["output_size"] = tuple(object_crop_raw["output_size"])
        object_crop = ObjectCropConfig(
            roi=ROI(**object_roi_raw) if object_roi_raw is not None else None,
            **object_crop_raw,
        )

        crop_raw = data.pop("crop", {}) or {}
        if "size" in crop_raw:
            crop_raw["size"] = tuple(crop_raw["size"])
        crop = CropConfig(**crop_raw)

        augs_raw = data.pop("augs", {}) or {}
        spatial = [AugSpec(**a) for a in augs_raw.pop("spatial", [])]
        swap_raw = augs_raw.pop("swap_defect_boxes", None)
        if isinstance(swap_raw, dict):
            swap_cfg: SwapDefectBoxesConfig | None = SwapDefectBoxesConfig(**swap_raw)
        elif swap_raw is True:
            swap_cfg = SwapDefectBoxesConfig()
        else:
            swap_cfg = None
        apply_mode = augs_raw.pop("apply", "one_of") or "one_of"
        if apply_mode not in ("one_of", "chain"):
            raise ValueError(f"augs.apply must be 'one_of' or 'chain', got {apply_mode!r}")
        augs = AugConfig(
            spatial=spatial,
            swap_defect_boxes=swap_cfg,
            apply=apply_mode,  # type: ignore[arg-type]
            **augs_raw,
        )

        norm = NormConfig(**data.pop("norm", {}) or {})
        mask_raw = data.pop("mask", {}) or {}
        if "pixel_to_class" in mask_raw and mask_raw["pixel_to_class"] is not None:
            mask_raw["pixel_to_class"] = {
                int(k): int(v) for k, v in dict(mask_raw["pixel_to_class"]).items()
            }
        mask = MaskConfig(**mask_raw)
        loader = LoaderConfig(**data.pop("loader", {}) or {})

        return DatasetConfig(
            sources=sources,
            files_meta=files_meta,
            features=features,
            temporal=temporal,
            object_crop=object_crop,
            crop=crop,
            augs=augs,
            norm=norm,
            mask=mask,
            loader=loader,
            **data,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)
