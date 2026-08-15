"""IRTDataset: orchestrates IO, features/sampling, crop, augs, formatting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from irt_data.cache import video_id_from_path
from irt_data.config import DatasetConfig, FileMeta
from irt_data.crops import RoiCropper
from irt_data.features import CachedFeatureExtractor, build_feature_extractor
from irt_data.formatter import TensorFormatter
from irt_data.io_backend import MaskReader, MatIOBackend, NpyMemmapBackend, discover_mat_files
from irt_data.samplers import SamplingPipeline, build_frame_sampler
from irt_data.transforms import TransformPipeline

logger = logging.getLogger(__name__)


class IRTDataset(Dataset):
    """Semantic segmentation dataset for IRT NDT videos.

    Modes
    -----
    features : returns image [C,H,W], mask [H,W]
    temporal : returns image [T,C,H,W], mask [H,W]
    """

    def __init__(
        self,
        cfg: DatasetConfig,
        backend: NpyMemmapBackend | MatIOBackend | None = None,
    ) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # resolve video ids from cache or mat sources
        self.video_ids: list[str] = []
        self.mask_reader: MaskReader
        self.backend: NpyMemmapBackend | MatIOBackend
        self._mat_stems: dict[str, str] = {}

        cache_dir = Path(cfg.cache_dir)
        index_path = cache_dir / "index.json"

        mat_map: dict[str, Path] = {}
        mask_dirs: dict[str, Path | None] = {}

        for src in cfg.sources:
            root = Path(src.root)
            found = discover_mat_files([root], pattern=src.pattern)
            for vid, path in found.items():
                mat_map[vid] = path
                self._mat_stems[vid] = path.stem
                mask_dirs[vid] = Path(src.masks) if src.masks else None

        if backend is not None:
            self.backend = backend
            if hasattr(backend, "list_ids"):
                self.video_ids = list(backend.list_ids())
            else:
                self.video_ids = sorted(mat_map.keys())
        elif index_path.exists():
            self.backend = NpyMemmapBackend(cache_dir)
            cached_ids = set(self.backend.list_ids())
            if mat_map:
                self.video_ids = sorted(vid for vid in mat_map if vid in cached_ids)
                missing = sorted(set(mat_map) - cached_ids)
                if missing:
                    logger.warning(
                        "Not in cache (run python -m irt_data.cache): %s",
                        missing[:5],
                    )
            else:
                self.video_ids = sorted(cached_ids)
        else:
            if not mat_map:
                raise FileNotFoundError(
                    "No videos found. Provide sources and/or build cache "
                    f"at {cache_dir}."
                )
            logger.warning(
                "Cache index missing at %s — using MatIOBackend (slow for temporal).",
                index_path,
            )
            self.backend = MatIOBackend(mat_map)
            self.video_ids = sorted(mat_map.keys())

        if not self.video_ids:
            raise RuntimeError("IRTDataset: empty video list")

        # mask reader: map each video_id to its mask dir
        # fill mask_dirs for cache-only videos
        for vid in self.video_ids:
            mask_dirs.setdefault(vid, None)
            if mask_dirs[vid] is None:
                # try first source that has masks
                for src in cfg.sources:
                    if src.masks:
                        mask_dirs[vid] = Path(src.masks)
                        break

        self.mask_reader = MaskReader(mask_dirs)
        for vid, stem in self._mat_stems.items():
            self.mask_reader.register_alias(vid, stem)
            # also alias without spaces
            self.mask_reader.register_alias(vid, stem.replace(" ", "_"))

        self.files_meta: dict[str, FileMeta] = dict(cfg.files_meta)
        self.cropper = RoiCropper(cfg.crop)
        self.transforms = TransformPipeline.from_config(cfg.augs)
        self.formatter = TensorFormatter(cfg.norm, cfg.mask)

        self.feature_extractor = None
        self.cached_features: CachedFeatureExtractor | None = None
        self.frame_sampler: SamplingPipeline | None = None

        if cfg.mode == "features":
            base = build_feature_extractor(cfg.features)
            self.cached_features = CachedFeatureExtractor(
                base, cfg.features, cache_dir=cfg.features.cache_dir
            )
            self.feature_extractor = self.cached_features
        else:
            self.frame_sampler = build_frame_sampler(cfg.temporal, train=cfg.train)

        # index table: video repeated samples_per_video times
        self._index: list[str] = []
        for vid in self.video_ids:
            self._index.extend([vid] * max(1, cfg.samples_per_video))

        self._bad: set[str] = set()

    def __len__(self) -> int:
        return len(self._index)

    def _meta(self, video_id: str) -> FileMeta | None:
        if video_id in self.files_meta:
            return self.files_meta[video_id]
        # try stem aliases
        stem = self._mat_stems.get(video_id, video_id)
        return self.files_meta.get(stem) or self.files_meta.get(stem.replace(" ", "_"))

    def _load_mask(self, video_id: str, H: int, W: int) -> tuple[np.ndarray, bool]:
        mask = self.mask_reader.read(video_id)
        if mask is None:
            policy = self.cfg.mask.missing
            if policy == "error":
                raise FileNotFoundError(f"Mask not found for {video_id}")
            if policy == "none":
                return np.full((H, W), self.cfg.mask.ignore_index, dtype=np.uint8), False
            return np.zeros((H, W), dtype=np.uint8), False
        if mask.shape[0] != H or mask.shape[1] != W:
            logger.warning(
                "Mask shape %s != video HW (%d,%d) for %s — resizing nearest",
                mask.shape,
                H,
                W,
                video_id,
            )
            from PIL import Image

            mask = np.array(
                Image.fromarray(mask).resize((W, H), resample=Image.NEAREST)
            )
        return mask, True

    def __getitem__(self, index: int) -> dict[str, Any]:
        # retry a few times on corrupt entries
        for _ in range(5):
            video_id = self._index[index % len(self._index)]
            if video_id in self._bad:
                index = int(self.rng.integers(0, len(self._index)))
                continue
            try:
                return self._getitem_one(video_id, index)
            except Exception as exc:
                logger.exception("Skipping broken sample %s: %s", video_id, exc)
                self._bad.add(video_id)
                index = int(self.rng.integers(0, len(self._index)))
        raise RuntimeError("Too many broken samples in IRTDataset")

    def _getitem_one(self, video_id: str, index: int) -> dict[str, Any]:
        # per-item RNG derived from seed + index for reproducibility under shuffle
        rng = np.random.default_rng(self.cfg.seed + index * 10007)
        meta = self._meta(video_id)
        T, H, W = self.backend.shape(video_id)
        stats = self.backend.stats(video_id)
        mask, has_mask = self._load_mask(video_id, H, W)
        box = self.cropper.plan(H, W, rng, meta)

        frame_indices: list[int] = []

        if self.cfg.mode == "features":
            assert self.cached_features is not None
            video = self.backend.read_all(video_id)
            feats = self.cached_features(video, video_id=video_id)  # (H,W,C)
            feats = self.cropper.apply_image(feats, box)
            mask_c = self.cropper.apply_mask(mask, box)
            feats, mask_c = self.transforms.apply_features(feats, mask_c)
            image_t, mask_t = self.formatter.format_features(
                feats, mask_c, has_mask, stats=stats
            )
        else:
            assert self.frame_sampler is not None
            idx = self.frame_sampler(T, rng, meta)
            frame_indices = idx.tolist()
            frames = self.backend.read_frames(video_id, idx)  # (T,H,W)
            frames = self.cropper.apply_frames(frames, box)
            mask_c = self.cropper.apply_mask(mask, box)
            frames, mask_c = self.transforms.apply_temporal(frames, mask_c)
            image_t, mask_t = self.formatter.format_temporal(
                frames,
                mask_c,
                has_mask,
                stats=stats,
                add_dt_channel=self.cfg.temporal.add_dt_channel,
            )

        return {
            "image": image_t,
            "mask": mask_t,
            "video_id": video_id,
            "frame_indices": torch.tensor(frame_indices, dtype=torch.long)
            if frame_indices
            else torch.zeros(0, dtype=torch.long),
            "crop": torch.tensor(box.as_tuple, dtype=torch.long),
            "has_mask": torch.tensor(has_mask, dtype=torch.bool),
        }
