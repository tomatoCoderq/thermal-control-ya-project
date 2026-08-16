"""I/O backends: read frames and masks without math."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from irt_data.cache import _load_mat_array, load_index, video_id_from_path

logger = logging.getLogger(__name__)


class IOBackend(Protocol):
    """Protocol for reading video frames as (T_sel, H, W) float32."""

    def num_frames(self, video_id: str) -> int: ...

    def shape(self, video_id: str) -> tuple[int, int, int]:
        """Return (T, H, W)."""
        ...

    def read_frames(self, video_id: str, indices: np.ndarray | list[int]) -> np.ndarray:
        """Return float32 array of shape (len(indices), H, W)."""
        ...

    def read_all(self, video_id: str) -> np.ndarray:
        """Return full video (T, H, W) float32 (may be expensive)."""
        ...

    def stats(self, video_id: str) -> dict[str, float]:
        """Return min/max/mean/std if known."""
        ...


class NpyMemmapBackend:
    """Read frames via np.load(..., mmap_mode='r') — only touches requested pages."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.index = load_index(self.cache_dir)
        self._maps: dict[str, np.memmap] = {}

    def list_ids(self) -> list[str]:
        return sorted(self.index["videos"].keys())

    def _entry(self, video_id: str) -> dict:
        if video_id not in self.index["videos"]:
            raise KeyError(f"video_id '{video_id}' not in cache index {self.cache_dir}")
        return self.index["videos"][video_id]

    def _mmap(self, video_id: str) -> np.memmap:
        if video_id not in self._maps:
            entry = self._entry(video_id)
            path = Path(entry["npy"])
            if not path.is_absolute():
                path = self.cache_dir / path.name
            self._maps[video_id] = np.load(path, mmap_mode="r")
        return self._maps[video_id]

    def num_frames(self, video_id: str) -> int:
        return int(self._entry(video_id)["T"])

    def shape(self, video_id: str) -> tuple[int, int, int]:
        e = self._entry(video_id)
        return int(e["T"]), int(e["H"]), int(e["W"])

    def read_frames(self, video_id: str, indices: np.ndarray | list[int]) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        arr = self._mmap(video_id)
        # fancy indexing on memmap copies selected frames
        frames = np.asarray(arr[idx], dtype=np.float32)
        return frames

    def read_all(self, video_id: str) -> np.ndarray:
        return np.asarray(self._mmap(video_id), dtype=np.float32)

    def stats(self, video_id: str) -> dict[str, float]:
        e = self._entry(video_id)
        out: dict[str, float] = {}
        for k in ("min", "max", "mean", "std"):
            if k in e and e[k] is not None:
                out[k] = float(e[k])
        return out


class MatIOBackend:
    """Fallback: load full .mat into RAM with a one-slot LRU cache."""

    def __init__(
        self,
        mat_paths: dict[str, Path],
        time_axis: int | None = None,
    ) -> None:
        self.mat_paths = {k: Path(v) for k, v in mat_paths.items()}
        self.time_axis = time_axis
        self._cached_id: str | None = None
        self._cached_video: np.ndarray | None = None

    def list_ids(self) -> list[str]:
        return sorted(self.mat_paths.keys())

    def _load(self, video_id: str) -> np.ndarray:
        if self._cached_id == video_id and self._cached_video is not None:
            return self._cached_video
        path = self.mat_paths[video_id]
        video, _ = _load_mat_array(path, time_axis=self.time_axis)
        self._cached_id = video_id
        self._cached_video = video
        return video

    def num_frames(self, video_id: str) -> int:
        return int(self._load(video_id).shape[0])

    def shape(self, video_id: str) -> tuple[int, int, int]:
        v = self._load(video_id)
        return int(v.shape[0]), int(v.shape[1]), int(v.shape[2])

    def read_frames(self, video_id: str, indices: np.ndarray | list[int]) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        return self._load(video_id)[idx].astype(np.float32)

    def read_all(self, video_id: str) -> np.ndarray:
        return self._load(video_id).astype(np.float32)

    def stats(self, video_id: str) -> dict[str, float]:
        v = self._load(video_id)
        return {
            "min": float(v.min()),
            "max": float(v.max()),
            "mean": float(v.mean()),
            "std": float(v.std()),
        }


class MaskReader:
    """Load PNG masks. Returns None if missing (caller decides policy)."""

    def __init__(self, mask_dirs: dict[str, Path | None]) -> None:
        """mask_dirs maps video_id -> directory containing {video_id}.png or stem.png."""
        self.mask_dirs = {k: (Path(v) if v is not None else None) for k, v in mask_dirs.items()}
        self._stem_aliases: dict[str, str] = {}

    def register_alias(self, video_id: str, stem: str) -> None:
        self._stem_aliases[video_id] = stem

    def path_for(self, video_id: str) -> Path | None:
        directory = self.mask_dirs.get(video_id)
        if directory is None:
            return None
        stem = self._stem_aliases.get(video_id, video_id)
        # try exact, then without underscore replacement, then common stems
        candidates = [
            directory / f"{stem}.png",
            directory / f"{video_id}.png",
            directory / f"{video_id.replace('_', ' ')}.png",
        ]
        # Kaggle ids like R_002 stay as-is; sample3 from sample3.mat
        for c in candidates:
            if c.exists():
                return c
        # fuzzy: any png whose stem matches ignoring case/spaces
        if directory.exists():
            target = stem.replace(" ", "_").lower()
            for p in directory.glob("*.png"):
                if p.stem.replace(" ", "_").lower() == target:
                    return p
        return None

    def read(self, video_id: str) -> np.ndarray | None:
        path = self.path_for(video_id)
        if path is None:
            return None
        arr = np.asarray(Image.open(path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.uint8)


def discover_mat_files(
    roots: list[Path],
    pattern: str = "*.mat",
) -> dict[str, Path]:
    """Map video_id -> mat path from one or more roots."""
    out: dict[str, Path] = {}
    for root in roots:
        root = Path(root)
        for p in sorted(root.glob(pattern)):
            vid = video_id_from_path(p)
            out[vid] = p
    return out
