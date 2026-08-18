"""Convert .mat IRT videos to float16 memmap .npy caches for fast random frame access."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _load_mat_array(
    path: Path,
    time_axis: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a .mat video and return (T, H, W) float32 plus metadata."""
    try:
        mat = sio.loadmat(str(path), simplify_cells=True)
    except TypeError:
        mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)

    key = None
    for candidate in ("data", "imageArray"):
        if candidate in mat and isinstance(mat[candidate], np.ndarray):
            key = candidate
            break
    if key is None:
        arrs = [
            (k, v)
            for k, v in mat.items()
            if isinstance(v, np.ndarray) and not str(k).startswith("__") and v.ndim >= 3
        ]
        if not arrs:
            raise KeyError(f"No 3D array in {path}")
        key, _ = max(arrs, key=lambda kv: kv[1].size)

    arr = np.asarray(mat[key])
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array in {path}, got {arr.shape}")

    if time_axis is not None:
        arr = np.moveaxis(arr, time_axis, 0)
    elif arr.shape[-1] > max(arr.shape[0], arr.shape[1]):
        # typical (H, W, T)
        arr = np.moveaxis(arr, -1, 0)
    elif arr.shape[0] <= max(arr.shape[1:]):
        # ambiguous but prefer moving largest dim to time if it is last or first
        if arr.shape[0] < arr.shape[-1]:
            arr = np.moveaxis(arr, -1, 0)

    arr = arr.astype(np.float32)

    fps = 0.0
    for fps_key in ("Fs", "FPS", "fps"):
        if fps_key in mat:
            try:
                fps = float(np.asarray(mat[fps_key]).squeeze())
            except Exception:
                fps = 0.0
            break

    meta = {
        "source": str(path),
        "key": key,
        "fps": fps if fps > 0 else None,
        "dtype_src": str(np.asarray(mat[key]).dtype),
    }
    return arr, meta


def video_id_from_path(path: Path) -> str:
    """Stable id: stem with spaces replaced."""
    return path.stem.replace(" ", "_")


def cache_one(
    mat_path: Path,
    out_dir: Path,
    time_axis: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert one .mat to float16 (T,H,W) .npy and return index entry."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vid = video_id_from_path(mat_path)
    npy_path = out_dir / f"{vid}.npy"

    if npy_path.exists() and not overwrite:
        try:
            arr = np.load(npy_path, mmap_mode="r")
            _ = arr.shape  # force header / mmap open
            logger.info("skip existing %s", npy_path.name)
            entry = {
                "video_id": vid,
                "npy": str(npy_path.resolve()),
                "shape": list(arr.shape),
                "T": int(arr.shape[0]),
                "H": int(arr.shape[1]),
                "W": int(arr.shape[2]),
                "dtype": str(arr.dtype),
                "source": str(mat_path),
                "skipped": True,
            }
            return entry
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "corrupt cache %s (%s) — rebuilding from %s",
                npy_path.name,
                exc,
                mat_path.name,
            )
            try:
                npy_path.unlink(missing_ok=True)
            except OSError:
                pass

    video, meta = _load_mat_array(mat_path, time_axis=time_axis)
    T, H, W = video.shape
    # stats in float32 before casting
    vmin = float(video.min())
    vmax = float(video.max())
    vmean = float(video.mean())
    vstd = float(video.std())

    np.save(npy_path, video.astype(np.float16))
    entry = {
        "video_id": vid,
        "npy": str(npy_path.resolve()),
        "shape": [T, H, W],
        "T": T,
        "H": H,
        "W": W,
        "dtype": "float16",
        "min": vmin,
        "max": vmax,
        "mean": vmean,
        "std": vstd,
        "fps": meta.get("fps"),
        "source": str(mat_path.resolve()),
        "key": meta.get("key"),
        "dtype_src": meta.get("dtype_src"),
    }
    logger.info("cached %s -> %s shape=%s", mat_path.name, npy_path.name, (T, H, W))
    return entry


def build_cache(
    sources: list[str | Path],
    out_dir: str | Path = "artifacts/cache",
    pattern: str = "*.mat",
    time_axis: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert all .mat under sources into float16 memmap caches + index.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"

    existing: dict[str, Any] = {"videos": {}}
    if index_path.exists() and not overwrite:
        with open(index_path, encoding="utf-8") as f:
            existing = json.load(f)

    files: list[Path] = []
    for src in sources:
        root = Path(src)
        if root.is_file() and root.suffix.lower() == ".mat":
            files.append(root)
        else:
            files.extend(sorted(root.glob(pattern)))

    videos = dict(existing.get("videos", {}))
    for path in tqdm(files, desc="cache .mat -> .npy"):
        try:
            entry = cache_one(path, out_dir, time_axis=time_axis, overwrite=overwrite)
            videos[entry["video_id"]] = entry
        except Exception as exc:
            logger.exception("failed to cache %s: %s", path, exc)

    index = {
        "cache_dir": str(out_dir.resolve()),
        "dtype": "float16",
        "layout": "THW",
        "videos": videos,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    logger.info("wrote %s (%d videos)", index_path, len(videos))
    return index


def load_index(cache_dir: str | Path) -> dict[str, Any]:
    path = Path(cache_dir) / "index.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Cache index not found: {path}. Run: python -m irt_data.cache --sources ..."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build float16 .npy cache from .mat videos")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["archive/data"],
        help="Folders or .mat files to convert",
    )
    parser.add_argument("--out", default="artifacts/cache", help="Output cache directory")
    parser.add_argument("--pattern", default="*.mat")
    parser.add_argument("--time-axis", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    build_cache(
        args.sources,
        out_dir=args.out,
        pattern=args.pattern,
        time_axis=args.time_axis,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
