"""Read an arbitrary `.mat` file into the exact tensor a `TermoDataset` item holds.

`TermoDataset.__getitem__` turns a `.mat` into a video tensor in three steps: read
the `(H, W, T)` array as float32, permute it to `(T, H, W)`, then resize the spatial
axes to 256x256 bilinearly. `load_video_from_mat` repeats those steps so a video
handed to the model at inference time is preprocessed exactly like a training
sample - `tests/test_video_io.py` asserts the two agree bit for bit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

STANDARD_SIZE: tuple[int, int] = (256, 256)


def mat_video_keys(mat_path: str | Path) -> list[str]:
    """Names of the 3-D arrays in a `.mat` file, i.e. the candidate videos."""
    mat = loadmat(str(mat_path))
    return [
        key
        for key, value in mat.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 3
    ]


def load_video_from_mat(
    mat_path: str | Path,
    mat_key: str | None = None,
    size: tuple[int, int] = STANDARD_SIZE,
) -> torch.Tensor:
    """`.mat` file → `(T, size[0], size[1])` float32 video, as `TermoDataset` serves it.

    With `mat_key=None` the file must hold exactly one 3-D array, which is then
    used; that is what makes an unfamiliar `.mat` readable without knowing whether
    its variable is called `data`, `imageArray` or something else.
    """
    mat = loadmat(str(mat_path))
    if mat_key is None:
        keys = [
            key
            for key, value in mat.items()
            if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 3
        ]
        if len(keys) != 1:
            raise KeyError(
                f"{mat_path}: expected exactly one 3-D array, found {keys or 'none'}; "
                "pass mat_key explicitly"
            )
        mat_key = keys[0]
    elif mat_key not in mat:
        available = [key for key in mat if not key.startswith("__")]
        raise KeyError(f"{mat_path}: no key {mat_key!r}; available keys: {available}")

    array = np.asarray(mat[mat_key])
    if array.ndim != 3:
        raise ValueError(f"{mat_path}: key {mat_key!r} has shape {array.shape}, expected (H, W, T)")

    video = torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1)
    return F.interpolate(video.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
