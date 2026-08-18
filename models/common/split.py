"""Video-level train/test split (no frame leakage). Same idea as HeatControl/u-net."""
from __future__ import annotations


def _series_key(vid: str) -> str:
    """R_002 → R, Z_010 → Z, sample12 → sample."""
    if "_" in vid:
        return vid.split("_", 1)[0]
    # sample12 / sample3
    i = 0
    while i < len(vid) and not vid[i].isdigit():
        i += 1
    return vid[:i] if i else vid


def split_videos(
    video_ids: list[str],
    test_every: int = 4,
) -> tuple[list[str], list[str]]:
    """Every n-th video of each series goes to test."""
    train, test = [], []
    keys = sorted({_series_key(v) for v in video_ids})
    for key in keys:
        series = sorted(v for v in video_ids if _series_key(v) == key)
        for i, vid in enumerate(series):
            (test if i % test_every == test_every - 1 else train).append(vid)
    return sorted(train), sorted(test)
