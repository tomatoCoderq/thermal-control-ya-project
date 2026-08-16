"""Fourier phase classification for the same data used by p5 experiments.

Run from the workspace root:
    python fourier-exps/main.py --precompute-only
    python fourier-exps/main.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.io import whosmat

from config import CFG
from fourier import (
    cache_matches,
    fourier_phase_features,
    load_thermal_video,
    save_feature_cache,
)


def list_videos() -> list[str]:
    videos = sorted(path.stem for path in CFG.paths.data_dir.glob("*.mat"))
    if CFG.train.max_videos is not None:
        videos = videos[:CFG.train.max_videos]
    return videos


def _video_shape(path: Path) -> tuple[int, int, int]:
    entries = whosmat(path)
    for name in ("imageArray", "data", "IMAGES"):
        for variable, shape, _dtype in entries:
            if variable == name and len(shape) == 3:
                return tuple(int(value) for value in shape)
    raise KeyError(f"3-D thermal array not found in {path}")


def _feature_settings(path: Path) -> dict:
    stat = path.stat()
    shape = _video_shape(path)
    return {
        **CFG.fourier.model_dump(mode="json", exclude={"n_channels"}),
        "n_frames": shape[-1],
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def precompute_features(force: bool = False) -> None:
    CFG.paths.feature_dir.mkdir(parents=True, exist_ok=True)
    videos = list_videos()
    if not videos:
        raise FileNotFoundError(f"no .mat files found in {CFG.paths.data_dir}")

    for index, video_id in enumerate(videos, 1):
        source = CFG.paths.data_dir / f"{video_id}.mat"
        output = CFG.paths.feature_dir / f"{video_id}.npy"
        settings = _feature_settings(source)
        if not force and cache_matches(output, settings):
            print(f"[{index:02d}/{len(videos):02d}] cache {video_id}")
            continue

        print(f"[{index:02d}/{len(videos):02d}] FFT {video_id}")
        thermal_video, sampling_hz = load_thermal_video(source)
        features, bins, start_frame = fourier_phase_features(
            thermal_video,
            first_bin=CFG.fourier.first_bin,
            n_frequencies=CFG.fourier.n_frequencies,
            phase_encoding=CFG.fourier.phase_encoding,
            window=CFG.fourier.window,
            detrend=CFG.fourier.detrend,
            start_at_peak=CFG.fourier.start_at_peak,
            row_chunk=CFG.fourier.row_chunk,
        )
        save_feature_cache(
            output,
            features,
            source=source,
            sampling_hz=sampling_hz,
            bins=bins,
            start_frame=start_frame,
            settings=settings,
        )
        metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        frequencies = ", ".join(f"{value:.5f}" for value in metadata["frequencies_hz"])
        print(f"    shape={features.shape}; frequencies Hz: {frequencies}")


def load_mask_classes(video_id: str) -> np.ndarray:
    mask_path = CFG.paths.mask_dir / f"{video_id}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"mask not found: {mask_path}")
    image = np.asarray(Image.open(mask_path).convert("L"))
    classes = np.zeros_like(image, dtype=np.uint8)
    known = np.zeros_like(image, dtype=bool)
    for gray, class_id in CFG.classes.gray2cls.items():
        selected = image == gray
        classes[selected] = class_id
        known |= selected
    if not known.all():
        values = sorted(int(value) for value in np.unique(image[~known]))
        raise ValueError(f"unknown gray levels in {mask_path.name}: {values}")
    return classes


def build_index() -> list[tuple[str, int, int, int]]:
    size = CFG.crop.crop
    rng = np.random.default_rng(CFG.train.seed)
    rows: list[tuple[str, int, int, int]] = []
    for video_id in list_videos():
        classes = load_mask_classes(video_id)
        height, width = classes.shape
        if height < size or width < size:
            raise ValueError(f"crop {size} does not fit mask {video_id}: {classes.shape}")

        for class_id in range(1, CFG.classes.n_classes):
            components, count = ndimage.label(classes == class_id)
            for component_id in range(1, count + 1):
                rr, cc = ndimage.center_of_mass(components == component_id)
                r0 = int(np.clip(rr - size // 2, 0, height - size))
                c0 = int(np.clip(cc - size // 2, 0, width - size))
                rows.append((video_id, r0, c0, class_id))

        background = classes[size:height - size, size:width - size] == 0
        ys, xs = np.where(background)
        if len(ys):
            selected = rng.choice(
                len(ys), min(CFG.crop.n_bg_per_video, len(ys)), replace=False
            )
            for item in selected:
                rows.append(
                    (video_id, int(ys[item] + size // 2), int(xs[item] + size // 2), 0)
                )
    return rows


def split_by_video(index):
    rng = np.random.default_rng(CFG.train.seed)
    videos = np.array(sorted({row[0] for row in index}))
    if len(videos) < 2:
        raise ValueError("training requires at least two videos")
    n_test = min(CFG.train.n_test_videos, len(videos) - 1)
    test_videos = set(rng.permutation(videos)[:n_test])
    train = [row for row in index if row[0] not in test_videos]
    test = [row for row in index if row[0] in test_videos]
    return train, test


def compute_norm(index) -> tuple[np.ndarray, np.ndarray]:
    if not index:
        raise ValueError("empty training index")
    size = CFG.crop.crop
    channel_sum = np.zeros(CFG.fourier.n_channels, dtype=np.float64)
    channel_sq_sum = np.zeros(CFG.fourier.n_channels, dtype=np.float64)
    count = 0
    cache = {}
    for video_id, r0, c0, _label in index:
        if video_id not in cache:
            cache[video_id] = np.load(
                CFG.paths.feature_dir / f"{video_id}.npy", mmap_mode="r"
            )
        crop = np.asarray(cache[video_id][:, r0:r0 + size, c0:c0 + size], dtype=np.float64)
        channel_sum += crop.sum(axis=(1, 2))
        channel_sq_sum += np.square(crop).sum(axis=(1, 2))
        count += crop.shape[1] * crop.shape[2]
    mean = channel_sum / count
    variance = np.maximum(channel_sq_sum / count - np.square(mean), 0.0)
    return mean.astype(np.float32), (np.sqrt(variance) + 1e-6).astype(np.float32)


def train_model() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "training requires PyTorch; install fourier-exps/requirements.txt"
        ) from exc

    from datasets import FourierDataset
    from engine import fit
    from losses import make_loss
    from model import make_model

    torch.manual_seed(CFG.train.seed)
    np.random.seed(CFG.train.seed)
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )

    index = build_index()
    train_index, test_index = split_by_video(index)
    mean, std = compute_norm(train_index)
    print(
        f"device={device} | channels={CFG.fourier.n_channels} | "
        f"crops={len(index)} train={len(train_index)} test={len(test_index)}"
    )

    train_dataset = FourierDataset(train_index, mean, std, train=True)
    test_dataset = FourierDataset(test_index, mean, std, train=False)
    train_loader = DataLoader(
        train_dataset, batch_size=CFG.train.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CFG.train.batch_size, shuffle=False
    )

    model = make_model(CFG.fourier.n_channels).to(device)
    loss_fn = make_loss(
        CFG.train.loss_name,
        CFG.classes.n_classes,
        [row[-1] for row in train_index],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.train.learning_rate)
    tag = f"fft_phase_{CFG.fourier.n_frequencies}f_{CFG.train.loss_name}"
    run_dir, result = fit(
        train_loader, test_loader, model, loss_fn, optimizer, device, tag
    )
    print("done:", result, "| artifacts:", run_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Fourier phase defect-depth classification")
    parser.add_argument("--precompute-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="rebuild valid feature caches")
    parser.add_argument("--max-videos", type=int, default=None, help="debug subset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_videos is not None:
        CFG.train.max_videos = args.max_videos
    precompute_features(force=args.force)
    if not args.precompute_only:
        train_model()


if __name__ == "__main__":
    main()





