"""Save a visual check of the configured specimen crop on raw thermal frames."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/thermal-control-matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from irt_data.config import DatasetConfig
from irt_data.dataset import IRTDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/augmentation_ppt.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 440, 900])
    parser.add_argument("--output", default="artifacts/object_crop_preview.png")
    args = parser.parse_args()

    import yaml

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.data_root:
        raw.get("dataset", raw)["sources"][0]["root"] = args.data_root
    if args.roi:
        x, y, w, h = args.roi
        raw.get("dataset", raw)["sources"][0]["object_roi"] = {
            "x": x, "y": y, "w": w, "h": h
        }
    cfg = DatasetConfig.from_dict(raw.get("dataset", raw))
    cfg.train = False
    dataset = IRTDataset(cfg)
    video_id = args.video_id or dataset.video_ids[0]
    if video_id not in dataset.video_ids:
        raise KeyError(f"Unknown video {video_id}; examples: {dataset.video_ids[:5]}")

    T, H, W = dataset.backend.shape(video_id)
    indices = [min(max(0, i), T - 1) for i in args.frames]
    frames = dataset.backend.read_frames(video_id, indices)
    meta = dataset._meta(video_id)
    source_roi = dataset._source_object_rois.get(video_id)
    box = dataset.object_cropper.box_for(H, W, meta, source_roi)
    cropped = dataset.object_cropper.apply_frames(frames, box)

    fig, axes = plt.subplots(len(indices), 2, figsize=(9, 4 * len(indices)), squeeze=False)
    for row, (frame_no, original, crop) in enumerate(zip(indices, frames, cropped)):
        axes[row, 0].imshow(original, cmap="inferno")
        axes[row, 0].add_patch(
            Rectangle(
                (box.x0, box.y0), box.x1 - box.x0, box.y1 - box.y0,
                fill=False, edgecolor="cyan", linewidth=2,
            )
        )
        axes[row, 0].set_title(f"{video_id}: raw frame {frame_no}")
        axes[row, 1].imshow(crop, cmap="inferno")
        axes[row, 1].set_title("object crop before resize/PPT")
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
