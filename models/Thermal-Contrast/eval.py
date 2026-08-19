"""Evaluate a trained checkpoint on the test split and show one sample."""
from __future__ import annotations

import argparse
from pathlib import Path

from paths import DATASETS_ROOT

import matplotlib.pyplot as plt
import torch

from channels import CHANNEL_NAMES, CHANNEL_TITLES, DEFAULT_PARAMS, ChannelParams
from common.device import get_device
from data import build_datasets
from inference import evaluate_dataset, load_trained_model, predict_channels
from train import CHECKPOINT_BEST, resolve_checkpoint


def show_sample(dataset, model, device, position: int) -> None:
    channels, mask = dataset.channels_and_mask(dataset.indices[position])
    out = predict_channels(model, channels, device)
    truth = mask.squeeze().numpy()
    video_id = dataset.video_ids[position]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for column, name in enumerate(CHANNEL_NAMES):
        axes[0, column].imshow(channels[column].numpy(), cmap="inferno", vmin=0, vmax=1)
        axes[0, column].set_title(CHANNEL_TITLES[name])
    axes[1, 0].imshow(truth, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("GT")
    axes[1, 1].imshow(out["prob"], cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title("probability")
    axes[1, 2].imshow(out["pred"], cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("prediction")
    axes[1, 3].imshow(channels[0].numpy(), cmap="inferno", vmin=0, vmax=1)
    axes[1, 3].contour(truth, levels=[0.5], colors="lime", linewidths=1.2)
    axes[1, 3].set_title("overlay")
    for axis in axes.ravel():
        axis.axis("off")
    fig.suptitle(video_id)
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Thermal-Contrast U-Net")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_BEST))
    parser.add_argument("--root", type=Path, default=DATASETS_ROOT)
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument("--test-every", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_PARAMS.num_frames)
    parser.add_argument("--show", type=int, default=0, help="test index to plot; -1 to skip")
    args = parser.parse_args()

    params = ChannelParams(num_frames=args.num_frames)
    _, test_ds = build_datasets(
        root=args.root, include=args.include, params=params, test_every=args.test_every, augment=False
    )
    device = get_device()
    model = load_trained_model(resolve_checkpoint(args.checkpoint), device)

    rows = evaluate_dataset(model, test_ds, device)
    print(f"\n{'video':12s} {'dice':>7s} {'iou':>7s} {'gt%':>7s} {'pred%':>7s}")
    for row in rows:
        print(
            f"{row['video_id']:12s} {row['dice']:7.4f} {row['iou']:7.4f} "
            f"{100 * row['gt_positive']:7.2f} {100 * row['pred_positive']:7.2f}"
        )
    mean_dice = sum(r["dice"] for r in rows) / len(rows)
    mean_iou = sum(r["iou"] for r in rows) / len(rows)
    print(f"\nmean over {len(rows)} test videos: dice={mean_dice:.4f} iou={mean_iou:.4f}")

    if args.show >= 0:
        show_sample(test_ds, model, device, args.show)


if __name__ == "__main__":
    main()
