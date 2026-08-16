"""Train SegFormer on PPT phase maps with video-level data splits."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from irt_data.config import DatasetConfig
from irt_data.dataset import IRTDataset
from irt_data.loaders import stack_collate, worker_init_fn
from segformer.metrics import binary_counts, cross_entropy_dice_loss, metrics_from_counts
from segformer.model import build_segformer
from segformer.splits import indices_for_videos, split_video_ids


def _loader(dataset, indices, cfg: DatasetConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=cfg.loader.batch_size,
        shuffle=shuffle,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        collate_fn=stack_collate,
        worker_init_fn=worker_init_fn if cfg.loader.num_workers > 0 else None,
    )


def _forward(model, images: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
    logits = model(pixel_values=images).logits
    return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


def precompute_features(dataset: IRTDataset) -> None:
    """Fill the PPT cache sequentially before multi-worker training starts."""
    first_index: dict[str, int] = {}
    for index, video_id in enumerate(dataset._index):
        first_index.setdefault(video_id, index)
    total = len(first_index)
    for number, (video_id, index) in enumerate(first_index.items(), start=1):
        print(f"PPT cache {number}/{total}: {video_id}")
        dataset._getitem_one(video_id, index)


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    counts = [0, 0, 0, 0]
    total_loss = 0.0
    batches = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = _forward(model, images, masks.shape[-2:])
        total_loss += float(cross_entropy_dice_loss(logits, masks))
        counts = [a + b for a, b in zip(counts, binary_counts(logits, masks))]
        batches += 1
    result = metrics_from_counts(*counts)
    result["loss"] = total_loss / max(1, batches)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segformer.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--mask-root", default=None)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()
    config_path = Path(args.config)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if "dataset_config" in raw:
        dataset_path = Path(raw["dataset_config"])
        if not dataset_path.is_absolute():
            candidate = config_path.parent / dataset_path
            dataset_path = candidate if candidate.exists() else dataset_path
        dataset_doc = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
        dataset_raw = dataset_doc.get("dataset", dataset_doc)
    else:
        dataset_raw = raw["dataset"]
    if args.data_root:
        dataset_raw["sources"][0]["root"] = args.data_root
    if args.mask_root:
        dataset_raw["sources"][0]["masks"] = args.mask_root
    if args.work_dir:
        work = Path(args.work_dir)
        dataset_raw["cache_dir"] = str(work / "cache")
        dataset_raw["features"]["cache_dir"] = str(work / "ppt_features")
        raw["training"]["output_dir"] = str(work / "segformer")
    ds_cfg = DatasetConfig.from_dict(deepcopy(dataset_raw))
    train_cfg = raw["training"]

    if ds_cfg.mode != "features" or ds_cfg.features.extractors != ["ppt"]:
        raise ValueError("SegFormer baseline expects mode=features and extractors=[ppt]")
    if ds_cfg.mask.kind != "binary" or ds_cfg.mask.missing != "error":
        raise ValueError("Supervised training requires binary masks and missing=error")

    torch.manual_seed(ds_cfg.seed)
    train_dataset = IRTDataset(ds_cfg)
    if bool(train_cfg.get("precompute_features", True)):
        precompute_features(train_dataset)
    eval_raw = deepcopy(dataset_raw)
    eval_raw["train"] = False
    eval_raw["samples_per_video"] = 1
    eval_dataset = IRTDataset(DatasetConfig.from_dict(eval_raw))

    split = split_video_ids(
        train_dataset.video_ids,
        val_fraction=float(train_cfg.get("val_fraction", 0.15)),
        test_fraction=float(train_cfg.get("test_fraction", 0.15)),
        seed=int(train_cfg.get("split_seed", 42)),
    )
    train_loader = _loader(
        train_dataset, indices_for_videos(train_dataset, split.train), ds_cfg, True
    )
    val_loader = _loader(
        eval_dataset, indices_for_videos(eval_dataset, split.val), ds_cfg, False
    )
    test_loader = _loader(
        eval_dataset, indices_for_videos(eval_dataset, split.test), ds_cfg, False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segformer(
        model_name=train_cfg.get("model_name", "nvidia/mit-b0"),
        pretrained=bool(train_cfg.get("pretrained", True)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 6e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )
    epochs = int(train_cfg.get("epochs", 30))
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    output_dir = Path(train_cfg.get("output_dir", "artifacts/segformer"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split.json").write_text(
        json.dumps(split.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_dice = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = _forward(model, images, masks.shape[-2:])
                loss = cross_entropy_dice_loss(
                    logits, masks, dice_weight=float(train_cfg.get("dice_weight", 0.5))
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())

        val = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch:03d} train_loss={running / max(1, len(train_loader)):.4f} "
            f"val_loss={val['loss']:.4f} dice={val['dice']:.4f} iou={val['iou']:.4f}"
        )
        if val["dice"] > best_dice:
            best_dice = val["dice"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val": val,
                    "split": split.__dict__,
                },
                output_dir / "best.pt",
            )

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, test_loader, device)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test, indent=2), encoding="utf-8"
    )
    print("test", json.dumps(test))


if __name__ == "__main__":
    main()
