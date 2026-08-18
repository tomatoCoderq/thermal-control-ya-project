#!/usr/bin/env python3
"""Import updated TPU table masks (256×320 kaggle layout) → 240×320 raw frame space.

Reads ``Sample_*_Static`` / ``Calib_Sample_*`` from an external folder and writes:

  labels/tpu_binary_masks/sample{n}.png     — 0/255 for models (yaml)
  labels/tpu_sample_masks/sample{n}.png/.npy  — depth gray + float mm map
  datasets/dataset_tpu/labels/table_mask/   — same names, 240×320 for thermo

Example::

    python scripts/import_tpu_table_masks.py \\
        --src "/Users/user/Downloads/Telegram Desktop/table_mask"
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC_H, SRC_W = 256, 320
DST_H, DST_W = 240, 320

BINARY_DIR = ROOT / "labels" / "tpu_binary_masks"
DEPTH_DIR = ROOT / "labels" / "tpu_sample_masks"
TABLE_DIR = ROOT / "datasets" / "dataset_tpu" / "labels" / "table_mask"

ROI_PAD = 8


def _resize_nearest_u8(arr: np.ndarray) -> np.ndarray:
    if arr.shape == (DST_H, DST_W):
        return arr.astype(np.uint8, copy=False)
    return cv2.resize(arr.astype(np.uint8), (DST_W, DST_H), interpolation=cv2.INTER_NEAREST)


def _resize_depth_npy(arr: np.ndarray) -> np.ndarray:
    if arr.shape == (DST_H, DST_W):
        return arr.astype(np.float32, copy=False)
    valid = (~np.isnan(arr)).astype(np.uint8)
    valid_r = cv2.resize(valid, (DST_W, DST_H), interpolation=cv2.INTER_NEAREST) > 0
    filled = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    depth_r = cv2.resize(filled, (DST_W, DST_H), interpolation=cv2.INTER_NEAREST)
    depth_r = depth_r.astype(np.float32)
    depth_r[~valid_r] = np.nan
    return depth_r


def _stem_to_sample_name(stem: str) -> str | None:
    """Sample_3_Static → sample3 ; Calib skipped for models."""
    m = re.match(r"Sample_(\d+)_Static", stem, re.I)
    if m:
        return f"sample{int(m.group(1))}"
    return None


def _defect_roi(box: np.ndarray, pad: int = ROI_PAD) -> dict[str, int]:
    ys, xs = np.where(box)
    if xs.size == 0:
        return {"x": 0, "y": 0, "w": DST_W, "h": DST_H}
    H, W = box.shape
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(W, int(xs.max()) + 1 + pad)
    y1 = min(H, int(ys.max()) + 1 + pad)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def import_one(stem: str, src_dir: Path) -> dict | None:
    from PIL import Image

    png_src = src_dir / f"{stem}.png"
    npy_src = src_dir / f"{stem}.npy"
    if not png_src.exists() and not npy_src.exists():
        return None

    if npy_src.exists():
        depth = _resize_depth_npy(np.load(npy_src))
    else:
        depth = None

    if png_src.exists():
        gray = _resize_nearest_u8(np.array(Image.open(png_src)))
    elif depth is not None:
        # reconstruct gray from depth where valid
        dmax = float(np.nanmax(depth)) if np.any(~np.isnan(depth)) else 1.0
        gray = np.zeros((DST_H, DST_W), dtype=np.uint8)
        valid = ~np.isnan(depth) & (depth > 0)
        gray[valid] = np.clip(np.round(255 * depth[valid] / dmax), 0, 255).astype(np.uint8)
    else:
        raise FileNotFoundError(f"No png/npy for {stem}")

    binary = np.zeros((DST_H, DST_W), dtype=np.uint8)
    if depth is not None:
        binary[~np.isnan(depth) & (depth > 0)] = 255
    else:
        binary[gray > 0] = 255

    # table_mask (thermo): keep depth naming
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray, mode="L").save(TABLE_DIR / f"{stem}.png")
    if depth is not None:
        np.save(TABLE_DIR / f"{stem}.npy", depth)
    elif npy_src.exists():
        shutil.copy2(npy_src, TABLE_DIR / f"{stem}.npy")

    sample_name = _stem_to_sample_name(stem)
    row: dict = {"table_name": stem, "sample_name": sample_name or ""}
    if sample_name is not None:
        BINARY_DIR.mkdir(parents=True, exist_ok=True)
        DEPTH_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary, mode="L").save(BINARY_DIR / f"{sample_name}.png")
        Image.fromarray(gray, mode="L").save(DEPTH_DIR / f"{sample_name}.png")
        if depth is not None:
            np.save(DEPTH_DIR / f"{sample_name}.npy", depth)
        row.update({"sample_no": int(sample_name.replace("sample", "")), **_defect_roi(binary > 0)})

    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Folder with Sample_*_Static.png/.npy (256×320)",
    )
    args = ap.parse_args()
    src_dir = args.src.expanduser().resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(src_dir)

    stems = sorted(
        {p.stem for p in src_dir.glob("*.png")} | {p.stem for p in src_dir.glob("*.npy")}
    )
    stems = [s for s in stems if "Sample" in s or "sample" in s]
    if not stems:
        raise FileNotFoundError(f"No Sample_* masks in {src_dir}")

    rows: list[dict] = []
    for stem in stems:
        row = import_one(stem, src_dir)
        if row is None:
            continue
        rows.append(row)
        tag = row.get("sample_name") or stem
        print(f"  {stem:28s} → {tag}  {DST_H}×{DST_W}")

    seg_rows = [r for r in rows if r.get("sample_name")]
    if seg_rows:
        with open(BINARY_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["sample_name", "sample_no", "x", "y", "w", "h"],
            )
            w.writeheader()
            for r in seg_rows:
                w.writerow({k: r[k] for k in w.fieldnames if k in r})

        from PIL import Image

        first_bin = np.array(Image.open(BINARY_DIR / f"{seg_rows[0]['sample_name']}.png")) > 0
        roi = _defect_roi(first_bin)
        for dest in (BINARY_DIR, DEPTH_DIR):
            with open(dest / "roi.yaml", "w", encoding="utf-8") as f:
                f.write(f"# ROI from imported table masks ({DST_H}×{DST_W})\n")
                f.write(f"roi_padding: {ROI_PAD}\n")
                f.write("roi:\n")
                for k, v in roi.items():
                    f.write(f"  {k}: {v}\n")

    with open(TABLE_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["table_name", "sample_name"])
        w.writeheader()
        w.writerows([{"table_name": r["table_name"], "sample_name": r.get("sample_name", "")} for r in rows])

    print(f"\nГотово: {len(rows)} table_mask → {TABLE_DIR}")
    print(f"        {len(seg_rows)} models → {BINARY_DIR}")

    from PIL import Image

    for p in TABLE_DIR.glob("*.png"):
        if p.stem in {"Sample_1_Static", "Sample_2_Static", "Sample_9_Static"}:
            p.unlink(missing_ok=True)
            p.with_suffix(".npy").unlink(missing_ok=True)
        else:
            arr = np.array(Image.open(p))
            if arr.shape != (DST_H, DST_W):
                raise RuntimeError(f"Unexpected size {arr.shape} in {p.name}")


if __name__ == "__main__":
    main()
