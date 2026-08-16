"""Материализация аугментированных датасетов в СЫРОМ виде (термокадры).

Для каждого видео:
  raw video (T,H,W) → ROI-crop рамки → resize → синхронная пространственная
  аугментация (одинаковая геометрия для ВСЕХ кадров и маски) → сохранение на диск.

Число кадров T сохраняется. PPT/TSR тут НЕ считаются — это сырьё, признаки
посчитает твой datasetLoader уже поверх аугментированных видео.

Используются модули пакета irt_data (crop/resize/aug), без IRTDataset/PPT.

Выход: datasets/augmented/<domain>/<vid>__aug{k}.npy   (float32, T×Ho×Wo)
                                   <vid>__aug{k}_mask.png (uint8, Ho×Wo)
       datasets/augmented/manifest.csv
"""
from __future__ import annotations

import csv
import glob
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

from irt_data.config import ObjectCropConfig, ROI, AugConfig, AugSpec
from irt_data.crops import ObjectCropper, resize_2d, apply_crop_thw
from irt_data.transforms import TransformPipeline

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "datasets" / "augmented"

# ── настройки генерации ──────────────────────────────────────────────────────
OUT_SIZE = (256, 256)     # размер после ROI-crop + resize
N_AUG = 1                 # сколько аугментированных копий на видео
SEED = 67

# ROI объекта (в координатах СЫРОГО кадра каждого домена)
KAGGLE_ROI = ROI(80, 25, 178, 190)   # kaggle raw 256x320
TPU_ROI = ROI(80, 43, 178, 120)      # tpu raw 240x320

# глубина залегания (мм) по номеру tpu-образца (Таблица 1)
TPU_DEPTH_MM = {
    1: 3.1, 2: 5.2, 3: 1.0, 4: 3.6, 5: 5.7, 6: 1.5, 7: 2.6, 8: 4.7,
    9: 0.5, 10: 4.2, 11: 2.1, 12: 4.0, 13: 6.1, 14: 1.9, 15: 2.1,
    16: 2.2, 17: 3.8, 18: 5.9, 19: 1.7, 20: 2.1, 21: 4.2,
}
GRAY2CLS = {0: 0, 51: 1, 102: 2, 153: 3, 204: 4, 255: 5}

# синхронные пространственные аугментации (train)
AUG = AugConfig(
    spatial=[
        AugSpec("HorizontalFlip", {"p": 0.5}),
        AugSpec("VerticalFlip", {"p": 0.5}),
        AugSpec("RandomRotate90", {"p": 0.5}),
    ],
    use_replay_fallback=True,   # одинаковые параметры на все кадры
)


# ── чтение сырья ─────────────────────────────────────────────────────────────
def load_raw_video(path: str) -> np.ndarray:
    """(T,H,W) float32 из .mat."""
    m = sio.loadmat(path)
    key = next(k for k in ("imageArray", "data", "IMAGES")
               if k in m and np.asarray(m[k]).ndim == 3)
    return np.transpose(np.asarray(m[key]).astype(np.float32), (2, 0, 1))


def kaggle_mask(vid: str) -> np.ndarray:
    im = np.array(Image.open(ROOT / "datasets/dataset_kaggle/labels/automated_mask" / f"{vid}.png"))
    cls = np.zeros_like(im, dtype=np.uint8)
    for g, c in GRAY2CLS.items():
        cls[im == g] = c
    return cls


def tpu_base_mask() -> np.ndarray:
    """Бинарная маска блоков из base_mask.png (сырые 240x320)."""
    a = np.array(Image.open(ROOT / "datasets/dataset_tpu/base_mask.png"))
    g = a[..., 1] if a.ndim == 3 else a       # блоки закодированы в зелёном канале
    return (g > 0).astype(np.uint8)


def tpu_sample_no(stem: str) -> int | None:
    s = stem.replace("Calib_", "").replace("_Static", "")
    try:
        return int(s.split("_")[1])
    except (IndexError, ValueError):
        return None


# ── ядро: crop → resize → sync aug ───────────────────────────────────────────
def process_video(video: np.ndarray, mask: np.ndarray, roi: ROI,
                  aug_pipe: TransformPipeline):
    """Вернуть (frames_aug (T,Ho,Wo) float32, mask_aug (Ho,Wo) uint8)."""
    cropper = ObjectCropper(ObjectCropConfig(
        enabled=True, roi=roi, output_size=OUT_SIZE, pad_mode="reflect"))
    H, W = video.shape[1:]
    box = cropper.box_for(H, W, meta=None)

    frames_c = apply_crop_thw(video, box, "reflect")           # (T, h, w)
    frames_c = np.stack([resize_2d(f, OUT_SIZE, is_mask=False) for f in frames_c])
    mask_c = cropper.apply_mask(mask, box)                      # (Ho,Wo)

    frames_a, mask_a = aug_pipe.apply_temporal(frames_c, mask_c)
    return frames_a.astype(np.float32), mask_a.astype(np.uint8)


# ── прогон по датасету ───────────────────────────────────────────────────────
def run(domain: str, limit: int | None = None):
    rng = np.random.default_rng(SEED)
    out_dir = OUT_ROOT / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    if domain == "kaggle":
        vids = sorted(glob.glob(str(ROOT / "datasets/dataset_kaggle/data/*.mat")))
        roi = KAGGLE_ROI
    elif domain == "tpu":
        vids = sorted(glob.glob(str(ROOT / "datasets/dataset_tpu/*.mat")))
        roi = TPU_ROI
        base_m = tpu_base_mask()
    else:
        raise ValueError(domain)

    if limit:
        vids = vids[:limit]

    for p in vids:
        stem = os.path.splitext(os.path.basename(p))[0].replace(" ", "_")
        video = load_raw_video(p)
        if domain == "kaggle":
            vid_key = os.path.splitext(os.path.basename(p))[0]
            mask = kaggle_mask(vid_key)
            depth = ""
        else:
            mask = base_m
            no = tpu_sample_no(stem)
            depth = TPU_DEPTH_MM.get(no, "")

        for k in range(N_AUG):
            # свой seed на копию (для воспроизводимости)
            import random
            random.seed(SEED + k); np.random.seed(SEED + k)
            aug_pipe = TransformPipeline(AUG)
            frames_a, mask_a = process_video(video, mask, roi, aug_pipe)

            vpath = out_dir / f"{stem}__aug{k}.npy"
            mpath = out_dir / f"{stem}__aug{k}_mask.png"
            np.save(vpath, frames_a)
            Image.fromarray(mask_a).save(mpath)
            rows.append(dict(domain=domain, vid=stem, aug=k, depth_mm=depth,
                             shape=str(frames_a.shape), mb=round(vpath.stat().st_size/1e6, 1),
                             video=vpath.name, mask=mpath.name))
            print(f"{domain}/{stem} aug{k}: {frames_a.shape} "
                  f"{rows[-1]['mb']} MB  depth={depth}")
    return rows


def main(limit=None, domains=("kaggle", "tpu")):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for d in domains:
        all_rows += run(d, limit=limit)
    with open(OUT_ROOT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "vid", "aug", "depth_mm",
                                          "shape", "mb", "video", "mask"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nвсего файлов: {len(all_rows)} | manifest: {OUT_ROOT/'manifest.csv'}")


if __name__ == "__main__":
    main()
