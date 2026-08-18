"""Генерация масок для TPU/sample*.mat по табличным данным.

Геометрия блоков одинакова у всех образцов — шаблон из base_mask.png
(сырой кадр 240×320, как data/sample*.mat).

Выходы:
  labels/tpu_binary_masks/
    sample{n}.png                — 240×320, бинарные 0/255 (как kaggle_binary_masks)
  labels/tpu_sample_masks/
    sample{n}.png/.npy           — 240×320, серый = глубина (мм)
  datasets/dataset_tpu/labels/table_mask/
    Sample_{n}_Static.png/.npy   — 240×320 (сырой кадр sample*.mat)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = ROOT / "features_tpu"
BASE_MASK = ROOT / "datasets" / "dataset_tpu" / "base_mask.png"
OUT_DIR = ROOT / "datasets" / "dataset_tpu" / "labels" / "table_mask"
SAMPLE_MASK_DIR = ROOT / "labels" / "tpu_sample_masks"
BINARY_MASK_DIR = ROOT / "labels" / "tpu_binary_masks"
DATA_DIR = ROOT / "data"

H_RAW, W_RAW = 240, 320  # сырой кадр sample*.mat
TPU_CROP_TB = 60
ROI_PAD = 8  # небольшой паддинг вокруг всех дефектов

# глубина залегания h (L), мм, по номеру образца (Таблица 1)
DEPTH_MM: dict[int, float] = {
    1: 3.1, 2: 5.2, 3: 1.0, 4: 3.6, 5: 5.7, 6: 1.5, 7: 2.6, 8: 4.7,
    9: 0.5, 10: 4.2, 11: 2.1, 12: 4.0, 13: 6.1, 14: 1.9, 15: 2.1,
    16: 2.2, 17: 3.8, 18: 5.9, 19: 1.7, 20: 2.1, 21: 4.2,
}

DEPTH_MAX = max(DEPTH_MM.values())


def name_to_sample_no(name: str) -> int | None:
    stem = name.replace("Calib_", "").replace("_Static", "")
    # Sample_4 / sample4 / sample_4
    stem = stem.replace("sample", "Sample")
    if stem.lower().startswith("sample") and "_" not in stem[6:]:
        # sample4 → Sample_4
        digits = "".join(ch for ch in stem if ch.isdigit())
        return int(digits) if digits else None
    try:
        return int(stem.split("_")[1])
    except (IndexError, ValueError):
        return None


def load_raw_box_mask() -> np.ndarray:
    """Бинарный шаблон блоков в координатах сырого кадра 240×320."""
    from PIL import Image

    a = np.array(Image.open(BASE_MASK))
    raw = (a[..., 1] > 0) if a.ndim == 3 else (a > 0)
    return raw.astype(bool)


def to_feat_mask(raw: np.ndarray) -> np.ndarray:
    """Legacy alias: masks теперь в сыром 240×320 (без upscaling)."""
    return raw.astype(bool)


def defect_roi(raw_box: np.ndarray, pad: int = ROI_PAD) -> dict[str, int]:
    """ROI (x,y,w,h), покрывающий все дефекты + pad, в сырых координатах."""
    ys, xs = np.where(raw_box)
    if xs.size == 0:
        raise RuntimeError("base_mask: нет дефектных пикселей")
    H, W = raw_box.shape
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(W, int(xs.max()) + 1 + pad)
    y1 = min(H, int(ys.max()) + 1 + pad)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def available_sample_nos() -> list[int]:
    """Номера sample*.mat в data/ (ожидаем 3..21)."""
    nos: list[int] = []
    for p in sorted(DATA_DIR.glob("sample*.mat")):
        no = name_to_sample_no(p.stem)
        if no is not None and no in DEPTH_MM:
            nos.append(no)
    return sorted(set(nos))


def iter_thermo_names() -> list[str]:
    if FEATURE_DIR.is_dir():
        names = sorted(p.stem for p in FEATURE_DIR.glob("*.npy"))
        if names:
            return names
    names: list[str] = []
    for no in sorted(DEPTH_MM):
        if no in (1, 2):
            names.append(f"Calib_Sample_{no}")
        names.append(f"Sample_{no}_Static")
    return names


def _save_binary(out_dir: Path, name: str, box: np.ndarray) -> np.ndarray:
    """Как kaggle_binary_masks: фон 0, дефекты 255."""
    from PIL import Image

    png = np.zeros(box.shape, dtype=np.uint8)
    png[box] = 255
    Image.fromarray(png, mode="L").save(out_dir / f"{name}.png")
    return png


def _save_pair(
    out_dir: Path,
    name: str,
    box: np.ndarray,
    depth: float,
) -> tuple[np.ndarray, int]:
    from PIL import Image

    gray = int(round(255 * depth / DEPTH_MAX))
    png = np.zeros(box.shape, dtype=np.uint8)
    png[box] = gray
    Image.fromarray(png, mode="L").save(out_dir / f"{name}.png")

    depth_map = np.full(box.shape, np.nan, dtype=np.float32)
    depth_map[box] = depth
    np.save(out_dir / f"{name}.npy", depth_map)
    return png, gray


def main() -> None:
    from PIL import Image  # noqa: F401

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_MASK_DIR.mkdir(parents=True, exist_ok=True)
    BINARY_MASK_DIR.mkdir(parents=True, exist_ok=True)

    raw_box = load_raw_box_mask()
    feat_box = to_feat_mask(raw_box)
    roi = defect_roi(raw_box, ROI_PAD)
    print(f"base_mask {raw_box.shape}, defects={int(raw_box.sum())} px")
    print(f"ROI (pad={ROI_PAD}): {roi}")

    # --- 1) raw masks for data/sample*.mat (irt_data) --------------------------
    sample_nos = available_sample_nos()
    if not sample_nos:
        sample_nos = [n for n in range(3, 22) if n in DEPTH_MM]
        print("data/sample*.mat не найдены — пишу маски sample3..21 по таблице")
    else:
        print(f"data/sample*.mat → номера: {sample_nos}")

    sample_rows = []
    for no in sample_nos:
        if no < 3:
            continue
        depth = DEPTH_MM[no]
        name = f"sample{no}"
        _save_binary(BINARY_MASK_DIR, name, raw_box)
        png, gray = _save_pair(SAMPLE_MASK_DIR, name, raw_box, depth)
        sample_rows.append(
            {"name": name, "sample_no": no, "depth_mm": depth, "gray": gray, **roi}
        )
        print(f"{name:12s} №{no:<2d} depth={depth:>4} мм  binary→{BINARY_MASK_DIR.name}  depth-gray={gray}")

    with open(SAMPLE_MASK_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["name", "sample_no", "depth_mm", "gray", "x", "y", "w", "h"],
        )
        w.writeheader()
        w.writerows(sample_rows)

    with open(BINARY_MASK_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "sample_no", "x", "y", "w", "h"])
        w.writeheader()
        for r in sample_rows:
            w.writerow(
                {
                    "name": r["name"],
                    "sample_no": r["sample_no"],
                    "x": r["x"],
                    "y": r["y"],
                    "w": r["w"],
                    "h": r["h"],
                }
            )

    for dest in (SAMPLE_MASK_DIR, BINARY_MASK_DIR):
        with open(dest / "roi.yaml", "w", encoding="utf-8") as f:
            f.write("# общий ROI для всех sample*.mat (все дефекты + pad)\n")
            f.write(f"roi_padding: {ROI_PAD}\n")
            f.write("roi:\n")
            for k, v in roi.items():
                f.write(f"  {k}: {v}\n")

    # --- 2) thermo table_mask 240×320 -------------------------------------------
    thermo_names = iter_thermo_names()
    rows = []
    previews = []
    for name in thermo_names:
        no = name_to_sample_no(name)
        if no is None or no not in DEPTH_MM:
            print(f"skip {name}: нет глубины в таблице")
            continue
        depth = DEPTH_MM[no]
        png, gray = _save_pair(OUT_DIR, name, feat_box, depth)
        rows.append({"name": name, "sample_no": no, "depth_mm": depth, "gray": gray})
        previews.append((name, depth, png))
        print(f"{name:24s} №{no:<2d} depth={depth:>4} мм gray={gray}  [240x320]")

    with open(OUT_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "sample_no", "depth_mm", "gray"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["sample_no"]))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        previews.sort(key=lambda t: t[1])
        n = len(previews)
        cols = 5
        rows_n = (n + cols - 1) // cols
        fig, axs = plt.subplots(rows_n, cols, figsize=(3 * cols, 2.4 * rows_n))
        axs = np.atleast_1d(axs).ravel()
        for ax, (name, depth, png) in zip(axs, previews):
            ax.imshow(png, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{name}\n{depth} мм", fontsize=8)
            ax.axis("off")
        for ax in axs[n:]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "preview.png", dpi=90)
        print("preview.png сохранён")
    except Exception as e:
        print("preview skipped:", e)

    print(f"\nГотово: {len(sample_rows)} binary → {BINARY_MASK_DIR}")
    print(f"         {len(sample_rows)} depth-gray → {SAMPLE_MASK_DIR}")
    print(f"         {len(rows)} thermo → {OUT_DIR}")
    print(f"ROI для files_meta: {roi}")


if __name__ == "__main__":
    main()
