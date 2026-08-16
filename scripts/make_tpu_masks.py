"""Генерация статичных масок глубины для dataset_tpu по табличным данным.

В каждом образце все дефекты залегают на ОДНОЙ глубине (Таблица 1 из отчёта),
а геометрия блоков одинакова у всех образцов. Шаблон блоков берём из ручной
base_mask.png (нарисована на сыром кадре 240x320) и приводим её тем же
препроцессингом, что и фичи (обрезка 60 px сверху/снизу + resize к kaggle
256x320, см. main._tpu_crop_resize), после чего заливаем глубиной образца.

Выход (в datasets/dataset_tpu/labels/table_mask/):
  <name>.png   — grayscale 256x320, фон=0, блоки = глубина в стиле kaggle
                 (белый = самый глубокий образец в наборе); только для просмотра.
  <name>.npy   — float32 256x320 карта глубины в мм, фон = NaN (для регрессии).
  manifest.csv — name, sample_no, depth_mm, gray.
  preview.png  — сетка всех масок.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = ROOT / "features_tpu"
BASE_MASK = ROOT / "datasets" / "dataset_tpu" / "base_mask.png"
OUT_DIR = ROOT / "datasets" / "dataset_tpu" / "labels" / "table_mask"

H, W = 256, 320          # размер kaggle-фич
TPU_CROP_TB = 60         # обрезка сверху/снизу, как в main._tpu_crop_resize

# --- глубина залегания h (L), мм, по номеру образца из Таблицы 1 ---------------
DEPTH_MM: dict[int, float] = {
    1: 3.1, 2: 5.2, 3: 1.0, 4: 3.6, 5: 5.7, 6: 1.5, 7: 2.6, 8: 4.7,
    9: 0.5, 10: 4.2, 11: 2.1, 12: 4.0, 13: 6.1, 14: 1.9, 15: 2.1,
    16: 2.2, 17: 3.8, 18: 5.9, 19: 1.7, 20: 2.1, 21: 4.2,
}

# сопоставление имени фичи -> номер образца
#   Calib_Sample_1/2  -> №1/№2 ;  Sample_N -> №N
def name_to_sample_no(name: str) -> int | None:
    stem = name.replace("Calib_", "").replace("_Static", "")
    # stem вида "Sample_4"
    try:
        return int(stem.split("_")[1])
    except (IndexError, ValueError):
        return None


DEPTH_MAX = max(DEPTH_MM.values())  # 6.1 -> белый


def build_box_mask() -> np.ndarray:
    """Шаблон блоков из base_mask.png, приведённый к размеру фич (256x320).

    base_mask нарисована на сыром кадре 240x320 → тот же трансформ, что и фичи:
    обрезка строк [60 : H-60] + resize к kaggle (nearest, чтобы маска осталась
    бинарной).
    """
    import cv2
    from PIL import Image

    a = np.array(Image.open(BASE_MASK))
    raw = (a[..., 1] > 0) if a.ndim == 3 else (a > 0)   # блоки помечены цветом
    crop = raw[TPU_CROP_TB: raw.shape[0] - TPU_CROP_TB, :]
    tpl = cv2.resize(crop.astype(np.uint8), (W, H),
                     interpolation=cv2.INTER_NEAREST).astype(bool)
    return tpl


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    box = build_box_mask()

    from PIL import Image

    names = sorted(p.stem for p in FEATURE_DIR.glob("*.npy"))
    rows = []
    previews = []
    for name in names:
        no = name_to_sample_no(name)
        if no is None or no not in DEPTH_MM:
            print(f"skip {name}: нет глубины в таблице")
            continue
        depth = DEPTH_MM[no]
        gray = int(round(255 * depth / DEPTH_MAX))

        png = np.zeros((H, W), dtype=np.uint8)
        png[box] = gray
        Image.fromarray(png, mode="L").save(OUT_DIR / f"{name}.png")

        depth_map = np.full((H, W), np.nan, dtype=np.float32)
        depth_map[box] = depth
        np.save(OUT_DIR / f"{name}.npy", depth_map)

        rows.append({"name": name, "sample_no": no, "depth_mm": depth, "gray": gray})
        previews.append((name, depth, png))
        print(f"{name:24s} №{no:<2d} depth={depth:>4} мм gray={gray}")

    with open(OUT_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "sample_no", "depth_mm", "gray"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["sample_no"]))

    # превью-сетка
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        previews.sort(key=lambda t: t[1])
        n = len(previews)
        cols = 5
        rows_n = (n + cols - 1) // cols
        fig, axs = plt.subplots(rows_n, cols, figsize=(3 * cols, 2.4 * rows_n))
        for ax, (name, depth, png) in zip(axs.flat, previews):
            ax.imshow(png, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{name}\n{depth} мм", fontsize=8)
            ax.axis("off")
        for ax in axs.flat[n:]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "preview.png", dpi=90)
        print("preview.png сохранён")
    except Exception as e:
        print("preview skipped:", e)

    print(f"\nГотово: {len(rows)} масок в {OUT_DIR}")


if __name__ == "__main__":
    main()
