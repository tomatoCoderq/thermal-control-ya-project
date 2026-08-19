"""Точка входа: сборка данных → модель → обучение с логированием.

Запуск:
    cd 5th-polynom-exps && python main.py

Что делает:
  1. предвычисляет TSR-признаки (полином 5°) в CFG.paths.feature_dir (кэш, один раз);
  2. строит кропы по маскам и метки глубины;
  3. делит по видео (без утечки), считает поканальную нормировку по train;
  4. собирает DataLoader'ы, модель SmallCNN, лосс (из CFG.train.loss_name);
  5. запускает engine.fit — обучение с логами в runs/<timestamp>_<tag>/.
"""
import glob
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import ndimage
from PIL import Image

import timm
import torch
from torch.utils.data import DataLoader

from config import CFG
from datasets import DatasetAnyChannels, Modes
from losses import Losses
from model import SmallCNN
from engine import fit

# ── Настройки запуска ────────────────────────────────────────────────────────
MODE = Modes.p5_d1          # набор каналов: p5 | p5_d1 | p5_d1_d2


# ── 1. Предвычисление TSR-признаков (полином 5°) ────────────────────────────
def list_videos():
    ids = [os.path.splitext(os.path.basename(p))[0]
           for p in glob.glob(os.path.join(CFG.paths.data_dir, "*.mat"))]
    ids = sorted(ids)
    return ids if CFG.train.max_videos is None else ids[:CFG.train.max_videos]


def tsr_coeffs(path, deg=CFG.tsr.poly_degree):
    m = sio.loadmat(path)
    key = next(k for k in ("imageArray", "data", "IMAGES")
               if k in m and np.asarray(m[k]).ndim == 3)
    X = np.transpose(np.asarray(m[key]).astype(np.float32), (2, 0, 1))  # (n,H,W)
    n, H, W = X.shape
    peak = int(np.argmax(X.reshape(n, -1).mean(1)))          # кадр пика нагрева
    base = X[:max(1, peak // 4)].mean(0)                     # базовая линия
    dT = np.clip(X[peak:] - base[None], 1e-3, None)          # ΔT>0 (остывание)
    tc = (np.arange(dT.shape[0]) + 1).astype(np.float32)
    logt = np.log(tc)
    coef = np.polyfit(logt, np.log(dT).reshape(len(tc), -1), deg)
    return coef.reshape(deg + 1, H, W).astype(np.float32)    # (deg+1,H,W)


def precompute_features():
    os.makedirs(CFG.paths.feature_dir, exist_ok=True)
    for i, vid in enumerate(list_videos(), 1):
        out = CFG.paths.feature_dir / f"{vid}.npy"
        if out.exists():
            continue
        np.save(out, tsr_coeffs(os.path.join(CFG.paths.data_dir, f"{vid}.mat")))
        print(f"[{i}] features {vid}")


# ── 2. Кропы по маскам ──────────────────────────────────────────────────────
def load_mask_cls(vid):
    im = np.array(Image.open(CFG.paths.mask_dir / f"{vid}.png"))
    cls = np.zeros_like(im, dtype=np.uint8)
    for g, c in CFG.classes.gray2cls.items():
        cls[im == g] = c
    return cls


def build_index():
    s = CFG.crop.crop
    idx = []
    for vid in list_videos():
        cls = load_mask_cls(vid)
        H, W = cls.shape
        for c in range(1, CFG.classes.n_classes):        # дефектные кропы
            lbl, n = ndimage.label(cls == c)
            for k in range(1, n + 1):
                rr, cc = ndimage.center_of_mass(lbl == k)
                r0 = int(np.clip(rr - s // 2, 0, H - s))
                c0 = int(np.clip(cc - s // 2, 0, W - s))
                idx.append((vid, r0, c0, c))
        bg = cls[s:H - s, s:W - s] == 0                  # фоновые кропы
        ys, xs = np.where(bg)
        if len(ys):
            sel = np.random.choice(len(ys),
                                   min(CFG.crop.n_bg_per_video, len(ys)),
                                   replace=False)
            for j in sel:
                idx.append((vid, int(ys[j] + s // 2), int(xs[j] + s // 2), 0))
    return idx


# ── 3. Сплит по видео + нормировка по train ─────────────────────────────────
def split_by_video(index):
    rng = np.random.default_rng(CFG.train.seed)
    vids = np.array(sorted({r[0] for r in index}))
    perm = rng.permutation(vids)
    test_vids = set(perm[:CFG.train.n_test_videos])
    train = [r for r in index if r[0] not in test_vids]
    test = [r for r in index if r[0] in test_vids]
    return train, test


def compute_norm(index):
    """μ/σ по 6 каналам коэффициентов (нормировка идёт до сборки производных)."""
    s = CFG.crop.crop
    cache = {v: np.load(CFG.paths.feature_dir / f"{v}.npy")
             for v in sorted({r[0] for r in index})}
    stk = np.stack([cache[v][:, r0:r0 + s, c0:c0 + s] for (v, r0, c0, _) in index])
    return stk.mean(axis=(0, 2, 3)), stk.std(axis=(0, 2, 3)) + 1e-6


def n_channels(mode: Modes) -> int:
    b = CFG.tsr.n_channels                    # 6 = poly_degree + 1
    return {Modes.p5: b, Modes.p5_d1: b + (b - 1),
            Modes.p5_d1_d2: b + (b - 1) + (b - 2)}[mode]


# ── 4. Лосс из конфига ──────────────────────────────────────────────────────
def make_loss(train_index):
    name = CFG.train.loss_name
    if name == "ce":
        return Losses.ce()
    if name == "label_smooth":
        return Losses.label_smooth()
    if name == "weighted_ce":
        freq = np.array([sum(1 for *_, l in train_index if l == k)
                         for k in range(CFG.classes.n_classes)], dtype=np.float64)
        w = freq.sum() / (freq + 1e-6)
        return Losses.weighted_ce((w / w.mean()).tolist())
    if name == "cost_sensitive_ce":
        return Losses.cost_sensitive_ce(CFG.classes.n_classes, kind="linear", alpha=1.0)
    if name == "qwk":
        return Losses.qwk(CFG.classes.n_classes)
    raise ValueError(f"неизвестный loss: {name}")


# ── 5. Инференс на dataset_tpu (масок нет → только предсказанные классы) ─────
TPU_FEAT_DIR = CFG.paths.feature_dir.parent / "features_tpu"
TPU_CROP_TB = 60          # обрезаем сверху и снизу по столько пикселей (фон)


def list_tpu_videos():
    return sorted(glob.glob(os.path.join(CFG.paths.tpu_dir, "*.mat")))


def _kaggle_hw():
    """Размер (H, W) кадра kaggle — под него приводим TPU (по кэшу признаков)."""
    files = sorted(glob.glob(str(CFG.paths.feature_dir / "*.npy")))
    return tuple(np.load(files[0]).shape[1:]) if files else (256, 320)


def _tpu_crop_resize(a, Hk, Wk):
    """Обрезать фон сверху/снизу (TPU_CROP_TB) и сжать в размер kaggle (Hk×Wk).

    Работает и для кадра (H,W), и для карты каналов (C,H,W).
    """
    import cv2
    a = a[..., TPU_CROP_TB:a.shape[-2] - TPU_CROP_TB, :]      # обрезка строк
    if a.ndim == 2:
        return cv2.resize(a, (Wk, Hk), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.stack([cv2.resize(a[i], (Wk, Hk), interpolation=cv2.INTER_LINEAR)
                     for i in range(a.shape[0])]).astype(np.float32)


def _tpu_id(path):
    return os.path.splitext(os.path.basename(path))[0].replace(" ", "_")


def precompute_tpu_features():
    """TSR (полином 5°) → обрезка фона сверху/снизу → сжатие в размер kaggle → кэш."""
    os.makedirs(TPU_FEAT_DIR, exist_ok=True)
    Hk, Wk = _kaggle_hw()
    # инвалидация устаревшего кэша (если размер уже не совпадает с целевым)
    existing = sorted(TPU_FEAT_DIR.glob("*.npy"))
    if existing and tuple(np.load(existing[0]).shape[1:]) != (Hk, Wk):
        for f in existing:
            f.unlink()
    for p in list_tpu_videos():
        out = TPU_FEAT_DIR / f"{_tpu_id(p)}.npy"
        if out.exists():
            continue
        coef = tsr_coeffs(p)                              # (6, 240, 320)
        coef = _tpu_crop_resize(coef, Hk, Wk)            # → (6, Hk, Wk) как kaggle
        np.save(out, coef)
        print("TPU features:", _tpu_id(p))


def _build_channels_np(coef6, mode):
    """Как DatasetAnyChannels._build_channels, но на numpy-массиве (C,H,W)."""
    if mode == Modes.p5_d1_d2:
        return np.concatenate([coef6, np.diff(coef6, n=1, axis=0),
                               np.diff(coef6, n=2, axis=0)], axis=0)
    if mode == Modes.p5_d1:
        return np.concatenate([coef6, np.diff(coef6, n=1, axis=0)], axis=0)
    return coef6


def predict_frame(model, coef, mean, std, mode, device, stride=None):
    """Скользящее окно CROP×CROP по кадру → карта предсказанных классов (по тайлам)."""
    s = CFG.crop.crop
    stride = stride or s
    H, W = coef.shape[1:]
    m = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
    sd = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
    rows = list(range(0, H - s + 1, stride))
    cols = list(range(0, W - s + 1, stride))
    tiles = []
    for r0 in rows:
        for c0 in cols:
            crop = (coef[:, r0:r0 + s, c0:c0 + s] - m) / sd   # нормировка по train
            tiles.append(_build_channels_np(crop, mode))
    X = torch.from_numpy(np.stack(tiles)).float().to(device)
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=1).cpu().numpy()
    return pred.reshape(len(rows), len(cols))                 # карта классов по тайлам


def detect_boxes(cls_map, stride, crop, min_tiles=1):
    """Из карты классов по тайлам → боксы дефектов (bbox в пикселях + класс).

    Соседние тайлы одного класса (≠ фон) объединяются в связную область.
    """
    boxes = []
    for c in range(1, CFG.classes.n_classes):                # 0 = фон, пропускаем
        lbl, n = ndimage.label(cls_map == c)
        for k in range(1, n + 1):
            ys, xs = np.where(lbl == k)
            if len(ys) < min_tiles:
                continue
            x0, y0 = int(xs.min()) * stride, int(ys.min()) * stride
            x1 = int(xs.max()) * stride + crop
            y1 = int(ys.max()) * stride + crop
            boxes.append(dict(bbox=(x0, y0, x1, y1), cls=c, n_tiles=int(len(ys))))
    return boxes


def test_tpu(model, mean, std, mode, device, stride=None, limit=None):
    """Инференс на dataset_tpu: полином 5° → модель → карта классов + боксы дефектов.

    Масок нет → метрик нет. Возврат: {образец: {'map': карта тайлов, 'boxes': [...]}}.
    """
    vids = list_tpu_videos()
    if not vids:
        print("dataset_tpu: .mat не найдены в", CFG.paths.tpu_dir)
        return {}
    precompute_tpu_features()
    if limit:
        vids = vids[:limit]
    s = CFG.crop.crop
    stride = stride or s
    names = CFG.classes.class_names
    results = {}
    print(f"\n=== инференс на dataset_tpu ({len(vids)} образцов, масок нет) ===")
    for p in vids:
        coef = np.load(TPU_FEAT_DIR / f"{_tpu_id(p)}.npy")    # готовая TSR-аппроксимация
        cmap = predict_frame(model, coef, mean, std, mode, device, stride)
        boxes = detect_boxes(cmap, stride, s)
        by_cls = {names[b["cls"]]: sum(1 for x in boxes if x["cls"] == b["cls"]) for b in boxes}
        print(f"{_tpu_id(p):26} дефектов(боксов): {len(boxes):2d} | по классам: {by_cls}")
        results[_tpu_id(p)] = dict(map=cmap, boxes=boxes)
    return results


def _tpu_frame(path):
    """Кадр остывания как фон — обрезан и сжат так же, как признаки (боксы совпадут)."""
    m = sio.loadmat(path)
    key = next(k for k in ("data", "imageArray", "IMAGES")
               if k in m and np.asarray(m[k]).ndim == 3)
    X = np.transpose(np.asarray(m[key]).astype(np.float32), (2, 0, 1))
    peak = int(np.argmax(X.reshape(X.shape[0], -1).mean(1)))
    frame = X[min(peak + 200, X.shape[0] - 1)]
    Hk, Wk = _kaggle_hw()
    return _tpu_crop_resize(frame, Hk, Wk)


def visualize_tpu(results, out_file=None, max_samples=6):
    """До 6 образцов в ОДНОМ файле: кадр + боксы дефектов с подписью класса."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out_file = Path(out_file or (CFG.paths.log_dir / "tpu_pred.png"))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    n_cls = CFG.classes.n_classes
    cmap = plt.get_cmap("tab10", n_cls)
    names = CFG.classes.class_names
    paths = {_tpu_id(p): p for p in list_tpu_videos()}

    items = list(results.items())[:max_samples]
    ncol = 3
    nrow = (len(items) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))
    axs = np.atleast_1d(axs).ravel()
    for ax, (stem, res) in zip(axs, items):
        ax.imshow(_tpu_frame(paths[stem]), cmap="inferno")
        ax.set_title(f"{stem}  (дефектов: {len(res['boxes'])})", fontsize=9)
        ax.axis("off")
        for b in res["boxes"]:
            x0, y0, x1, y1 = b["bbox"]
            col = cmap(b["cls"])
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, edgecolor=col, linewidth=2))
            ax.text(x0 + 1, max(0, y0 - 2), names[b["cls"]], color=col,
                    fontsize=8, weight="bold", va="bottom")
    for ax in axs[len(items):]:
        ax.axis("off")
    fig.suptitle("dataset_tpu — обнаруженные дефекты (бокс + класс)", y=1.02)
    plt.tight_layout()
    fig.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("боксы TPU сохранены в", out_file)


# ── 6. Сборка и запуск ──────────────────────────────────────────────────────
def main():
    torch.manual_seed(CFG.train.seed)
    np.random.seed(CFG.train.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    print("device:", device, "| mode:", MODE.value, "| loss:", CFG.train.loss_name)
    precompute_features()

    index = build_index()
    train_idx, test_idx = split_by_video(index)
    print(f"кропов: {len(index)} | train: {len(train_idx)} | test: {len(test_idx)}")

    mean, std = compute_norm(train_idx)
    train_ds = DatasetAnyChannels(train_idx, mean, std, MODE, train=True)
    test_ds = DatasetAnyChannels(test_idx, mean, std, MODE, train=False)
    train_ld = DataLoader(train_ds, batch_size=CFG.train.batch_size, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=CFG.train.batch_size, shuffle=False)

    # model = SmallCNN(in_channels=n_channels(MODE)).to(device)
    # return timm.create_model(name, in_chans=in_ch, num_classes=n_cls,
    #                              pretrained=pretrained)
    model = timm.create_model(CFG.model.name, pretrained=False, in_chans=n_channels(MODE),
                              num_classes=CFG.classes.n_classes).to(device)
    loss_fn = make_loss(train_idx).to(device)     # веса лосса — на то же устройство
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.train.learning_rate)

    tag = f"{MODE.value}_{CFG.train.loss_name}"
    run_dir, history, result = fit(train_ld, model, loss_fn, optimizer,
                                   device, val_loader=test_ld, tag=tag)
    print("готово:", result, "| артефакты:", run_dir)

    # Инференс на dataset_tpu: те же TSR-признаки (полином 5°) → модель.
    # Масок нет → метрики не считаем, выводим классы и сохраняем карты.
    tpu_maps = test_tpu(model, mean, std, MODE, device)
    visualize_tpu(tpu_maps)


if __name__ == "__main__":
    main()
