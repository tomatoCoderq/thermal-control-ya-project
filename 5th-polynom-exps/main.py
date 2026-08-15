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

import numpy as np
import scipy.io as sio
from scipy import ndimage
from PIL import Image

import torch
from torch.utils.data import DataLoader

from config import CFG
from datasets import DatasetAnyChannels, Modes
from losses import Losses
from model import SmallCNN
from engine import fit

# ── Настройки запуска ────────────────────────────────────────────────────────
MODE = Modes.p5_d1_d2          # набор каналов: p5 | p5_d1 | p5_d1_d2


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
    raise ValueError(f"неизвестный loss: {name}")


# ── 5. Сборка и запуск ──────────────────────────────────────────────────────
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

    model = SmallCNN(in_channels=n_channels(MODE)).to(device)
    loss_fn = make_loss(train_idx).to(device)     # веса лосса — на то же устройство
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.train.learning_rate)

    tag = f"{MODE.value}_{CFG.train.loss_name}"
    run_dir, history, result = fit(train_ld, model, loss_fn, optimizer,
                                   device, val_loader=test_ld, tag=tag)
    print("готово:", result, "| артефакты:", run_dir)


if __name__ == "__main__":
    main()
